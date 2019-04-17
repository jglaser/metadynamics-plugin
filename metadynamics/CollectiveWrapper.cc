/*! \file CollectiveWrapper.cc
    \brief Wraps a CollectiveVariable around a regular ForceCompute
 */
#include "CollectiveWrapper.h"

#ifdef ENABLE_CUDA
#include "WellTemperedEnsemble.cuh"
#endif

namespace py = pybind11;

CollectiveWrapper::CollectiveWrapper(std::shared_ptr<SystemDefinition> sysdef,
                                     std::shared_ptr<ForceCompute> fc,
                                       const std::string& name)
    : CollectiveVariable(sysdef, name), m_fc(fc), m_energy(0.0)
    {
    #ifdef ENABLE_CUDA
    if (m_exec_conf->exec_mode == ExecutionConfiguration::GPU)
        {
        GlobalArray<Scalar> sum(1,m_exec_conf);
        m_sum.swap(sum);
        TAG_ALLOCATION(m_sum);
        }

    cudaDeviceProp dev_prop = m_exec_conf->dev_prop;
    m_tuner_reduce.reset(new Autotuner(dev_prop.warpSize, dev_prop.maxThreadsPerBlock, dev_prop.warpSize, 5, 100000, name+"_reduce", this->m_exec_conf));
    m_tuner_scale.reset(new Autotuner(dev_prop.warpSize, dev_prop.maxThreadsPerBlock, dev_prop.warpSize, 5, 100000, name+"_scale", this->m_exec_conf));
    #endif
    }

void CollectiveWrapper::computeCV(unsigned int timestep)
    {
    // compute the force (note this only computes once per time step)
    m_fc->compute(timestep);

    if (m_prof)
        m_prof->push(m_exec_conf,"Collective wrap");

    #ifdef ENABLE_CUDA
    if (m_exec_conf->exec_mode == ExecutionConfiguration::GPU)
        {
        computeCVGPU(timestep);
        }
    else
    #endif
        {
        ArrayHandle<Scalar4> h_force(m_fc->getForceArray(), access_location::host, access_mode::readwrite);

        unsigned int N = m_pdata->getN();

        m_energy = Scalar(0.0);

        // Sum up potential energy
        for (unsigned int i = 0; i < N; ++i)
            {
            m_energy += h_force.data[i].w;
            }
        }

    Scalar external_energy = m_fc->getExternalEnergy();
    m_energy += external_energy;

#ifdef ENABLE_MPI
    // reduce Fourier modes on on all processors
    if (m_pdata->getDomainDecomposition())
        {
        MPI_Allreduce(MPI_IN_PLACE, &m_energy, 1, MPI_HOOMD_SCALAR, MPI_SUM, m_exec_conf->getMPICommunicator());
        }
#endif

    if (m_prof)
        m_prof->pop();
    }

#ifdef ENABLE_CUDA
void CollectiveWrapper::computeCVGPU(unsigned int timestep)
    {
    ArrayHandle<Scalar4> d_force(m_fc->getForceArray(), access_location::device, access_mode::read);

    // maximum size, per GPU, for a block_size of one
    unsigned int scratch_size = (m_pdata->getN()+m_pdata->getNGhosts()+1);
    ScopedAllocation<Scalar> d_scratch(m_exec_conf->getCachedAllocatorManaged(), scratch_size*m_exec_conf->getNumActiveGPUs());

        {
        ArrayHandle<Scalar> d_sum(m_sum, access_location::device, access_mode::overwrite);

        // reset sum
        cudaMemsetAsync(d_sum.data, 0, sizeof(Scalar));

        m_exec_conf->beginMultiGPU();
        m_tuner_reduce->begin();

        gpu_reduce_potential_energy(d_scratch.data,
            d_force.data,
            m_pdata->getGPUPartition(),
            m_pdata->getNGhosts(),
            scratch_size,
            d_sum.data,
            false,
            m_tuner_reduce->getParam());
        if (m_exec_conf->isCUDAErrorCheckingEnabled()) CHECK_CUDA_ERROR();

        m_tuner_reduce->end();
        m_exec_conf->endMultiGPU();
        }
    
    ArrayHandle<Scalar> h_sum(m_sum, access_location::host, access_mode::read);
    m_energy = *h_sum.data;
    }

void CollectiveWrapper::computeBiasForcesGPU(unsigned int timestep)
    {
    ArrayHandle<Scalar4> d_force(m_fc->getForceArray(), access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar4> d_torque(m_fc->getTorqueArray(), access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar> d_virial(m_fc->getVirialArray(), access_location::device, access_mode::readwrite);

    unsigned int pitch = m_fc->getVirialArray().getPitch();
    Scalar fac = m_bias;

    m_exec_conf->beginMultiGPU();
    m_tuner_scale->begin();
    
    gpu_scale_netforce(d_force.data,
        d_torque.data,
        d_virial.data,
        pitch,
        fac,
        m_pdata->getGPUPartition(),
        m_pdata->getNGhosts(),
        m_tuner_scale->getParam());

    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();

    m_tuner_scale->end();
    m_exec_conf->endMultiGPU();
    }
#endif

void CollectiveWrapper::computeBiasForces(unsigned int timestep)
    {
    // compute the force (note this only computes once per time step)
    m_fc->compute(timestep);

    if (m_prof)
        m_prof->push("Collective wrap");

    Scalar fac = m_bias;

    #ifdef ENABLE_CUDA
    if (m_exec_conf->exec_mode == ExecutionConfiguration::GPU)
        {
        computeBiasForcesGPU(timestep);
        }
    else
    #endif
        {
        ArrayHandle<Scalar4> h_force(m_fc->getForceArray(), access_location::host, access_mode::readwrite);
        ArrayHandle<Scalar4> h_torque(m_fc->getTorqueArray(), access_location::host, access_mode::readwrite);
        ArrayHandle<Scalar> h_virial(m_fc->getVirialArray(), access_location::host, access_mode::readwrite);

        unsigned int N = m_pdata->getN()+m_pdata->getNGhosts();
        unsigned int pitch = m_fc->getVirialArray().getPitch();

        // apply bias factor
        for (unsigned int i = 0; i < N; ++i)
            {
            h_force.data[i].x *= fac;
            h_force.data[i].y *= fac;
            h_force.data[i].z *= fac;

            h_torque.data[i].x *= fac;
            h_torque.data[i].y *= fac;
            h_torque.data[i].z *= fac;
            h_torque.data[i].w *= fac;

            h_virial.data[i + 0*pitch] *= fac;
            h_virial.data[i + 1*pitch] *= fac;
            h_virial.data[i + 2*pitch] *= fac;
            h_virial.data[i + 3*pitch] *= fac;
            h_virial.data[i + 4*pitch] *= fac;
            h_virial.data[i + 5*pitch] *= fac;
            }
        }

    if (m_prof)
        m_prof->pop();
    }


void export_CollectiveWrapper(py::module& m)
    {
    py::class_<CollectiveWrapper, std::shared_ptr<CollectiveWrapper> > (m, "CollectiveWrapper", py::base<CollectiveVariable>() )
        .def(py::init< std::shared_ptr<SystemDefinition>, std::shared_ptr<ForceCompute>, const std::string& > ())
        ;

    }
