/*! \file WellTemperedEnsemble.cc
    \brief Implements the potential energy as a collective variable (well-tempered ensemble)
 */
#include "WellTemperedEnsemble.h"

#ifdef ENABLE_HIP
#include "WellTemperedEnsemble.cuh"
#endif

namespace py = pybind11;

WellTemperedEnsemble::WellTemperedEnsemble(std::shared_ptr<SystemDefinition> sysdef,
                                       const std::string& name)
    : CollectiveVariable(sysdef, name), m_pe(0.0), m_log_name("cv_potential_energy")
    {
    #ifdef ENABLE_HIP
    if (m_exec_conf->exec_mode == ExecutionConfiguration::GPU)
        {
        GlobalArray<Scalar> sum(1,m_exec_conf);
        m_sum.swap(sum);
        TAG_ALLOCATION(m_sum);
        }

    hipDeviceProp_t dev_prop = m_exec_conf->dev_prop;
    // power of two blocks sizes for reduction
    std::vector<unsigned int> valid_params;
    for (unsigned int block_size = dev_prop.warpSize; block_size <= (unsigned int) dev_prop.maxThreadsPerBlock; block_size *= 2)
        {
        valid_params.push_back(block_size);
        }

    m_tuner_reduce.reset(new Autotuner(valid_params, 5, 100000, name+"_reduce", this->m_exec_conf));
    m_tuner_scale.reset(new Autotuner(dev_prop.warpSize, dev_prop.maxThreadsPerBlock, dev_prop.warpSize, 5, 100000, "wte_scale", this->m_exec_conf));
    #endif
    }

void WellTemperedEnsemble::computeCV(unsigned int timestep)
    {
    if (m_prof)
        m_prof->push(m_exec_conf,"Well-Tempered Ensemble");

    #ifdef ENABLE_HIP
    if (m_exec_conf->exec_mode == ExecutionConfiguration::GPU)
        {
        computeCVGPU(timestep);
        }
    else
    #endif
        {
        ArrayHandle<Scalar4> h_net_force(m_pdata->getNetForce(), access_location::host, access_mode::read);

        unsigned int N = m_pdata->getN();

        m_pe = Scalar(0.0);

        // Sum up potential energy
        for (unsigned int i = 0; i < N; ++i)
            {
            m_pe += h_net_force.data[i].w;
            }
        }

    m_pe += m_pdata->getExternalEnergy();

#ifdef ENABLE_MPI
    // reduce Fourier modes on on all processors
    if (m_pdata->getDomainDecomposition())
        {
        MPI_Allreduce(MPI_IN_PLACE, &m_pe, 1, MPI_HOOMD_SCALAR, MPI_SUM, m_exec_conf->getMPICommunicator());
        }
#endif

    if (m_prof)
        m_prof->pop();
    }

#ifdef ENABLE_HIP
void WellTemperedEnsemble::computeCVGPU(unsigned int timestep)
    {
    ArrayHandle<Scalar4> d_net_force(m_pdata->getNetForce(), access_location::device, access_mode::read);

    // maximum size, per GPU, for a block_size of one
    unsigned int scratch_size = (m_pdata->getN()+1);
    ScopedAllocation<Scalar> d_scratch(m_exec_conf->getCachedAllocatorManaged(), scratch_size*m_exec_conf->getNumActiveGPUs());

        {
        ArrayHandle<Scalar> d_sum(m_sum, access_location::device, access_mode::overwrite);

        // reset sum
        cudaMemsetAsync(d_sum.data, 0, sizeof(Scalar));

        m_exec_conf->beginMultiGPU();
        m_tuner_reduce->begin();

        gpu_reduce_potential_energy(d_scratch.data,
            d_net_force.data,
            m_pdata->getGPUPartition(),
            0, // nghost
            scratch_size,
            d_sum.data,
            false,
            m_tuner_reduce->getParam());
        if (m_exec_conf->isCUDAErrorCheckingEnabled()) CHECK_CUDA_ERROR();

        m_tuner_reduce->end();
        m_exec_conf->endMultiGPU();
        }
    
    ArrayHandle<Scalar> h_sum(m_sum, access_location::host, access_mode::read);
    m_pe = *h_sum.data;
    }

void WellTemperedEnsemble::computeBiasForcesGPU(unsigned int timestep)
    {
    ArrayHandle<Scalar4> d_net_force(m_pdata->getNetForce(), access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar4> d_net_torque(m_pdata->getNetTorqueArray(), access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar> d_net_virial(m_pdata->getNetVirial(), access_location::device, access_mode::readwrite);

    unsigned int pitch = m_pdata->getNetVirial().getPitch();
    Scalar fac = Scalar(1.0)+m_bias;

    m_exec_conf->beginMultiGPU();
    m_tuner_scale->begin();
    
    gpu_scale_netforce(d_net_force.data,
        d_net_torque.data,
        d_net_virial.data,
        pitch,
        fac,
        m_pdata->getGPUPartition(),
        0, // nghost
        m_tuner_scale->getParam());

    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();

    m_tuner_scale->end();
    m_exec_conf->endMultiGPU();
    }
#endif

void WellTemperedEnsemble::computeBiasForces(unsigned int timestep)
    {
    if (m_prof)
        m_prof->push("Well-Tempered Ensemble");

    Scalar fac = Scalar(1.0)+m_bias;

    #ifdef ENABLE_HIP
    if (m_exec_conf->exec_mode == ExecutionConfiguration::GPU)
        {
        computeBiasForcesGPU(timestep);
        }
    else
    #endif
        {
        // Note: this Compute operates directly on the net force, therefore it needs to be called
        // after every other force
        ArrayHandle<Scalar4> h_net_force(m_pdata->getNetForce(), access_location::host, access_mode::readwrite);
        ArrayHandle<Scalar4> h_net_torque(m_pdata->getNetTorqueArray(), access_location::host, access_mode::readwrite);
        ArrayHandle<Scalar> h_net_virial(m_pdata->getNetVirial(), access_location::host, access_mode::readwrite);

        unsigned int N = m_pdata->getN();
        unsigned int pitch = m_pdata->getNetVirial().getPitch();

        // apply bias factor
        for (unsigned int i = 0; i < N; ++i)
            {
            h_net_force.data[i].x *= fac;
            h_net_force.data[i].y *= fac;
            h_net_force.data[i].z *= fac;

            h_net_torque.data[i].x *= fac;
            h_net_torque.data[i].y *= fac;
            h_net_torque.data[i].z *= fac;
            h_net_torque.data[i].w *= fac;

            h_net_virial.data[i + 0*pitch] *= fac;
            h_net_virial.data[i + 1*pitch] *= fac;
            h_net_virial.data[i + 2*pitch] *= fac;
            h_net_virial.data[i + 3*pitch] *= fac;
            h_net_virial.data[i + 4*pitch] *= fac;
            h_net_virial.data[i + 5*pitch] *= fac;
            }
        }

    for (unsigned int i = 0; i < 6; ++i)
        {
        Scalar v = m_pdata->getExternalVirial(i);
        m_pdata->setExternalVirial(i,fac*v);
        }

    if (m_prof)
        m_prof->pop();
    }


void export_WellTemperedEnsemble(py::module& m)
    {
    py::class_<WellTemperedEnsemble, std::shared_ptr<WellTemperedEnsemble> > (m, "WellTemperedEnsemble", py::base<CollectiveVariable>() )
        .def(py::init< std::shared_ptr<SystemDefinition>, const std::string& > ())
        ;

    }
