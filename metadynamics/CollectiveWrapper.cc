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
        GPUFlags<Scalar> sum(m_exec_conf);
        sum.resetFlags(0);
        m_sum.swap(sum);
        }
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

    unsigned int block_size = 512;
    unsigned int n_blocks = m_pdata->getN() / block_size + 1;

    ScopedAllocation<Scalar> d_scratch(m_exec_conf->getCachedAllocator(), n_blocks);

    gpu_reduce_potential_energy(d_scratch.data,
        d_force.data,
        m_pdata->getN()+m_pdata->getNGhosts(),
        m_sum.getDeviceFlags(),
        n_blocks,
        block_size,
        false);

    if (m_exec_conf->isCUDAErrorCheckingEnabled()) CHECK_CUDA_ERROR();

    m_energy = m_sum.readFlags();
    }

void CollectiveWrapper::computeBiasForcesGPU(unsigned int timestep)
    {
    ArrayHandle<Scalar4> d_force(m_fc->getForceArray(), access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar4> d_torque(m_fc->getTorqueArray(), access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar> d_virial(m_fc->getVirialArray(), access_location::device, access_mode::readwrite);

    unsigned int pitch = m_fc->getVirialArray().getPitch();
    Scalar fac = Scalar(1.0) + m_bias;

    gpu_scale_netforce(d_force.data, d_torque.data, d_virial.data, pitch, fac, m_pdata->getN()+m_pdata->getNGhosts());

    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    }
#endif

void CollectiveWrapper::computeBiasForces(unsigned int timestep)
    {
    // compute the force (note this only computes once per time step)
    m_fc->compute(timestep);

    if (m_prof)
        m_prof->push("Collective wrap");

    Scalar fac = Scalar(1.0) + m_bias;

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

    for (unsigned int i = 0; i < 6; ++i)
        {
        Scalar v = m_fc->getExternalVirial(i);

        // note: we cannot change the external virial on the ForceCompute, so add it here
        m_external_virial[i] = (fac-Scalar(1.0))*v;
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
