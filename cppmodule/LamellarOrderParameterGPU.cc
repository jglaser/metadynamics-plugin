/*! \file LamellarOrderParameterGPU.cc
 *  \brief Implements the LamellarOrderParameterGPU class
 */
#include "LamellarOrderParameterGPU.h"

#ifdef ENABLE_CUDA

#include "LamellarOrderParameterGPU.cuh"

LamellarOrderParameterGPU::LamellarOrderParameterGPU(boost::shared_ptr<SystemDefinition> sysdef,
                          const std::vector<Scalar>& mode,
                          const std::vector<int3>& lattice_vectors,
                          const std::string& suffix)
    : LamellarOrderParameter(sysdef, mode, lattice_vectors, suffix)
    {

    GPUArray<Scalar> gpu_mode(mode.size(), m_exec_conf);
    m_gpu_mode.swap(gpu_mode);

    // Load mode information
    ArrayHandle<Scalar> h_gpu_mode(m_gpu_mode, access_location::host, access_mode::overwrite);
    for (unsigned int i = 0; i < mode.size(); i++)
        h_gpu_mode.data[i] = mode[i];

    m_block_size = 128;
    unsigned int max_n_blocks = m_pdata->getMaxN()/m_block_size + 1;

    GPUArray<Scalar2> fourier_mode_scratch(mode.size()*max_n_blocks, m_exec_conf);
    m_fourier_mode_scratch.swap(fourier_mode_scratch);
    }

void LamellarOrderParameterGPU::computeCV(unsigned int timestep)
    {
    if (m_prof)
        m_prof->push(m_exec_conf, "Lamellar");

    ArrayHandle<Scalar4> d_postype(m_pdata->getPositions(), access_location::device, access_mode::read);

    unsigned int max_n_blocks = m_pdata->getMaxN()/m_block_size + 1;
    if (m_fourier_mode_scratch.getNumElements() < max_n_blocks*m_fourier_modes.getNumElements())
        {
        m_fourier_mode_scratch.resize(max_n_blocks*m_fourier_modes.getNumElements());
        }

        {
        ArrayHandle<int3> d_lattice_vectors(m_lattice_vectors, access_location::device, access_mode::read);
        ArrayHandle<Scalar2> d_fourier_modes(m_fourier_modes, access_location::device, access_mode::overwrite);
        ArrayHandle<Scalar> d_gpu_mode(m_gpu_mode, access_location::device, access_mode::read);
        ArrayHandle<Scalar2> d_fourier_mode_scratch(m_fourier_mode_scratch, access_location::device, access_mode::overwrite);

        // calculate Fourier modes
        gpu_calculate_fourier_modes(m_lattice_vectors.getNumElements(),
                                    d_lattice_vectors.data,
                                    m_pdata->getN(),
                                    d_postype.data,
                                    d_gpu_mode.data,
                                    d_fourier_modes.data,
                                    m_block_size,
                                    d_fourier_mode_scratch.data,
                                    m_pdata->getGlobalBox());

        if (m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();
        }

    unsigned int N = m_pdata->getNGlobal();

#ifdef ENABLE_MPI
    // reduce Fourier modes on all processors
    if (m_pdata->getDomainDecomposition())
        {
        ArrayHandle<Scalar2> h_fourier_modes(m_fourier_modes, access_location::host, access_mode::readwrite);
        MPI_Allreduce(MPI_IN_PLACE,h_fourier_modes.data,m_fourier_modes.getNumElements(), MPI_HOOMD_SCALAR, MPI_SUM, m_exec_conf->getMPICommunicator());
        }
#endif

    ArrayHandle<Scalar2> h_fourier_modes(m_fourier_modes, access_location::host, access_mode::read);
    // calculate value of collective variable 
    Scalar sum = 0.0;
    for (unsigned k = 0; k < m_fourier_modes.getNumElements(); k++)
        {
        Scalar2 fmode = h_fourier_modes.data[k];
        Scalar norm_sq = fmode.x*fmode.x+fmode.y*fmode.y;
        sum += norm_sq*norm_sq;
        }

    sum /= (Scalar) N*(Scalar)N*(Scalar)N*(Scalar)N;

    m_cv = pow(sum,Scalar(1.0/4.0));

    if (m_prof)
        m_prof->pop(m_exec_conf);

    m_cv_last_updated = timestep;
    }


void LamellarOrderParameterGPU::computeBiasForces(unsigned int timestep)
    {
    if (m_prof)
        m_prof->push(m_exec_conf, "Lamellar");

    if (m_cv_last_updated < timestep || timestep == 0)
        computeCV(timestep);

    ArrayHandle<Scalar4> d_postype(m_pdata->getPositions(), access_location::device, access_mode::read);

        {
        ArrayHandle<int3> d_lattice_vectors(m_lattice_vectors, access_location::device, access_mode::read);
        ArrayHandle<Scalar> d_gpu_mode(m_gpu_mode, access_location::device, access_mode::read);
        ArrayHandle<Scalar4> d_force(m_force, access_location::device, access_mode::overwrite);
        ArrayHandleAsync<Scalar2> d_fourier_modes(m_fourier_modes, access_location::device, access_mode::read);

        // calculate forces
        gpu_compute_sq_forces(m_pdata->getN(),
                             d_postype.data,
                             d_force.data,
                             m_lattice_vectors.getNumElements(),
                             d_lattice_vectors.data,
                             d_gpu_mode.data,
                             m_pdata->getNGlobal(),
                             m_bias,
                             d_fourier_modes.data,
                             m_cv,
                             m_pdata->getGlobalBox());
        if (m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();
        }

    if (m_prof)
        m_prof->pop(m_exec_conf);

    }

void export_LamellarOrderParameterGPU()
    {
    class_<LamellarOrderParameterGPU, boost::shared_ptr<LamellarOrderParameterGPU>, bases<LamellarOrderParameter>, boost::noncopyable >
        ("LamellarOrderParameterGPU", init< boost::shared_ptr<SystemDefinition>,
                                         const std::vector<Scalar>&,
                                         const std::vector<int3>&,
                                         const std::string& >());
    }
#endif
