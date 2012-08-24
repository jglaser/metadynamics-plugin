/*! \file LamellarOrderParameterGPU.cc
 *  \brief Implements the LamellarOrderParameterGPU class
 */
#include "LamellarOrderParameterGPU.h"

#ifdef ENABLE_CUDA
#include "LamellarOrderParameterGPU.cuh"

LamellarOrderParameterGPU::LamellarOrderParameterGPU(boost::shared_ptr<SystemDefinition> sysdef,
                          const std::vector<Scalar>& mode,
                          const std::vector<int3>& lattice_vectors,
                          const std::vector<Scalar>& phases,
                          const std::string& suffix)
    : LamellarOrderParameter(sysdef, mode, lattice_vectors, phases, suffix)
    {

    GPUArray<Scalar> gpu_mode(mode.size(), m_exec_conf);
    m_gpu_mode.swap(gpu_mode);

    // Load mode information
    ArrayHandle<Scalar> h_gpu_mode(m_gpu_mode, access_location::host, access_mode::overwrite);
    for (unsigned int i = 0; i < mode.size(); i++)
        h_gpu_mode.data[i] = mode[i];

    m_block_size = 512;
    unsigned int max_n_blocks = m_pdata->getMaxN()/m_block_size + 1;

    GPUArray<Scalar> fourier_mode_scratch(mode.size()*max_n_blocks, m_exec_conf);
    m_fourier_mode_scratch.swap(fourier_mode_scratch);

    m_wave_vectors_updated = 0;
    }

void LamellarOrderParameterGPU::computeCV(unsigned int timestep)
    {
    if (m_prof)
        m_prof->push(m_exec_conf, "Lamellar");

    ArrayHandle<Scalar4> d_postype(m_pdata->getPositions(), access_location::device, access_mode::read);

    // initialize wave vectors
    if (m_wave_vectors_updated < timestep)
        {
        calculateWaveVectors();
        m_wave_vectors_updated = timestep;
        }

    if (m_fourier_mode_scratch.getNumElements() != m_pdata->getMaxN())
        {
        unsigned int max_n_blocks = m_pdata->getMaxN()/m_block_size + 1;
        m_fourier_mode_scratch.resize(max_n_blocks*m_fourier_modes.getNumElements());
        }

        {
        ArrayHandle<Scalar3> d_wave_vectors(m_wave_vectors, access_location::device, access_mode::read);
        ArrayHandle<Scalar> d_fourier_modes(m_fourier_modes, access_location::device, access_mode::overwrite);
        ArrayHandle<Scalar> d_gpu_mode(m_gpu_mode, access_location::device, access_mode::read);
        ArrayHandle<Scalar> d_phases(m_phases, access_location::device, access_mode::read);
        ArrayHandle<Scalar> d_fourier_mode_scratch(m_fourier_mode_scratch, access_location::device, access_mode::overwrite);

        // calculate Fourier modes
        gpu_calculate_fourier_modes(m_wave_vectors.getNumElements(),
                                    d_wave_vectors.data,
                                    m_pdata->getN(),
                                    d_postype.data,
                                    d_gpu_mode.data,
                                    d_fourier_modes.data,
                                    d_phases.data,
                                    m_block_size,
                                    d_fourier_mode_scratch.data
                                    );

        CHECK_CUDA_ERROR();
        }

    ArrayHandle<Scalar> h_fourier_modes(m_fourier_modes, access_location::host, access_mode::read);
    Scalar3 L = m_pdata->getGlobalBox().getL();
    Scalar N = m_pdata->getNGlobal();

    // calculate value of collective variable (sum of real parts of fourier modes)
    Scalar sum = 0.0;
    for (unsigned k = 0; k < m_fourier_modes.getNumElements(); k++)
        sum += h_fourier_modes.data[k];

    sum /= N;

#ifdef ENABLE_MPI
    // reduce value of collective variable on root processor
    if (m_pdata->getDomainDecomposition())
        boost::mpi::reduce(*m_exec_conf->getMPICommunicator(), sum, m_sum, std::plus<Scalar>(), 0);
    else
#endif
        m_sum = sum;

    if (m_prof)
        m_prof->pop(m_exec_conf);
    }


void LamellarOrderParameterGPU::computeForces(unsigned int timestep)
    {
    if (m_prof)
        m_prof->push(m_exec_conf, "Lamellar");

    ArrayHandle<Scalar4> d_postype(m_pdata->getPositions(), access_location::device, access_mode::read);

    // initialize wave vectors
    if (m_wave_vectors_updated < timestep)
        {
        calculateWaveVectors();
        m_wave_vectors_updated = timestep;
        }

        {
        ArrayHandle<Scalar3> d_wave_vectors(m_wave_vectors, access_location::device, access_mode::read);
        ArrayHandle<Scalar> d_gpu_mode(m_gpu_mode, access_location::device, access_mode::read);
        ArrayHandle<Scalar> d_phases(m_phases, access_location::device, access_mode::read);
        ArrayHandle<Scalar4> d_force(m_force, access_location::device, access_mode::overwrite);

        // calculate forces
        gpu_compute_sq_forces(m_pdata->getN(),
                             d_postype.data,
                             d_force.data,
                             m_wave_vectors.getNumElements(),
                             d_wave_vectors.data,
                             d_gpu_mode.data,
                             m_pdata->getNGlobal(),
                             m_bias,
                             d_phases.data);
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
                                         const std::vector<Scalar>&,
                                         const std::string& >());
    }
#endif
