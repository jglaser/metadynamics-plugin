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

    }

void LamellarOrderParameterGPU::computeForces(unsigned int timestep)
    {
    if (m_prof)
        m_prof->push("cv lamellar");
    ArrayHandle<Scalar4> d_postype(m_pdata->getPositions(), access_location::device, access_mode::read);

    // initialize wave vectors
    calculateWaveVectors();

        {
        ArrayHandle<Scalar3> d_wave_vectors(m_wave_vectors, access_location::device, access_mode::read);
        ArrayHandle<Scalar2> d_fourier_modes(m_fourier_modes, access_location::device, access_mode::overwrite);
        ArrayHandle<Scalar> d_gpu_mode(m_gpu_mode, access_location::device, access_mode::read);
        ArrayHandle<Scalar> d_phases(m_phases, access_location::device, access_mode::read);

        // calculate Fourier modes
        gpu_calculate_fourier_modes(m_wave_vectors.getNumElements(),
                                    d_wave_vectors.data,
                                    m_pdata->getN(),
                                    d_postype.data,
                                    d_gpu_mode.data,
                                    d_fourier_modes.data,
                                    d_phases.data);

        CHECK_CUDA_ERROR();

        ArrayHandle<Scalar4> d_force(m_force, access_location::device, access_mode::overwrite);
        ArrayHandle<Scalar> d_virial(m_virial, access_location::device, access_mode::overwrite);

        // calculate forces
        gpu_compute_sq_forces(m_pdata->getN(),
                             d_postype.data,
                             d_force.data,
                             d_virial.data,
                             m_wave_vectors.getNumElements(),
                             d_wave_vectors.data,
                             d_gpu_mode.data,
                             m_pdata->getGlobalBox(),
                             m_bias,
                             d_phases.data);
        CHECK_CUDA_ERROR();
        }

    ArrayHandle<Scalar2> h_fourier_modes(m_fourier_modes, access_location::host, access_mode::read);
    Scalar3 L = m_pdata->getGlobalBox().getL();
    Scalar V = L.x*L.y*L.z;

    // calculate value of collective variable (sum of real parts of fourier modes)
    m_sum = 0.0;
    for (unsigned k = 0; k < m_fourier_modes.getNumElements(); k++)
        {
        Scalar2 fourier_mode = h_fourier_modes.data[k];
        m_sum += fourier_mode.x;
        }

    m_sum /= V;

    if (m_prof)
        m_prof->pop();

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
