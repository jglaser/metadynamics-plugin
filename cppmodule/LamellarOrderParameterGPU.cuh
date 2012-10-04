/*! \file LamellarOrderParameterGPU.cuh
 *  \brief Defines the GPU routines for LamellarOrderParameterGPU
 */
#include <hoomd/hoomd_config.h>
#include <hoomd/BoxDim.h>

/*! Calculates the fourier modes for the collective variable

    \param n_wave Number of modes
    \param d_wave_vectors Device array of wave vectors
    \param n_particles Number of particles
    \param d_postype Device array of particle positions and types
    \param d_mode Device array of per-type mode coefficients
    \param d_fourier_modes The fourier modes (Output device array)
    \param d_phases Device array of per-mode phase shifts
    \param block_size Block size for fourier mode reduction
    \param d_fourier_mode_scratch Scratch space for fourier mode reduction

    \returns the CUDA status
 */
cudaError_t gpu_calculate_fourier_modes(unsigned int n_wave,
                                 Scalar3 *d_wave_vectors,
                                 unsigned int n_particles,
                                 Scalar4 *d_postype,
                                 Scalar *d_mode,
                                 Scalar2 *d_fourier_modes,
                                 unsigned int block_size,
                                 Scalar2 *d_fourier_mode_scratch);

/*! Calculates the negative derivative of the collective variable with
    respect to particle positions
  
    \param N Number of particles
    \param d_postype Array of particle positions and types
    \param d_force Array of per-particle forces
    \param n_wave Number of modes
    \param d_wave_vectors Device array of wave vectors
    \param d_mode Device array of per-type mode coefficients
    \param n_global Total number of particles in system
    \param bias The bias factor to multiply the forces with
    \param fourier_modes Array of fourier modes
    \param sum_of_sq Sum of structure factors
    \returns the CUDA status
*/
cudaError_t gpu_compute_sq_forces(unsigned int N,
                                  Scalar4 *d_postype,
                                  Scalar4 *d_force,
                                  unsigned int n_wave,
                                  Scalar3 *d_wave_vectors,
                                  Scalar *d_mode,
                                  unsigned int n_global,
                                  Scalar bias,
                                  Scalar2 *fourier_modes,
                                  Scalar cv_val);
