#include <hoomd/hoomd_config.h>
#include <hoomd/BoxDim.h>

cudaError_t gpu_calculate_fourier_modes(unsigned int n_wave,
                                 Scalar3 *d_wave_vectors,
                                 unsigned int n_particles,
                                 Scalar4 *d_postype,
                                 Scalar *d_mode,
                                 Scalar2 *d_fourier_modes);

cudaError_t gpu_compute_sq_forces(unsigned int N,
                                  Scalar4 *d_postype,
                                  Scalar4 *d_force,
                                  Scalar *d_virial,
                                  unsigned int n_wave,
                                  Scalar3 *d_wave_vectors,
                                  Scalar2 *d_fourier_modes,
                                  Scalar *d_mode,
                                  const BoxDim global_box,
                                  Scalar bias);
