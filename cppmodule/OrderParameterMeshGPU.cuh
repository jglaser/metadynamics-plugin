#include <hoomd/hoomd_config.h>
#include <hoomd/HOOMDMath.h>
#include <hoomd/Index1D.h>
#include <hoomd/BoxDim.h>

#include <cufft.h>

void gpu_assign_particles(const unsigned int N,
                          const Scalar4 *d_postype,
                          cufftReal *d_mesh,
                          const Index3D& mesh_idx,
                          const Scalar *d_mode,
                          const BoxDim& box);

void gpu_update_meshes(const unsigned int n_wave_vectors,
                     cufftComplex *d_fourier_mesh,
                     cufftComplex *d_fourier_mesh_G,
                     const Scalar *d_inf_f,
                     const Scalar3 *d_k,
                     const Scalar V_cell,
                     cufftComplex *d_fourier_mesh_force);

void gpu_interpolate_forces(const unsigned int N,
                             const Scalar4 *d_postype,
                             Scalar4 *d_force,
                             const Scalar bias,
                             const cufftReal *d_ifourier_mesh_force,
                             Scalar4 *d_force_mesh,
                             const Index3D& mesh_idx,
                             const Scalar *d_mode,
                             const BoxDim& box);

void gpu_compute_cv(unsigned int n_wave_vectors,
                           Scalar *d_sum_partial,
                           Scalar *d_sum,
                           const cufftComplex *d_fourier_mesh,
                           const cufftComplex *d_fourier_mesh_G,
                           const unsigned int block_size,
                           const Index3D& mesh_idx);

void gpu_compute_influence_function(const Index3D& mesh_idx,
                                    const unsigned int N,
                                    Scalar *d_inf_f,
                                    Scalar3 *d_k,
                                    const BoxDim& box,
                                    const Scalar qstarsq);
 
