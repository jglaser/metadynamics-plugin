#include <hoomd/hoomd_config.h>
#include <hoomd/HOOMDMath.h>
#include <hoomd/Index1D.h>
#include <hoomd/BoxDim.h>

#include <cufft.h>

void gpu_assign_particles(const unsigned int N,
                          const Scalar4 *d_postype,
                          cufftComplex *d_mesh,
                          const Index3D& mesh_idx,
                          const Scalar *d_mode,
                          const unsigned int *d_cadj,
                          const Index2D& cadji,
                          const BoxDim& box);

void gpu_update_meshes(const unsigned int n_wave_vectors,
                     cufftComplex *d_fourier_mesh,
                     cufftComplex *d_fourier_mesh_G,
                     const Scalar *d_inf_f,
                     const Scalar3 *d_k,
                     const Scalar V_cell,
                     cufftComplex *d_fourier_mesh_x,
                     cufftComplex *d_fourier_mesh_y,
                     cufftComplex *d_fourier_mesh_z);

void gpu_interpolate_forces(const unsigned int N,
                             const Scalar4 *d_postype,
                             Scalar4 *d_force,
                             const Scalar bias,
                             const cufftComplex *d_force_mesh_x,
                             const cufftComplex *d_force_mesh_y,
                             const cufftComplex *d_force_mesh_z,
                             const Index3D& mesh_idx,
                             const Scalar *d_mode,
                             const unsigned int *d_cadj,
                             const Index2D& cadji,
                             const BoxDim& box);

void gpu_compute_cv(unsigned int n_wave_vectors,
                           Scalar *d_sum_partial,
                           Scalar *d_sum,
                           cufftComplex *d_fourier_mesh,
                           cufftComplex *d_fourier_mesh_G,
                           const unsigned int block_size);
