#include <hoomd/hoomd_config.h>
#include <hoomd/HOOMDMath.h>
#include <hoomd/Index1D.h>
#include <hoomd/BoxDim.h>

#ifdef ENABLE_MPI
#include <hoomd/DFFTIndex.h>
#endif

#include <cufft.h>

void gpu_bin_particles(const unsigned int N,
                       const Scalar4 *d_postype,
                       Scalar4 *d_particle_bins,
                       unsigned int *d_n_cell,
                       unsigned int *d_overflow,
                       const unsigned int maxn,
                       const Index3D& mesh_idx,
                       const uint3 n_ghost_cells,
                       const Scalar *d_mode,
                       const BoxDim& box);

void gpu_assign_binned_particles_to_mesh(const Index3D& mesh_idx,
                                         const uint3 n_ghost_cells,
                                         const Index2D& badji,
                                         const unsigned int *d_n_bin_adj,
                                         const int4 *d_bin_adj,
                                         const Scalar4 *d_particle_bins,
                                         const unsigned int *d_n_cell,
                                         const unsigned int maxn,
                                         cufftComplex *d_mesh,
                                         const BoxDim& box,
                                         const bool local_fft);

void gpu_assign_particles_30(const unsigned int N,
                          const Scalar4 *d_postype,
                          cufftComplex *d_mesh,
                          const Index3D& mesh_idx,
                          const uint3 n_ghost_cells,
                          const Scalar *d_mode,
                          const BoxDim& box,
                          const bool local_fft);

void gpu_compute_mesh_virial(const unsigned int n_wave_vectors,
                             cufftComplex *d_fourier_mesh,
                             cufftComplex *d_fourier_mesh_G,
                             Scalar *d_virial_mesh,
                             const Scalar3 *d_k,
                             const Scalar qstarsq,
                             const bool exclude_dc);

void gpu_update_meshes(const unsigned int n_wave_vectors,
                         cufftComplex *d_fourier_mesh,
                         cufftComplex *d_fourier_mesh_G,
                         const Scalar *d_inf_f,
                         const Scalar3 *d_k,
                         const Scalar V_cell,
                         const unsigned int N_global,
                         cufftComplex *d_fourier_mesh_force_x,
                         cufftComplex *d_fourier_mesh_force_y,
                         cufftComplex *d_fourier_mesh_force_z);

void gpu_coalesce_forces(const unsigned int num_force_cells,
                         const cufftComplex *d_force_mesh_x,
                         const cufftComplex *d_force_mesh_y,
                         const cufftComplex *d_force_mesh_z,
                         Scalar4 *d_force_mesh);

void gpu_interpolate_forces(const unsigned int N,
                             const Scalar4 *d_postype,
                             Scalar4 *d_force,
                             const Scalar bias,
                             Scalar4 *d_force_mesh,
                             const Index3D& mesh_idx,
                             const uint3 n_ghost_cells,
                             const Scalar *d_mode,
                             const BoxDim& box,
                             const BoxDim& global_box,
                             const bool local_fft);

void gpu_compute_cv(unsigned int n_wave_vectors,
                   Scalar *d_sum_partial,
                   Scalar *d_sum,
                   const cufftComplex *d_fourier_mesh,
                   const cufftComplex *d_fourier_mesh_G,
                   const unsigned int block_size,
                   const Index3D& mesh_idx,
                   const bool exclude_dc);

void gpu_compute_virial(unsigned int n_wave_vectors,
                   Scalar *d_sum_virial_partial,
                   Scalar *d_sum_virial,
                   const Scalar *d_mesh_virial,
                   const unsigned int block_size);

void gpu_compute_influence_function(const Index3D& mesh_idx,
                                    const uint3 n_ghost_cells,
                                    const uint3 global_dim,
                                    Scalar *d_inf_f,
                                    Scalar3 *d_k,
                                    const BoxDim& global_box,
                                    const Scalar qstarsq,
#ifdef ENABLE_MPI
                                    const DFFTIndex dffti,
#endif
                                    const bool local_fft);
