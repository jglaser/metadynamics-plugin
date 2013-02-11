#include "OrderParameterMeshGPU.cuh"

//! Implements workaround atomic float addition on sm_1x hardware
__device__ inline void atomicFloatAdd(float* address, float value)
    {
#if (__CUDA_ARCH__ < 200)
    float old = value;
    float new_old;
    do
        {
        new_old = atomicExch(address, 0.0f);
        new_old += old;
        }
    while ((old = atomicExch(address, new_old))!=0.0f);
#else
    atomicAdd(address, value);
#endif
    }


/*! \param x Distance on mesh in units of the mesh size
 */
__device__ Scalar assignTSC(Scalar x)
    {
    Scalar xsq = x*x;
    Scalar xabs = sqrtf(xsq);

    if (xsq <= Scalar(1.0/4.0))
        return Scalar(3.0/4.0) - xsq;
    else if (xsq <= Scalar(9.0/4.0))
        return Scalar(1.0/2.0)*(Scalar(3.0/2.0)-xabs)*(Scalar(3.0/2.0)-xabs);
    else
        return Scalar(0.0);
    }

__device__ uint3 find_cell(const Scalar3& pos,
                           const unsigned int& inner_nx,
                           const unsigned int& inner_ny,
                           const unsigned int& inner_nz,
                           const uint3& n_ghost_cells,
                           const BoxDim& box
                           )
    {
    // compute coordinates in units of the mesh size
    Scalar3 f = box.makeFraction(pos);

    Scalar3 reduced_pos = make_scalar3(f.x * (Scalar)inner_nx,
                                       f.y * (Scalar)inner_ny,
                                       f.z * (Scalar)inner_nz);

    // find cell the particle is in
    int ix = reduced_pos.x;
    int iy = reduced_pos.y;
    int iz = reduced_pos.z;

    // handle particles on the boundary
    if (ix == (int)inner_nx)
        ix --;
    if (iy == (int)inner_ny)
        iy--;
    if (iz == (int)inner_nz) 
        iz--;

    ix += (int) n_ghost_cells.x/2;
    iy += (int) n_ghost_cells.y/2;
    iz += (int) n_ghost_cells.z/2;

    return make_uint3(ix, iy, iz);
    }

//! Assignment of particles to mesh using three-point scheme (triangular shaped cloud)
/*! This is a second order accurate scheme with continuous value and continuous derivative
 */
template<bool local_fft>
__global__ void gpu_assign_particles_kernel(const unsigned int N,
                                       const Scalar4 *d_postype,
                                       cufftComplex *d_mesh,
                                       const Index3D mesh_idx,
                                       const uint3 n_ghost_cells,
                                       const Scalar *d_mode,
                                       const BoxDim box)
    {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= N) return;

    int3 inner_dim = make_int3(mesh_idx.getW()-n_ghost_cells.x,
                               mesh_idx.getH()-n_ghost_cells.y,
                               mesh_idx.getD()-n_ghost_cells.z);

    Scalar V_cell = box.getVolume()/(Scalar)(inner_dim.x*inner_dim.y*inner_dim.z);
 
    Scalar4 postype = d_postype[idx];

    Scalar3 pos = make_scalar3(postype.x, postype.y, postype.z);
    unsigned int type = __float_as_int(postype.w);
    Scalar mode = d_mode[type];

    // compute coordinates in units of the mesh size
    uint3 cell_coord = find_cell(pos, inner_dim.x, inner_dim.y, inner_dim.z, n_ghost_cells, box);

    // center of cell (in units of the cell size)
    Scalar3 c = make_scalar3((Scalar)cell_coord.x-(Scalar)(n_ghost_cells.x/2)+Scalar(0.5),
                             (Scalar)cell_coord.y-(Scalar)(n_ghost_cells.y/2)+Scalar(0.5),
                             (Scalar)cell_coord.z-(Scalar)(n_ghost_cells.z/2)+Scalar(0.5));

    Scalar3 p = box.makeFraction(pos)*make_scalar3(inner_dim.x, inner_dim.y, inner_dim.z);
    Scalar3 shift = p-c;

    // assign particle to cell and next neighbors
    for (int i = -1; i <= 1 ; ++i)
    	for (int j = -1; j <= 1; ++j)
            for (int k = -1; k <= 1; ++k)
                {
                int neighi = (int)cell_coord.x + i;
                int neighj = (int)cell_coord.y + j;
                int neighk = (int)cell_coord.z + k;

                if (! n_ghost_cells.x)
                    {
                    if (neighi == inner_dim.x)
                        neighi = 0;
                    else if (neighi < 0)
                        neighi += inner_dim.x;
                    }

                if (! n_ghost_cells.y)
                    {
                    if (neighj == inner_dim.y)
                        neighj = 0;
                    else if (neighj < 0)
                        neighj += inner_dim.y;
                    }

                if (! n_ghost_cells.z)
                    {
                    if (neighk == inner_dim.z)
                        neighk = 0;
                    else if (neighk < 0)
                        neighk += inner_dim.z;
                    }
                
                Scalar3 dx_frac = shift - make_scalar3(i,j,k);
                
                // compute fraction of particle density assigned to cell
                Scalar density_fraction = assignTSC(dx_frac.x)*assignTSC(dx_frac.y)*assignTSC(dx_frac.z)/V_cell;

                unsigned int cell_idx;
                if (local_fft)
                    // use cuFFT's memory layout
                    cell_idx = neighk + inner_dim.z * (neighj + inner_dim.y * neighi);
                else
                    cell_idx = mesh_idx(neighi, neighj, neighk);

                atomicFloatAdd(&d_mesh[cell_idx].x, mode*density_fraction);
                }
                 
    }

__global__ void gpu_bin_particles_kernel(const unsigned int N,
                                         const Scalar4 *d_postype,
                                         Scalar4 *d_particle_bins,
                                         unsigned int *d_n_cell,
                                         unsigned int *d_overflow,
                                         const unsigned int maxn,
                                         const Index3D mesh_idx,
                                         const uint3 n_ghost_cells,
                                         const Scalar *d_mode,
                                         const BoxDim box)
    {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= N) return;

    int3 inner_dim = make_int3(mesh_idx.getW()-n_ghost_cells.x,
                               mesh_idx.getH()-n_ghost_cells.y,
                               mesh_idx.getD()-n_ghost_cells.z);
 
    Scalar4 postype = d_postype[idx];

    Scalar3 pos = make_scalar3(postype.x, postype.y, postype.z);
    unsigned int type = __float_as_int(postype.w);
    Scalar mode = d_mode[type];

    // compute coordinates in units of the cell size
    uint3 bin_coord = find_cell(pos, inner_dim.x, inner_dim.y, inner_dim.z, make_uint3(0,0,0), box);

    unsigned int bin_idx = bin_coord.z + inner_dim.z * (bin_coord.y + inner_dim.y * bin_coord.x);

    unsigned int n = atomicInc(&d_n_cell[bin_idx], 0xffffffff);

    if (n >= maxn)
        {
        // overflow
        atomicMax(d_overflow, n+1);
        }
    else
        {
        // store distance to bin center in bin in units of bin size
        Scalar3 f = box.makeFraction(pos);
        f = f*make_scalar3((Scalar)inner_dim.x,(Scalar)inner_dim.y,(Scalar)inner_dim.z);
        Scalar3 c = make_scalar3((Scalar)bin_coord.x + Scalar(0.5),
                                 (Scalar)bin_coord.y + Scalar(0.5),
                                 (Scalar)bin_coord.z + Scalar(0.5));
        Scalar3 shift = f - c;

        d_particle_bins[bin_idx*maxn+n] = make_scalar4(shift.x,shift.y,shift.z, mode);
        }
    }

void gpu_bin_particles(const unsigned int N,
                       const Scalar4 *d_postype,
                       Scalar4 *d_particle_bins,
                       unsigned int *d_n_cell,
                       unsigned int *d_overflow,
                       const unsigned int maxn,
                       const Index3D& mesh_idx,
                       const uint3 n_ghost_cells,
                       const Scalar *d_mode,
                       const BoxDim& box)
    {
    unsigned int block_size = 512;

    gpu_bin_particles_kernel<<<N/block_size+1, block_size>>>(N,
                                                             d_postype,
                                                             d_particle_bins,
                                                             d_n_cell,
                                                             d_overflow,
                                                             maxn,
                                                             mesh_idx,
                                                             n_ghost_cells,
                                                             d_mode,
                                                             box);
    }


texture<Scalar4, 1, cudaReadModeElementType> particle_bins_tex;

template<bool local_fft>
__global__ void gpu_assign_binned_particles_to_mesh_kernel(const unsigned int inner_nx,
                                                           const unsigned int inner_ny,
                                                           const unsigned int inner_nz,
                                                           const Index3D mesh_idx,
                                                           const uint3 n_ghost_cells,
                                                           const unsigned int *d_n_cell,
                                                           const unsigned int maxn,
                                                           cufftComplex *d_mesh,
                                                           const BoxDim box)
    {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i>= (int)mesh_idx.getW() || j >= (int)mesh_idx.getH() || k >= (int)mesh_idx.getD()) return;

    unsigned int cell_idx;
    if (local_fft)
        // for local FFT, use cuFFT's memory layout
        cell_idx = k + mesh_idx.getD() * (j + mesh_idx.getH() * i);
    else
        cell_idx = mesh_idx(i, j, k);

    Scalar V_cell = box.getVolume()/(Scalar)(inner_nx*inner_ny*inner_nz);
    Scalar grid_val(0.0);

    // loop over particles in neighboring bins
    for (int l = -1; l <= 1 ; ++l)
    	for (int m = -1; m <= 1; ++m)
            for (int n = -1; n <= 1; ++n)
                {
                int neighi = i + l;
                int neighj = j + m;
                int neighk = k + n;

                // when using ghost cells, only add particles from inner bins
                if (n_ghost_cells.x)
                    {
                    if ((neighi >= (int)(inner_nx+n_ghost_cells.x/2)) ||
                        (neighi < (int)n_ghost_cells.x/2)) continue;
                    }
                else
                    {
                    if (neighi == (int)inner_nx)
                        neighi = 0;
                    else if (neighi < 0)
                        neighi += (int)inner_nx; 
                    }

                if (n_ghost_cells.y)
                    {
                    if ((neighj >= (int)(inner_ny+n_ghost_cells.y/2)) ||
                        (neighj < (int)n_ghost_cells.y/2)) continue;
                    }
                else
                    {
                    if (neighj == (int)inner_ny)
                        neighj = 0;
                    else if (neighj < 0)
                        neighj += (int) inner_ny;
                    }

                if (n_ghost_cells.z)
                    {
                    if ((neighk >= (int)(inner_nz+n_ghost_cells.z/2)) ||
                        (neighk < (int)n_ghost_cells.z/2)) continue;
                    }
                else
                    {
                    if (neighk == (int)inner_nz)
                        neighk = 0;
                    else if (neighk < 0)
                        neighk += (int)inner_nz;
                    }
                uint3 bin_idx = make_uint3((unsigned int)neighi - n_ghost_cells.x/2,
                                           (unsigned int)neighj - n_ghost_cells.y/2,
                                           (unsigned int)neighk - n_ghost_cells.z/2);
                unsigned int neigh_bin = bin_idx.z + inner_nz * (bin_idx.y + inner_ny * bin_idx.x);

                unsigned int n_bin = d_n_cell[neigh_bin];
                Scalar3 cell_shift = make_scalar3((Scalar)l,(Scalar)m,(Scalar)n);

                // loop over particles in bin
                for (unsigned int neigh_idx = 0; neigh_idx < n_bin; neigh_idx++)
                    {
                    Scalar4 xyzm = tex1Dfetch(particle_bins_tex, maxn*neigh_bin+neigh_idx);
                    Scalar3 shift_frac = make_scalar3(xyzm.x, xyzm.y, xyzm.z);

                    Scalar3 dx_frac = shift_frac + cell_shift;

                    // compute fraction of particle density assigned to cell
                    Scalar mode = xyzm.w;
                    grid_val += mode*assignTSC(dx_frac.x)*assignTSC(dx_frac.y)*assignTSC(dx_frac.z)/V_cell;
                    }
                } // end of loop over neighboring bins

    // write out mesh value
    d_mesh[cell_idx].x = grid_val;
    }

void gpu_assign_binned_particles_to_mesh(const Index3D& mesh_idx,
                                         const uint3 n_ghost_cells,
                                         const Scalar4 *d_particle_bins,
                                         const unsigned int *d_n_cell,
                                         const unsigned int maxn,
                                         cufftComplex *d_mesh,
                                         const BoxDim& box,
                                         const bool local_fft)
    {
    unsigned int block_size = 8;
   
    uint3 inner_dim = make_uint3(mesh_idx.getW() - n_ghost_cells.x,
                                 mesh_idx.getH() - n_ghost_cells.y,
                                 mesh_idx.getD() - n_ghost_cells.z);

    unsigned int num_bins = inner_dim.x*inner_dim.y*inner_dim.z;

    particle_bins_tex.normalized = false;
    particle_bins_tex.filterMode = cudaFilterModePoint;
    cudaBindTexture(0, particle_bins_tex, d_particle_bins, sizeof(Scalar4)*num_bins*maxn);


    dim3 blockDim(block_size,block_size,block_size);
    dim3 gridDim((mesh_idx.getW() % block_size == 0) ? mesh_idx.getW()/block_size : mesh_idx.getW()/block_size+1,
                 (mesh_idx.getH() % block_size == 0) ? mesh_idx.getH()/block_size : mesh_idx.getH()/block_size+1,
                 (mesh_idx.getD() % block_size == 0) ? mesh_idx.getD()/block_size : mesh_idx.getD()/block_size+1);

   
    if (local_fft)
        gpu_assign_binned_particles_to_mesh_kernel<true><<<gridDim,blockDim>>>(inner_dim.x,
                                                                              inner_dim.y,
                                                                              inner_dim.z,
                                                                              mesh_idx,
                                                                              n_ghost_cells,
                                                                              d_n_cell,
                                                                              maxn,
                                                                              d_mesh,
                                                                              box);
    else
        gpu_assign_binned_particles_to_mesh_kernel<false><<<gridDim,blockDim>>>(inner_dim.x,
                                                                               inner_dim.y,
                                                                               inner_dim.z,
                                                                               mesh_idx,
                                                                               n_ghost_cells,
                                                                               d_n_cell,
                                                                               maxn,
                                                                               d_mesh,
                                                                               box);
    }

void gpu_assign_particles_30(const unsigned int N,
                          const Scalar4 *d_postype,
                          cufftComplex *d_mesh,
                          const Index3D& mesh_idx,
                          const uint3 n_ghost_cells,
                          const Scalar *d_mode,
                          const BoxDim& box,
                          const bool local_fft)
    {

    unsigned int block_size = 512;

    if (local_fft) 
        gpu_assign_particles_kernel<true><<<N/block_size+1, block_size>>>(N,
                                                                          d_postype,
                                                                          d_mesh,
                                                                          mesh_idx,
                                                                          n_ghost_cells,
                                                                          d_mode,
                                                                          box);
    else    
        gpu_assign_particles_kernel<false><<<N/block_size+1, block_size>>>(N,
                                                                          d_postype,
                                                                          d_mesh,
                                                                          mesh_idx,
                                                                          n_ghost_cells,
                                                                          d_mode,
                                                                          box);
 
    }

__global__ void gpu_compute_mesh_virial_kernel(const unsigned int n_wave_vectors,
                                         cufftComplex *d_fourier_mesh,
                                         cufftComplex *d_fourier_mesh_G,
                                         Scalar *d_virial_mesh,
                                         const Scalar3 *d_k,
                                         const Scalar qstarsq,
                                         const bool exclude_dc)
    {
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx >= n_wave_vectors) return;

    if (!exclude_dc || idx != 0)
        {
        // non-zero wave vector
        cufftComplex f_g = d_fourier_mesh_G[idx];
        cufftComplex f = d_fourier_mesh[idx];

        Scalar rhog = f_g.x * f.x + f_g.y * f.y;
        Scalar3 k = d_k[idx];
        Scalar kfac = Scalar(1.0)/qstarsq;
        d_virial_mesh[0*n_wave_vectors+idx] = rhog*(Scalar(1.0) - kfac*k.x*k.x); // xx
        d_virial_mesh[1*n_wave_vectors+idx] = rhog*(            - kfac*k.x*k.y); // xy
        d_virial_mesh[2*n_wave_vectors+idx] = rhog*(            - kfac*k.x*k.z); // xz
        d_virial_mesh[3*n_wave_vectors+idx] = rhog*(Scalar(1.0) - kfac*k.y*k.y); // yy
        d_virial_mesh[4*n_wave_vectors+idx] = rhog*(            - kfac*k.y*k.z); // yz
        d_virial_mesh[5*n_wave_vectors+idx] = rhog*(Scalar(1.0) - kfac*k.z*k.z); // zz
        }
    else
        {
        d_virial_mesh[0*n_wave_vectors+idx] = Scalar(0.0);
        d_virial_mesh[1*n_wave_vectors+idx] = Scalar(0.0);
        d_virial_mesh[2*n_wave_vectors+idx] = Scalar(0.0);
        d_virial_mesh[3*n_wave_vectors+idx] = Scalar(0.0);
        d_virial_mesh[4*n_wave_vectors+idx] = Scalar(0.0);
        d_virial_mesh[5*n_wave_vectors+idx] = Scalar(0.0);
        } 
    }

void gpu_compute_mesh_virial(const unsigned int n_wave_vectors,
                             cufftComplex *d_fourier_mesh,
                             cufftComplex *d_fourier_mesh_G,
                             Scalar *d_virial_mesh,
                             const Scalar3 *d_k,
                             const Scalar qstarsq,
                             const bool exclude_dc)
    {
    const unsigned int block_size = 512;

    gpu_compute_mesh_virial_kernel<<<n_wave_vectors/block_size+1, block_size>>>(n_wave_vectors,
                                                                          d_fourier_mesh,
                                                                          d_fourier_mesh_G,
                                                                          d_virial_mesh,
                                                                          d_k,
                                                                          qstarsq,
                                                                          exclude_dc);
    }
 
__global__ void gpu_update_meshes_kernel(const unsigned int n_wave_vectors,
                                         cufftComplex *d_fourier_mesh,
                                         cufftComplex *d_fourier_mesh_G,
                                         const Scalar *d_inf_f,
                                         const Scalar3 *d_k,
                                         const Scalar V_cell,
                                         cufftComplex *d_fourier_mesh_force_x,
                                         cufftComplex *d_fourier_mesh_force_y,
                                         cufftComplex *d_fourier_mesh_force_z)
    {
    unsigned int k = blockDim.x * blockIdx.x + threadIdx.x;

    if (k >= n_wave_vectors) return;

    cufftComplex f = d_fourier_mesh[k];

    f.x *= V_cell;
    f.y *= V_cell;

    Scalar val = f.x*f.x+f.y*f.y;

    cufftComplex fourier_G;
    fourier_G.x =f.x * val * d_inf_f[k];
    fourier_G.y =f.y * val * d_inf_f[k];

    Scalar3 kval = Scalar(2.0)*d_k[k];
    d_fourier_mesh_force_x[k].x = -fourier_G.y*kval.x;
    d_fourier_mesh_force_x[k].y = fourier_G.x*kval.x;

    d_fourier_mesh_force_y[k].x = -fourier_G.y*kval.y;
    d_fourier_mesh_force_y[k].y = fourier_G.x*kval.y;

    d_fourier_mesh_force_z[k].x = -fourier_G.y*kval.z;
    d_fourier_mesh_force_z[k].y = fourier_G.x*kval.z;

    d_fourier_mesh[k] = f;
    d_fourier_mesh_G[k] = fourier_G;
    }

void gpu_update_meshes(const unsigned int n_wave_vectors,
                         cufftComplex *d_fourier_mesh,
                         cufftComplex *d_fourier_mesh_G,
                         const Scalar *d_inf_f,
                         const Scalar3 *d_k,
                         const Scalar V_cell,
                         cufftComplex *d_fourier_mesh_force_x,
                         cufftComplex *d_fourier_mesh_force_y,
                         cufftComplex *d_fourier_mesh_force_z)

    {
    const unsigned int block_size = 512;

    gpu_update_meshes_kernel<<<n_wave_vectors/block_size+1, block_size>>>(n_wave_vectors,
                                                                          d_fourier_mesh,
                                                                          d_fourier_mesh_G,
                                                                          d_inf_f,
                                                                          d_k,
                                                                          V_cell,
                                                                          d_fourier_mesh_force_x,
                                                                          d_fourier_mesh_force_y,
                                                                          d_fourier_mesh_force_z);
    }

//! Texture for reading particle positions
texture<Scalar4, 1, cudaReadModeElementType> force_mesh_tex;

__global__ void gpu_coalesce_forces_kernel(const unsigned int n_force_cells,
                                           const cufftComplex *d_force_mesh_x,
                                           const cufftComplex *d_force_mesh_y,
                                           const cufftComplex *d_force_mesh_z,
                                           Scalar4 *d_force_mesh)
    {
    unsigned int k = blockIdx.x*blockDim.x+threadIdx.x;

    if (k >= n_force_cells) return;

    d_force_mesh[k] = make_scalar4(d_force_mesh_x[k].x,
                                   d_force_mesh_y[k].x,
                                   d_force_mesh_z[k].x,
                                   0.0);
    }

template<bool local_fft>
__global__ void gpu_interpolate_forces_kernel(const unsigned int N,
                                              const unsigned int Nglobal,
                                              const Scalar4 *d_postype,
                                              Scalar4 *d_force,
                                              const Scalar bias,
                                              const Index3D mesh_idx,
                                              const uint3 n_ghost_cells,
                                              const Scalar *d_mode,
                                              const BoxDim box,
                                              const Scalar V)
    {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= N) return;

    int3 inner_dim = make_int3(mesh_idx.getW()-n_ghost_cells.x,
                               mesh_idx.getH()-n_ghost_cells.y,
                               mesh_idx.getD()-n_ghost_cells.z);

    Scalar4 postype = d_postype[idx];

    Scalar3 pos = make_scalar3(postype.x, postype.y, postype.z);
    unsigned int type = __float_as_int(postype.w);
    Scalar mode = d_mode[type];

    // find cell the particle is in
    uint3 cell_coord = find_cell(pos, inner_dim.x, inner_dim.y, inner_dim.z, n_ghost_cells, box);

    // center of cell (in units of the cell size)
    Scalar3 c = make_scalar3((Scalar)cell_coord.x-(Scalar)(n_ghost_cells.x/2)+Scalar(0.5),
                             (Scalar)cell_coord.y-(Scalar)(n_ghost_cells.y/2)+Scalar(0.5),
                             (Scalar)cell_coord.z-(Scalar)(n_ghost_cells.z/2)+Scalar(0.5));

    Scalar3 p = box.makeFraction(pos)*make_scalar3(inner_dim.x, inner_dim.y, inner_dim.z);
    Scalar3 shift = p-c;

    Scalar3 force = make_scalar3(0.0,0.0,0.0);

    // assign particle to cell and next neighbors
    for (int i = -1; i <= 1 ; ++i)
    	for (int j = -1; j <= 1; ++j)
            for (int k = -1; k <= 1; ++k)
                {
                int neighi = (int) cell_coord.x + i;
                int neighj = (int) cell_coord.y + j;
                int neighk = (int) cell_coord.z + k;

                if (! n_ghost_cells.x)
                    {
                    if (neighi == inner_dim.x)
                        neighi = 0;
                    else if (neighi < 0)
                        neighi += inner_dim.x;
                    }

                if (! n_ghost_cells.y)
                    {
                    if (neighj == inner_dim.y)
                        neighj = 0;
                    else if (neighj < 0)
                        neighj += inner_dim.y;
                    }

                if (! n_ghost_cells.z)
                    {
                    if (neighk == inner_dim.z)
                        neighk = 0;
                    else if (neighk < 0)
                        neighk += inner_dim.z;
                    } 

                Scalar3 dx_frac = shift - make_scalar3((Scalar)i,(Scalar)j,(Scalar)k);

                // compute fraction of particle density assigned to cell
                unsigned int cell_idx;
                if (local_fft)
                    // use cuFFT's memory layout
                    cell_idx = neighk + inner_dim.z * (neighj + inner_dim.y * neighi);
                else
                    cell_idx = mesh_idx(neighi, neighj, neighk);

                Scalar4 mesh_force = tex1Dfetch(force_mesh_tex,cell_idx);

                force += -assignTSC(dx_frac.x)*assignTSC(dx_frac.y)*assignTSC(dx_frac.z)*mode*
                        make_scalar3(mesh_force.x,mesh_force.y,mesh_force.z);
                }  

    // Multiply with bias potential derivative
    Scalar Nsq = Nglobal * Nglobal;
    force *= bias/V/Nsq/Nsq;

    d_force[idx] = make_scalar4(force.x,force.y,force.z,0.0);
    }

void gpu_coalesce_forces(const unsigned int num_force_cells,
                         const cufftComplex *d_force_mesh_x,
                         const cufftComplex *d_force_mesh_y,
                         const cufftComplex *d_force_mesh_z,
                         Scalar4 *d_force_mesh)
    {
    unsigned int block_size = 512;
    unsigned int n_blocks = num_force_cells/block_size;
    if (num_force_cells % block_size) n_blocks+=1;
    gpu_coalesce_forces_kernel<<<n_blocks, block_size>>>(num_force_cells,
                                                         d_force_mesh_x,
                                                         d_force_mesh_y,
                                                         d_force_mesh_z,
                                                         d_force_mesh);
    }

void gpu_interpolate_forces(const unsigned int N,
                             const unsigned int Nglobal,
                             const Scalar4 *d_postype,
                             Scalar4 *d_force,
                             const Scalar bias,
                             Scalar4 *d_force_mesh,
                             const Index3D& mesh_idx,
                             const uint3 n_ghost_cells,
                             const Scalar *d_mode,
                             const BoxDim& box,
                             const BoxDim& global_box,
                             const bool local_fft)
    {
    const unsigned int block_size = 512;

    // force mesh includes ghost cells
    unsigned int num_cells = mesh_idx.getNumElements();
    force_mesh_tex.normalized = false;
    force_mesh_tex.filterMode = cudaFilterModePoint;
    cudaBindTexture(0, force_mesh_tex, d_force_mesh, sizeof(Scalar4)*num_cells);

    if (local_fft)
        gpu_interpolate_forces_kernel<true><<<N/block_size+1,block_size>>>(N,
                                                                     Nglobal,
                                                                     d_postype,
                                                                     d_force,
                                                                     bias,
                                                                     mesh_idx,
                                                                     n_ghost_cells,
                                                                     d_mode,
                                                                     box,
                                                                     global_box.getVolume());
    else
        gpu_interpolate_forces_kernel<false><<<N/block_size+1,block_size>>>(N,
                                                                     Nglobal,
                                                                     d_postype,
                                                                     d_force,
                                                                     bias,
                                                                     mesh_idx,
                                                                     n_ghost_cells,
                                                                     d_mode,
                                                                     box,
                                                                     global_box.getVolume());
    }

__global__ void kernel_calculate_cv_partial(
            int n_wave_vectors,
            Scalar *sum_partial,
            const cufftComplex *d_fourier_mesh,
            const cufftComplex *d_fourier_mesh_G,
            const bool exclude_dc)
    {
    extern __shared__ Scalar sdata[];

    unsigned int tidx = threadIdx.x;

    int j = blockIdx.x * blockDim.x + threadIdx.x;

    Scalar mySum = Scalar(0.0);

    if (j < n_wave_vectors) {
        if (! exclude_dc || j != 0)
            mySum = d_fourier_mesh[j].x * d_fourier_mesh_G[j].x + d_fourier_mesh[j].y * d_fourier_mesh_G[j].y;
        }

    sdata[tidx] = mySum;

   __syncthreads();

    // reduce the sum
    int offs = blockDim.x >> 1;
    while (offs > 0)
        {
        if (tidx < offs)
            {
            sdata[tidx] += sdata[tidx + offs];
            }
        offs >>= 1;
        __syncthreads();
        }

    // write result to global memeory
    if (tidx == 0)
       sum_partial[blockIdx.x] = sdata[0];
    }

__global__ void kernel_final_reduce_cv(Scalar* sum_partial,
                                       unsigned int nblocks,
                                       Scalar *sum)
    {
    extern __shared__ Scalar smem[];

    if (threadIdx.x == 0)
       *sum = Scalar(0.0);

    for (int start = 0; start< nblocks; start += blockDim.x)
        {
        __syncthreads();
        if (start + threadIdx.x < nblocks)
            smem[threadIdx.x] = sum_partial[start + threadIdx.x];
        else
            smem[threadIdx.x] = Scalar(0.0);

        __syncthreads();

        // reduce the sum
        int offs = blockDim.x >> 1;
        while (offs > 0)
            {
            if (threadIdx.x < offs)
                smem[threadIdx.x] += smem[threadIdx.x + offs];
            offs >>= 1;
            __syncthreads();
            }

         if (threadIdx.x == 0)
            {
            *sum += smem[0];
            }
        }
    }

void gpu_compute_cv(unsigned int n_wave_vectors,
                   Scalar *d_sum_partial,
                   Scalar *d_sum,
                   const cufftComplex *d_fourier_mesh,
                   const cufftComplex *d_fourier_mesh_G,
                   const unsigned int block_size,
                   const Index3D& mesh_idx,
                   const bool exclude_dc)
    {
    unsigned int n_blocks = n_wave_vectors/block_size + 1;

    unsigned int shared_size = block_size * sizeof(Scalar);
    kernel_calculate_cv_partial<<<n_blocks, block_size, shared_size>>>(
               n_wave_vectors,
               d_sum_partial,
               d_fourier_mesh,
               d_fourier_mesh_G,
               exclude_dc);

    // calculate final sum of mesh values
    const unsigned int final_block_size = 512;
    shared_size = final_block_size*sizeof(Scalar);
    kernel_final_reduce_cv<<<1, final_block_size,shared_size>>>(d_sum_partial,
                                                                n_blocks,
                                                                d_sum);
    }

__global__ void kernel_calculate_virial_partial(
            int n_wave_vectors,
            Scalar *sum_virial_partial,
            const Scalar *d_mesh_virial)
    {
    extern __shared__ Scalar sdata[];

    int j = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int tidx = threadIdx.x;

    Scalar mySum_xx = Scalar(0.0);
    Scalar mySum_xy = Scalar(0.0);
    Scalar mySum_xz = Scalar(0.0);
    Scalar mySum_yy = Scalar(0.0);
    Scalar mySum_yz = Scalar(0.0);
    Scalar mySum_zz = Scalar(0.0);

    if (j < n_wave_vectors)
        {
        mySum_xx = d_mesh_virial[0*n_wave_vectors+j];
        mySum_xy = d_mesh_virial[1*n_wave_vectors+j];
        mySum_xz = d_mesh_virial[2*n_wave_vectors+j];
        mySum_yy = d_mesh_virial[3*n_wave_vectors+j];
        mySum_yz = d_mesh_virial[4*n_wave_vectors+j];
        mySum_zz = d_mesh_virial[5*n_wave_vectors+j];
        }

    sdata[0*blockDim.x+tidx] = mySum_xx;
    sdata[1*blockDim.x+tidx] = mySum_xy;
    sdata[2*blockDim.x+tidx] = mySum_xz;
    sdata[3*blockDim.x+tidx] = mySum_yy;
    sdata[4*blockDim.x+tidx] = mySum_yz;
    sdata[5*blockDim.x+tidx] = mySum_zz;

   __syncthreads();

    // reduce the sum
    int offs = blockDim.x >> 1;
    while (offs > 0)
        {
        if (tidx < offs)
            {
            sdata[0*blockDim.x+tidx] += sdata[0*blockDim.x+tidx + offs];
            sdata[1*blockDim.x+tidx] += sdata[1*blockDim.x+tidx + offs];
            sdata[2*blockDim.x+tidx] += sdata[2*blockDim.x+tidx + offs];
            sdata[3*blockDim.x+tidx] += sdata[3*blockDim.x+tidx + offs];
            sdata[4*blockDim.x+tidx] += sdata[4*blockDim.x+tidx + offs];
            sdata[5*blockDim.x+tidx] += sdata[5*blockDim.x+tidx + offs];
            }
        offs >>= 1;
        __syncthreads();
        }

    // write result to global memeory
    if (tidx == 0)
        {
        sum_virial_partial[0*gridDim.x+blockIdx.x] = sdata[0*blockDim.x];
        sum_virial_partial[1*gridDim.x+blockIdx.x] = sdata[1*blockDim.x];
        sum_virial_partial[2*gridDim.x+blockIdx.x] = sdata[2*blockDim.x];
        sum_virial_partial[3*gridDim.x+blockIdx.x] = sdata[3*blockDim.x];
        sum_virial_partial[4*gridDim.x+blockIdx.x] = sdata[4*blockDim.x];
        sum_virial_partial[5*gridDim.x+blockIdx.x] = sdata[5*blockDim.x];
        }
    }


__global__ void kernel_final_reduce_virial(Scalar* sum_virial_partial,
                                           unsigned int nblocks,
                                           Scalar *sum_virial)
    {
    extern __shared__ Scalar smem[];

    if (threadIdx.x == 0)
        {
        sum_virial[0] = Scalar(0.0);
        sum_virial[1] = Scalar(0.0);
        sum_virial[2] = Scalar(0.0);
        sum_virial[3] = Scalar(0.0);
        sum_virial[4] = Scalar(0.0);
        sum_virial[5] = Scalar(0.0);
        }

    for (int start = 0; start< nblocks; start += blockDim.x)
        {
        __syncthreads();
        if (start + threadIdx.x < nblocks)
            {
            smem[0*blockDim.x+threadIdx.x] = sum_virial_partial[0*nblocks+start+threadIdx.x];
            smem[1*blockDim.x+threadIdx.x] = sum_virial_partial[1*nblocks+start+threadIdx.x];
            smem[2*blockDim.x+threadIdx.x] = sum_virial_partial[2*nblocks+start+threadIdx.x];
            smem[3*blockDim.x+threadIdx.x] = sum_virial_partial[3*nblocks+start+threadIdx.x];
            smem[4*blockDim.x+threadIdx.x] = sum_virial_partial[4*nblocks+start+threadIdx.x];
            smem[5*blockDim.x+threadIdx.x] = sum_virial_partial[5*nblocks+start+threadIdx.x];
            }
        else
            {
            smem[0*blockDim.x+threadIdx.x] = Scalar(0.0);
            smem[1*blockDim.x+threadIdx.x] = Scalar(0.0);
            smem[2*blockDim.x+threadIdx.x] = Scalar(0.0);
            smem[3*blockDim.x+threadIdx.x] = Scalar(0.0);
            smem[4*blockDim.x+threadIdx.x] = Scalar(0.0);
            smem[5*blockDim.x+threadIdx.x] = Scalar(0.0);
            }

        __syncthreads();

        // reduce the sum
        int offs = blockDim.x >> 1;
        while (offs > 0)
            {
            if (threadIdx.x < offs)
                {
                smem[0*blockDim.x+threadIdx.x] += smem[0*blockDim.x+threadIdx.x + offs];
                smem[1*blockDim.x+threadIdx.x] += smem[1*blockDim.x+threadIdx.x + offs];
                smem[2*blockDim.x+threadIdx.x] += smem[2*blockDim.x+threadIdx.x + offs];
                smem[3*blockDim.x+threadIdx.x] += smem[3*blockDim.x+threadIdx.x + offs];
                smem[4*blockDim.x+threadIdx.x] += smem[4*blockDim.x+threadIdx.x + offs];
                smem[5*blockDim.x+threadIdx.x] += smem[5*blockDim.x+threadIdx.x + offs];
                }
            offs >>= 1;
            __syncthreads();
            }

         if (threadIdx.x == 0)
            {
            sum_virial[0] += smem[0*blockDim.x];
            sum_virial[1] += smem[1*blockDim.x];
            sum_virial[2] += smem[2*blockDim.x];
            sum_virial[3] += smem[3*blockDim.x];
            sum_virial[4] += smem[4*blockDim.x];
            sum_virial[5] += smem[5*blockDim.x];
            }
        }
    }

void gpu_compute_virial(unsigned int n_wave_vectors,
                   Scalar *d_sum_virial_partial,
                   Scalar *d_sum_virial,
                   const Scalar *d_mesh_virial,
                   const unsigned int block_size,
                   const Index3D& mesh_idx)
    {
    unsigned int n_blocks = n_wave_vectors/block_size + 1;

    unsigned int shared_size = 6* block_size * sizeof(Scalar);
    kernel_calculate_virial_partial<<<n_blocks, block_size, shared_size>>>(
               n_wave_vectors,
               d_sum_virial_partial,
               d_mesh_virial);

    // calculate final virial values 
    const unsigned int final_block_size = 512;
    shared_size = 6*final_block_size*sizeof(Scalar);
    kernel_final_reduce_virial<<<1, final_block_size,shared_size>>>(d_sum_virial_partial,
                                                                  n_blocks,
                                                                  d_sum_virial);
    }

__device__ Scalar convolution_kernel(Scalar ksq, Scalar qstarsq)
    {
    return expf(-ksq/qstarsq*Scalar(1.0/2.0));
    }

template<bool local_fft>
__global__ void gpu_compute_influence_function_kernel(const Index3D mesh_idx,
                                          const uint3 n_ghost_cells,
                                          const uint3 global_dim,
                                          Scalar *d_inf_f,
                                          Scalar3 *d_k,
                                          const Scalar3 b1,
                                          const Scalar3 b2,
                                          const Scalar3 b3,
                                          const Scalar V_box,
                                          const Scalar qstarsq
#ifdef ENABLE_MPI
                                          , const DFFTIndex dffti
#endif
                                          )
    {
    unsigned int kidx = blockIdx.x * blockDim.x + threadIdx.x;

    int3 inner_dim = make_int3(mesh_idx.getW() - n_ghost_cells.x,
                               mesh_idx.getH() - n_ghost_cells.y,
                               mesh_idx.getD() - n_ghost_cells.z);
    if (kidx >= inner_dim.x*inner_dim.y*inner_dim.z) return;

    int l,m,n;
    if (local_fft)
        {
        uint3 nvec = mesh_idx.getTriple(kidx);
        l = nvec.x; m = nvec.y; n = nvec.z;
        }
#ifdef ENABLE_MPI
    else
        {
        uint3 nvec = dffti(kidx);
        l = nvec.x; m = nvec.y; n = nvec.z;
        }
#endif

    unsigned int ix = l;
    unsigned int iy = m;
    unsigned int iz = n;

    // compute Miller indices
    if (l >= (int)(global_dim.x/2 + global_dim.x%2))
        l -= (int) global_dim.x;
    if (m >= (int)(global_dim.y/2 + global_dim.y%2))
        m -= (int) global_dim.y;
    if (n >= (int)(global_dim.z/2 + global_dim.z%2))
        n -= (int) global_dim.z;
    
    Scalar3 kval = (Scalar)l*b1+(Scalar)m*b2+(Scalar)n*b3;
    Scalar ksq = dot(kval,kval);

    Scalar val = convolution_kernel(ksq,qstarsq)*V_box;

    unsigned int cell_idx;
    if (local_fft)
        // use cuFFT's memory layout
        cell_idx = iz + inner_dim.z * (iy + inner_dim.y * ix);
    else
        cell_idx = kidx;

    d_inf_f[cell_idx] = val;
    d_k[cell_idx] = kval;
    }

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
                                    const bool local_fft) 
    { 
    // compute reciprocal lattice vectors
    Scalar3 a1 = global_box.getLatticeVector(0);
    Scalar3 a2 = global_box.getLatticeVector(1);
    Scalar3 a3 = global_box.getLatticeVector(2);

    Scalar V_box = global_box.getVolume();
    Scalar3 b1 = Scalar(2.0*M_PI)*make_scalar3(a2.y*a3.z-a2.z*a3.y, a2.z*a3.x-a2.x*a3.z, a2.x*a3.y-a2.y*a3.x)/V_box;
    Scalar3 b2 = Scalar(2.0*M_PI)*make_scalar3(a3.y*a1.z-a3.z*a1.y, a3.z*a1.x-a3.x*a1.z, a3.x*a1.y-a3.y*a1.x)/V_box;
    Scalar3 b3 = Scalar(2.0*M_PI)*make_scalar3(a1.y*a2.z-a1.z*a2.y, a1.z*a2.x-a1.x*a2.z, a1.x*a2.y-a1.y*a2.x)/V_box;

    uint3 inner_dim = make_uint3(mesh_idx.getW()-n_ghost_cells.x,
                                 mesh_idx.getH()-n_ghost_cells.y,
                                 mesh_idx.getD()-n_ghost_cells.z);
    
    unsigned int num_wave_vectors = inner_dim.x*inner_dim.y*inner_dim.z;

    unsigned int block_size = 512;
    unsigned int n_blocks = num_wave_vectors/block_size;
    if (num_wave_vectors % block_size) n_blocks += 1;

    if (local_fft)
        gpu_compute_influence_function_kernel<true><<<n_blocks, block_size>>>(mesh_idx,
                                                                              n_ghost_cells,
                                                                              global_dim,
                                                                              d_inf_f,
                                                                              d_k,
                                                                              b1,
                                                                              b2,
                                                                              b3,
                                                                              V_box, 
                                                                              qstarsq
#ifdef ENABLE_MPI
                                                                              , dffti
#endif
   
                                                                              );
#ifdef ENABLE_MPI
    else
        gpu_compute_influence_function_kernel<false><<<n_blocks,block_size>>>(mesh_idx,
                                                                             n_ghost_cells,
                                                                             global_dim,
                                                                             d_inf_f,
                                                                             d_k,
                                                                             b1,
                                                                             b2,
                                                                             b3,
                                                                             V_box,
                                                                             qstarsq,
                                                                             dffti);
#endif 
    } 
