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
__device__ inline Scalar assignTSC(Scalar x)
    {
    Scalar xsq = x*x;
    Scalar fac =(Scalar(3.0/2.0)-copysignf(x,Scalar(1.0)));

    Scalar ret(0.0);
    if (xsq <= Scalar(1.0/4.0))
        ret = Scalar(3.0/4.0) - xsq;
    else if (xsq <= Scalar(9.0/4.0))
        ret = Scalar(1.0/2.0)*fac*fac;

    return ret;
    }

/*! \param x Distance on mesh in units of the mesh size
 */
__device__ inline Scalar assignTSCderiv(Scalar x)
    {
    Scalar xsq = x*x;
    Scalar xabs = copysignf(x,Scalar(1.0));
    Scalar fac =(Scalar(3.0/2.0)-xabs);

    Scalar ret(0.0);
    if (xsq <= Scalar(1.0/4.0))
        ret = -Scalar(2.0)*x;
    else if (xsq <= Scalar(9.0/4.0))
        ret = -fac*x/xabs;

    return ret;
    }


__device__ int3 find_cell(const Scalar3& pos,
                           const unsigned int& inner_nx,
                           const unsigned int& inner_ny,
                           const unsigned int& inner_nz,
                           const uint3& n_ghost_cells,
                           const BoxDim& box
                           )
    {
    // compute coordinates in units of the mesh size
    Scalar3 f = box.makeFraction(pos);
    uchar3 periodic = box.getPeriodic();

    Scalar3 reduced_pos = make_scalar3(f.x * (Scalar)inner_nx,
                                       f.y * (Scalar)inner_ny,
                                       f.z * (Scalar)inner_nz);
    
    reduced_pos += make_scalar3(n_ghost_cells.x/2, n_ghost_cells.y/2, n_ghost_cells.z/2);

    // find cell the particle is in
    int ix = ((reduced_pos.x >= 0) ? reduced_pos.x : (reduced_pos.x - Scalar(1.0)));
    int iy = ((reduced_pos.y >= 0) ? reduced_pos.y : (reduced_pos.y - Scalar(1.0)));
    int iz = ((reduced_pos.z >= 0) ? reduced_pos.z : (reduced_pos.z - Scalar(1.0)));

    // handle particles on the boundary
    if (periodic.x && ix == (int)inner_nx)
        ix = 0;
    if (periodic.y && iy == (int)inner_ny)
        iy = 0;
    if (periodic.z && iz == (int)inner_nz) 
        iz = 0;

    return make_int3(ix, iy, iz);
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

    Scalar4 postype = d_postype[idx];

    Scalar3 pos = make_scalar3(postype.x, postype.y, postype.z);
    unsigned int type = __float_as_int(postype.w);
    Scalar mode = d_mode[type];

    // compute coordinates in units of the mesh size
    int3 cell_coord = find_cell(pos, mesh_idx.getW(), mesh_idx.getH(), mesh_idx.getD(),
                                 make_uint3(0,0,0),box);

    // center of cell (in units of the cell size)
    Scalar3 c = make_scalar3((Scalar)cell_coord.x+Scalar(0.5),
                             (Scalar)cell_coord.y+Scalar(0.5),
                             (Scalar)cell_coord.z+Scalar(0.5));

    Scalar3 p = box.makeFraction(pos)*make_scalar3(mesh_idx.getW(), mesh_idx.getH(), mesh_idx.getD());
    Scalar3 shift = p-c;

    // assign particle to cell and next neighbors
    for (int i = -1; i <= 1 ; ++i)
    	for (int j = -1; j <= 1; ++j)
            for (int k = -1; k <= 1; ++k)
                {
                int neighi = cell_coord.x + i;
                int neighj = cell_coord.y + j;
                int neighk = cell_coord.z + k;

                if (! n_ghost_cells.x)
                    {
                    if (neighi == mesh_idx.getW())
                        neighi = 0;
                    else if (neighi < 0)
                        neighi += mesh_idx.getW();
                    }
                else if (neighi < 0 || neighi >= (int) mesh_idx.getW()) continue;

                if (! n_ghost_cells.y)
                    {
                    if (neighj == mesh_idx.getH())
                        neighj = 0;
                    else if (neighj < 0)
                        neighj += mesh_idx.getH();
                    }
                else if (neighj < 0 || neighj >= (int) mesh_idx.getH()) continue;

                if (! n_ghost_cells.z)
                    {
                    if (neighk == mesh_idx.getD())
                        neighk = 0;
                    else if (neighk < 0)
                        neighk += mesh_idx.getD();
                    }
                else if (neighk < 0 || neighk >= (int) mesh_idx.getD()) continue;
                
                Scalar3 dx_frac = shift - make_scalar3(i,j,k);
                
                // compute fraction of particle density assigned to cell
                Scalar density_fraction = assignTSC(dx_frac.x)*assignTSC(dx_frac.y)*assignTSC(dx_frac.z);

                unsigned int cell_idx;
                if (local_fft)
                    // use cuFFT's memory layout
                    cell_idx = neighk + mesh_idx.getD() * (neighj + mesh_idx.getH() * neighi);
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
                                         const Index2D bin_idx,
                                         const Index3D mesh_idx,
                                         const uint3 n_ghost_bins,
                                         const Scalar *d_mode,
                                         const BoxDim box)
    {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= N) return;
    
    Scalar4 postype = d_postype[idx];

    Scalar3 pos = make_scalar3(postype.x, postype.y, postype.z);
    unsigned int type = __float_as_int(postype.w);
    Scalar mode = d_mode[type];

    int3 bin_dim = make_int3(mesh_idx.getW()+n_ghost_bins.x,
                             mesh_idx.getH()+n_ghost_bins.y,
                             mesh_idx.getD()+n_ghost_bins.z);

    // compute coordinates in units of the cell size
    int3 bin_coord = find_cell(pos, mesh_idx.getW(), mesh_idx.getH(), mesh_idx.getD(), n_ghost_bins, box);

    // ignore ghost particles that are not within our domain
    if (bin_coord.x < 0 || bin_coord.x >= bin_dim.x ||
        bin_coord.y < 0 || bin_coord.y >= bin_dim.y ||
        bin_coord.z < 0 || bin_coord.z >= bin_dim.z) return;

    unsigned int bin = bin_coord.z + bin_dim.z * (bin_coord.y + bin_dim.y * bin_coord.x);

    unsigned int n = atomicInc(&d_n_cell[bin], 0xffffffff);

    if (n >= bin_idx.getH())
        {
        // overflow
        atomicMax(d_overflow, n+1);
        return;
        }

    // store distance to bin center in bin in units of bin size
    Scalar3 f = box.makeFraction(pos);
    f = f*make_scalar3(mesh_idx.getW(), mesh_idx.getH(), mesh_idx.getD());
    Scalar3 c = make_scalar3((Scalar)bin_coord.x + Scalar(0.5) - Scalar(n_ghost_bins.x/2),
                             (Scalar)bin_coord.y + Scalar(0.5) - Scalar(n_ghost_bins.y/2),
                             (Scalar)bin_coord.z + Scalar(0.5) - Scalar(n_ghost_bins.z/2));
    Scalar3 shift = f - c;

    d_particle_bins[bin_idx(bin,n)] = make_scalar4(shift.x,shift.y,shift.z, mode);
    }

void gpu_bin_particles(const unsigned int N,
                       const Scalar4 *d_postype,
                       Scalar4 *d_particle_bins,
                       unsigned int *d_n_cell,
                       unsigned int *d_overflow,
                       const Index2D& bin_idx,
                       const Index3D& mesh_idx,
                       const uint3 n_ghost_bins,
                       const Scalar *d_mode,
                       const BoxDim& box)
    {
    unsigned int block_size = 512;

    gpu_bin_particles_kernel<<<N/block_size+1, block_size>>>(N,
                                                             d_postype,
                                                             d_particle_bins,
                                                             d_n_cell,
                                                             d_overflow,
                                                             bin_idx,
                                                             mesh_idx,
                                                             n_ghost_bins,
                                                             d_mode,
                                                             box);
    }

template<bool local_fft>
__global__ void gpu_assign_binned_particles_to_scratch_kernel(const Index3D mesh_idx,
                                                           const uint3 n_ghost_bins,
                                                           const Scalar4 *d_particle_bins,
                                                           const unsigned int *d_n_cell,
                                                           Scalar *d_mesh_scratch,
                                                           const Index2D bin_idx,
                                                           const Index2D scratch_idx)
    {
    extern __shared__ Scalar scratch_neighbors[];

    unsigned int bin = blockIdx.x * blockDim.x + threadIdx.x;

    if (bin >= bin_idx.getW()) return;

    int3 bin_dim = make_int3(mesh_idx.getW()+n_ghost_bins.x,
                             mesh_idx.getH()+n_ghost_bins.y,
                             mesh_idx.getD()+n_ghost_bins.z);

    // grid coordinates of bin
    int i,j,k;
    i = bin /bin_dim.y / bin_dim.z;
    j = (bin - i * bin_dim.y*bin_dim.z)/bin_dim.z;
    k = bin % bin_dim.z;

    // reset shared memory
    for (unsigned int sidx = 0; sidx < scratch_idx.getH(); ++sidx)
        scratch_neighbors[scratch_idx.getH()*threadIdx.x+sidx] = Scalar(0.0);

    // loop over particles in bin
    unsigned int n_bin = d_n_cell[bin];

    Scalar assign_x, assign_y, assign_z;
    Scalar shift, shift_sq, fac;
    for (unsigned int idx = 0; idx < n_bin; ++idx)
        {
        Scalar4 xyzm = d_particle_bins[bin_idx(bin,idx)];
        
        int neigh_bin_idx = -1;

        // loop over neighboring bins
        for (int l = -1; l <= 1 ; ++l)
            {
            // precalculate assignment factor
            shift = xyzm.x - (Scalar)l;
            shift_sq = shift*shift;
            fac = (Scalar(3.0/2.0)-copysignf(shift,Scalar(1.0)));

            if (!l)
                assign_x = Scalar(3.0/4.0)-shift_sq;
            else
                assign_x = Scalar(1.0/2.0)*fac*fac;
 
            for (int m = -1; m <= 1; ++m)
                {
                shift = xyzm.y - (Scalar)m;
                shift_sq = shift*shift;
                fac = (Scalar(3.0/2.0)-copysignf(shift,Scalar(1.0)));

                if (!m)
                    assign_y = Scalar(3.0/4.0)-shift_sq;
                else
                    assign_y = Scalar(1.0/2.0)*fac*fac;
     
                for (int n = -1; n <= 1; ++n)
                    {
                    shift = xyzm.z - (Scalar)n;
                    shift_sq = shift*shift;
                    fac = (Scalar(3.0/2.0)-copysignf(shift,Scalar(1.0)));

                    if (!n)
                        assign_z = Scalar(3.0/4.0)-shift_sq;
                    else
                        assign_z = Scalar(1.0/2.0)*fac*fac;
     
                    int neighi = i + l;
                    int neighj = j + m;
                    int neighk = k + n;

                    neigh_bin_idx++;
                    if (neighi >= (int)(bin_dim.x-n_ghost_bins.x/2))
                        {
                        if (local_fft || ! n_ghost_bins.x)
                            neighi = 0;
                        else
                            continue;
                        }
                    else if (neighi < (int)(n_ghost_bins.x/2))
                        {
                        if (local_fft || ! n_ghost_bins.x)
                            neighi += (int)bin_dim.x;
                        else
                            continue;
                        }

                    if (neighj >= (int)(bin_dim.y-n_ghost_bins.y/2))
                        {
                        if (local_fft || ! n_ghost_bins.y)
                            neighj = 0;
                        else
                            continue;
                        }
                    else if (neighj < (int)(n_ghost_bins.y/2))
                        {
                        if (local_fft || ! n_ghost_bins.y)
                            neighj += (int)bin_dim.y;
                        else
                            continue;
                        }

                    if (neighk >= (int)(bin_dim.z-n_ghost_bins.z/2))
                        {
                        if (local_fft || ! n_ghost_bins.z)
                            neighk = 0;
                        else
                            continue;
                        }
                    else if (neighk < (int)(n_ghost_bins.z/2))
                        {
                        if (local_fft || ! n_ghost_bins.z)
                            neighk += (int)bin_dim.z;
                        else
                            continue;
                        } 
     
                    // compute fraction of particle density assigned to cell from particles
                    // in this bin
                    Scalar mode = xyzm.w;
                    scratch_neighbors[scratch_idx.getH()*threadIdx.x+neigh_bin_idx] += mode*assign_x*assign_y*assign_z;
                    }
                }
            } // end of loop over neighboring bins
        } // end of ptl loop

    // write out shared memory to neighboring cells
    // loop over neighboring bins
    int neigh_bin_idx = -1;
    for (int l = -1; l <= 1 ; ++l)
        for (int m = -1; m <= 1; ++m)
            for (int n = -1; n <= 1; ++n)
                {
                int neighi = i + l;
                int neighj = j + m;
                int neighk = k + n;

                neigh_bin_idx++;
                if (neighi >= (int)(bin_dim.x-n_ghost_bins.x/2))
                    {
                    if (local_fft || ! n_ghost_bins.x)
                        neighi = 0;
                    else
                        continue;
                    }
                else if (neighi < (int)(n_ghost_bins.x/2))
                    {
                    if (local_fft || ! n_ghost_bins.x)
                        neighi += (int)bin_dim.x;
                    else
                        continue;
                    }

                if (neighj >= (int)(bin_dim.y-n_ghost_bins.y/2))
                    {
                    if (local_fft || ! n_ghost_bins.y)
                        neighj = 0;
                    else
                        continue;
                    }
                else if (neighj < (int)(n_ghost_bins.y/2))
                    {
                    if (local_fft || ! n_ghost_bins.y)
                        neighj += (int)bin_dim.y;
                    else
                        continue;
                    }

                if (neighk >= (int)(bin_dim.z-n_ghost_bins.z/2))
                    {
                    if (local_fft || ! n_ghost_bins.z)
                        neighk = 0;
                    else
                        continue;
                    }
                else if (neighk < (int)(n_ghost_bins.z/2))
                    {
                    if (local_fft || ! n_ghost_bins.z)
                        neighk += (int)bin_dim.z;
                    else
                        continue;
                    } 

                uint3 scratch_cell_coord = make_uint3(neighi - n_ghost_bins.x/2,
                                                      neighj - n_ghost_bins.y/2,
                                                      neighk - n_ghost_bins.z/2);

                // write out to global memory
                unsigned int cell_idx;
                if (local_fft)
                    // use cuFFT's memory layout
                    cell_idx = scratch_cell_coord.z + mesh_idx.getD() * (scratch_cell_coord.y + mesh_idx.getH() * scratch_cell_coord.x);
                else
                    cell_idx = mesh_idx(scratch_cell_coord.x,
                                        scratch_cell_coord.y,
                                        scratch_cell_coord.z);


                d_mesh_scratch[scratch_idx(cell_idx,neigh_bin_idx)] =
                    scratch_neighbors[scratch_idx.getH()*threadIdx.x+neigh_bin_idx];
                }
    }

__global__ void gpu_reduce_scratch_kernel(const Index3D mesh_idx,
                               const Scalar *d_mesh_scratch,
                               const Index2D scratch_idx,
                               cufftComplex *d_mesh)
    {
    unsigned int cell_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (cell_idx >= mesh_idx.getNumElements()) return;

    // simply add up contents of scratch cell
    Scalar grid_val(0.0);
    for (unsigned int sidx = 0; sidx < scratch_idx.getH(); ++sidx)
        grid_val += d_mesh_scratch[scratch_idx(cell_idx,sidx)];

    d_mesh[cell_idx].x = grid_val;
    d_mesh[cell_idx].y = Scalar(0.0);
    }

void gpu_assign_binned_particles_to_mesh(const Index3D& mesh_idx,
                                         const uint3 n_ghost_bins,
                                         const Scalar4 *d_particle_bins,
                                         Scalar *d_mesh_scratch,
                                         const Index2D& bin_idx,
                                         const Index2D& scratch_idx,
                                         const unsigned int *d_n_cell,
                                         cufftComplex *d_mesh,
                                         const bool local_fft)
    {
    uint3 inner_dim = make_uint3(mesh_idx.getW(), mesh_idx.getH(), mesh_idx.getD());

    unsigned int block_size = 64;
    unsigned int n_blocks = bin_idx.getW()/block_size;
    if (bin_idx.getW()%block_size) n_blocks +=1;

    unsigned int shared_size = block_size*scratch_idx.getH()*sizeof(Scalar);

    if (local_fft)
        gpu_assign_binned_particles_to_scratch_kernel<true><<<n_blocks,block_size,shared_size>>>(
                                                                                  mesh_idx,
                                                                                  n_ghost_bins,
                                                                                  d_particle_bins,
                                                                                  d_n_cell,
                                                                                  d_mesh_scratch,
                                                                                  bin_idx,
                                                                                  scratch_idx);
    else
        gpu_assign_binned_particles_to_scratch_kernel<false><<<n_blocks,block_size,shared_size>>>(
                                                                                  mesh_idx,
                                                                                  n_ghost_bins,
                                                                                  d_particle_bins,
                                                                                  d_n_cell,
                                                                                  d_mesh_scratch,
                                                                                  bin_idx,
                                                                                  scratch_idx);

        
    block_size = 512;
    n_blocks = mesh_idx.getNumElements()/block_size;
    if (mesh_idx.getNumElements()%block_size) n_blocks +=1;
    gpu_reduce_scratch_kernel<<<n_blocks, block_size>>>(mesh_idx,
                                                        d_mesh_scratch,
                                                        scratch_idx,
                                                        d_mesh);
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
        
        Scalar ksq = dot(k,k);
        Scalar knorm = sqrtf(ksq);
        Scalar k_cut = sqrtf(qstarsq);
        Scalar fac = expf(-Scalar(12.0)*(knorm/k_cut-Scalar(1.0)));
        Scalar kfac = -Scalar(6.0)/(Scalar(1.0)+fac)/knorm/k_cut;
//        Scalar kfac = -Scalar(1.0/2.0)/qstarsq;
        d_virial_mesh[0*n_wave_vectors+idx] = rhog*kfac*k.x*k.x; // xx
        d_virial_mesh[1*n_wave_vectors+idx] = rhog*kfac*k.x*k.y; // xy
        d_virial_mesh[2*n_wave_vectors+idx] = rhog*kfac*k.x*k.z; // xz
        d_virial_mesh[3*n_wave_vectors+idx] = rhog*kfac*k.y*k.y; // yy
        d_virial_mesh[4*n_wave_vectors+idx] = rhog*kfac*k.y*k.z; // yz
        d_virial_mesh[5*n_wave_vectors+idx] = rhog*kfac*k.z*k.z; // zz
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
                                         const unsigned int N_global)
    {
    unsigned int k = blockDim.x * blockIdx.x + threadIdx.x;

    if (k >= n_wave_vectors) return;

    cufftComplex f = d_fourier_mesh[k];

    // Normalization
    f.x /= (Scalar)N_global;
    f.y /= (Scalar)N_global;
    Scalar val = f.x*f.x+f.y*f.y;

    cufftComplex fourier_G;
    fourier_G.x =f.x * val * d_inf_f[k];
    fourier_G.y =f.y * val * d_inf_f[k];

    d_fourier_mesh[k] = f;
    d_fourier_mesh_G[k] = fourier_G;
    }

void gpu_update_meshes(const unsigned int n_wave_vectors,
                         cufftComplex *d_fourier_mesh,
                         cufftComplex *d_fourier_mesh_G,
                         const Scalar *d_inf_f,
                         const Scalar3 *d_k,
                         const unsigned int N_global)

    {
    const unsigned int block_size = 512;

    gpu_update_meshes_kernel<<<n_wave_vectors/block_size+1, block_size>>>(n_wave_vectors,
                                                                          d_fourier_mesh,
                                                                          d_fourier_mesh_G,
                                                                          d_inf_f,
                                                                          d_k,
                                                                          N_global);
    }

//! Texture for reading particle positions
texture<cufftComplex, 1, cudaReadModeElementType> inv_fourier_mesh_tex;

template<bool local_fft>
__global__ void gpu_compute_forces_kernel(const unsigned int N,
                                          const Scalar4 *d_postype,
                                          Scalar4 *d_force,
                                          const Scalar bias,
                                          const Index3D mesh_idx,
                                          const uint3 n_ghost_cells,
                                          const Scalar *d_mode,
                                          const BoxDim box,
                                          const Scalar V,
                                          const unsigned int n_global,
                                          const Scalar3 b1,
                                          const Scalar3 b2,
                                          const Scalar3 b3)
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
    int3 cell_coord = find_cell(pos, inner_dim.x, inner_dim.y, inner_dim.z, n_ghost_cells, box);

    // center of cell (in units of the cell size)
    Scalar3 c = make_scalar3((Scalar)cell_coord.x-(Scalar)(n_ghost_cells.x/2)+Scalar(0.5),
                             (Scalar)cell_coord.y-(Scalar)(n_ghost_cells.y/2)+Scalar(0.5),
                             (Scalar)cell_coord.z-(Scalar)(n_ghost_cells.z/2)+Scalar(0.5));

    Scalar3 p = box.makeFraction(pos)*make_scalar3(inner_dim.x, inner_dim.y, inner_dim.z);
    Scalar3 shift_c = p-c;

    Scalar3 force = make_scalar3(0.0,0.0,0.0);

    // assign particle to cell and next neighbors
    Scalar assign_x, assign_y, assign_z;
    Scalar assign_x_deriv, assign_y_deriv, assign_z_deriv;
    Scalar shift, shift_abs,shift_sq, fac;
 
    for (int i = -1; i <= 1 ; ++i)
        {
        // precalculate assignment factor
        shift = shift_c.x - (Scalar)i;
        shift_sq = shift*shift;
        shift_abs = copysignf(shift,Scalar(1.0));
        fac = (Scalar(3.0/2.0)-shift_abs);

        if (!i)
            {
            assign_x = Scalar(3.0/4.0)-shift_sq;
            assign_x_deriv = -Scalar(2.0)*shift;
            }
        else
            {
            assign_x = Scalar(1.0/2.0)*fac*fac;
            assign_x_deriv = -fac*shift/shift_abs;
            }
 
    	for (int j = -1; j <= 1; ++j)
            {
            shift = shift_c.y - (Scalar)j;
            shift_sq = shift*shift;
            shift_abs = copysignf(shift,Scalar(1.0));
            fac = (Scalar(3.0/2.0)-shift_abs);

            if (!j)
                {
                assign_y = Scalar(3.0/4.0)-shift_sq;
                assign_y_deriv = -Scalar(2.0)*shift;
                }
            else
                {
                assign_y = Scalar(1.0/2.0)*fac*fac;
                assign_y_deriv = -fac*shift/shift_abs;
                }
     
            for (int k = -1; k <= 1; ++k)
                {
                shift = shift_c.z - (Scalar)k;
                shift_sq = shift*shift;
                shift_abs = copysignf(shift,Scalar(1.0));
                fac = (Scalar(3.0/2.0)-shift_abs);

                if (!k)
                    {
                    assign_z = Scalar(3.0/4.0)-shift_sq;
                    assign_z_deriv = -Scalar(2.0)*shift;
                    }
                else
                    {
                    assign_z = Scalar(1.0/2.0)*fac*fac;
                    assign_z_deriv = -fac*shift/shift_abs;
                    }

                int neighi = (int) cell_coord.x + i;
                int neighj = (int) cell_coord.y + j;
                int neighk = (int) cell_coord.z + k;

                if (local_fft || ! n_ghost_cells.x)
                    {
                    if (neighi == inner_dim.x)
                        neighi = 0;
                    else if (neighi < 0)
                        neighi += inner_dim.x;
                    }

                if (local_fft || ! n_ghost_cells.y)
                    {
                    if (neighj == inner_dim.y)
                        neighj = 0;
                    else if (neighj < 0)
                        neighj += inner_dim.y;
                    }

                if (local_fft || ! n_ghost_cells.z)
                    {
                    if (neighk == inner_dim.z)
                        neighk = 0;
                    else if (neighk < 0)
                        neighk += inner_dim.z;
                    } 


                // compute fraction of particle density assigned to cell
                unsigned int cell_idx;
                if (local_fft)
                    // use cuFFT's memory layout
                    cell_idx = neighk + inner_dim.z * (neighj + inner_dim.y * neighi);
                else
                    cell_idx = mesh_idx(neighi, neighj, neighk);

                cufftComplex inv_mesh = tex1Dfetch(inv_fourier_mesh_tex,cell_idx);

                force += -(Scalar)inner_dim.x*b1*mode*assign_x_deriv*assign_y*assign_z*inv_mesh.x;
                force += -(Scalar)inner_dim.y*b2*mode*assign_x*assign_y_deriv*assign_z*inv_mesh.x;
                force += -(Scalar)inner_dim.z*b3*mode*assign_x*assign_y*assign_z_deriv*inv_mesh.x;
                }  
            }
        } // end neighbor cells loop

    // Normalization
    force *= Scalar(2.0)/(Scalar)n_global;

    // Multiply with bias potential derivative
    force *= bias;

    d_force[idx] = make_scalar4(force.x,force.y,force.z,0.0);
    }

void gpu_compute_forces(const unsigned int N,
                        const Scalar4 *d_postype,
                        Scalar4 *d_force,
                        const Scalar bias,
                        const cufftComplex *d_inv_fourier_mesh,
                        const Index3D& mesh_idx,
                        const uint3 n_ghost_cells,
                        const Scalar *d_mode,
                        const BoxDim& box,
                        const BoxDim& global_box,
                        const bool local_fft,
                        const unsigned int n_global)
    {
    const unsigned int block_size = 512;

    // force mesh includes ghost cells
    unsigned int num_cells = mesh_idx.getNumElements();
    inv_fourier_mesh_tex.normalized = false;
    inv_fourier_mesh_tex.filterMode = cudaFilterModePoint;
    cudaBindTexture(0, inv_fourier_mesh_tex, d_inv_fourier_mesh, sizeof(Scalar4)*num_cells);

    // compute local inverse lattice vectors
    Scalar3 a1 = box.getLatticeVector(0);
    Scalar3 a2 = box.getLatticeVector(1);
    Scalar3 a3 = box.getLatticeVector(2);

    Scalar V_box = box.getVolume();
    Scalar3 b1 = make_scalar3(a2.y*a3.z-a2.z*a3.y, a2.z*a3.x-a2.x*a3.z, a2.x*a3.y-a2.y*a3.x)/V_box;
    Scalar3 b2 = make_scalar3(a3.y*a1.z-a3.z*a1.y, a3.z*a1.x-a3.x*a1.z, a3.x*a1.y-a3.y*a1.x)/V_box;
    Scalar3 b3 = make_scalar3(a1.y*a2.z-a1.z*a2.y, a1.z*a2.x-a1.x*a2.z, a1.x*a2.y-a1.y*a2.x)/V_box;

    if (local_fft)
        gpu_compute_forces_kernel<true><<<N/block_size+1,block_size>>>(N,
                                                                     d_postype,
                                                                     d_force,
                                                                     bias,
                                                                     mesh_idx,
                                                                     n_ghost_cells,
                                                                     d_mode,
                                                                     box,
                                                                     global_box.getVolume(),
                                                                     n_global,
                                                                     b1,
                                                                     b2,
                                                                     b3);
    else
        gpu_compute_forces_kernel<false><<<N/block_size+1,block_size>>>(N,
                                                                     d_postype,
                                                                     d_force,
                                                                     bias,
                                                                     mesh_idx,
                                                                     n_ghost_cells,
                                                                     d_mode,
                                                                     box,
                                                                     global_box.getVolume(),
                                                                     n_global,
                                                                     b1,
                                                                     b2,
                                                                     b3);

    cudaUnbindTexture(inv_fourier_mesh_tex);
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

    // write result to global memory
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
                   const unsigned int block_size)
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
//    return expf(-ksq/qstarsq*Scalar(1.0/2.0));
    Scalar knorm = sqrtf(ksq);
    Scalar k_cut = sqrtf(qstarsq);
    return Scalar(1.0)/(Scalar(1.0)+expf(Scalar(12.0)*(knorm/k_cut-Scalar(1.0))));
    }

template<bool local_fft>
__global__ void gpu_compute_influence_function_kernel(const Index3D mesh_idx,
                                          const unsigned int n_wave_vectors,
                                          const uint3 global_dim,
                                          Scalar *d_inf_f,
                                          Scalar3 *d_k,
                                          const Scalar3 b1,
                                          const Scalar3 b2,
                                          const Scalar3 b3,
                                          const Scalar qstarsq,
                                          const int3 *zero_modes,
                                          const unsigned int n_zero_modes
#ifdef ENABLE_MPI
                                          , const DFFTIndex dffti
#endif
                                          )
    {
    unsigned int kidx = blockIdx.x * blockDim.x + threadIdx.x;

    if (kidx >= n_wave_vectors) return;

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
    
    bool zero = false;
    for (unsigned int i = 0; i < n_zero_modes; ++i)
        {
        if (zero_modes[i].x == l && zero_modes[i].y == m && zero_modes[i].z == n)
            {
            zero = true;
            break;
            }
        }

    Scalar val(0.0);
    Scalar3 kval = (Scalar)l*b1+(Scalar)m*b2+(Scalar)n*b3;
    unsigned int cell_idx;
    if (local_fft)
        // use cuFFT's memory layout
        cell_idx = iz + mesh_idx.getD() * (iy + mesh_idx.getH()* ix);
    else
        cell_idx = kidx;

    if (!zero)
        {
        Scalar ksq = dot(kval,kval);
        val = convolution_kernel(ksq,qstarsq);
        }

    // write out result
    d_inf_f[cell_idx] = val;
    d_k[cell_idx] = kval;
    }

void gpu_compute_influence_function(const Index3D& mesh_idx,
                                    const uint3 global_dim,
                                    Scalar *d_inf_f,
                                    Scalar3 *d_k,
                                    const BoxDim& global_box,
                                    const Scalar qstarsq,
                                    const int3 *d_zero_modes,
                                    const unsigned int n_zero_modes,
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

    unsigned int num_wave_vectors = mesh_idx.getW()*mesh_idx.getH()*mesh_idx.getD();

    unsigned int block_size = 512;
    unsigned int n_blocks = num_wave_vectors/block_size;
    if (num_wave_vectors % block_size) n_blocks += 1;

    if (local_fft)
        gpu_compute_influence_function_kernel<true><<<n_blocks, block_size>>>(mesh_idx,
                                                                              num_wave_vectors,
                                                                              global_dim,
                                                                              d_inf_f,
                                                                              d_k,
                                                                              b1,
                                                                              b2,
                                                                              b3,
                                                                              qstarsq,
                                                                              d_zero_modes,
                                                                              n_zero_modes
#ifdef ENABLE_MPI
                                                                              , dffti
#endif
   
                                                                              );
#ifdef ENABLE_MPI
    else
        gpu_compute_influence_function_kernel<false><<<n_blocks,block_size>>>(mesh_idx,
                                                                             num_wave_vectors,
                                                                             global_dim,
                                                                             d_inf_f,
                                                                             d_k,
                                                                             b1,
                                                                             b2,
                                                                             b3,
                                                                             qstarsq,
                                                                             d_zero_modes,
                                                                             n_zero_modes,
                                                                             dffti);
#endif 
    } 

__global__ void gpu_compute_qmax_partial_kernel(
            int n_wave_vectors,
            Scalar4 *max_partial,
            const Scalar3 *d_k,
            const cufftComplex *d_fourier_mesh)
    {
    extern __shared__ Scalar4 sdata_max[];

    unsigned int tidx = threadIdx.x;

    int j = blockIdx.x * blockDim.x + threadIdx.x;

    Scalar4 max_q = make_scalar4(0.0,0.0,0.0,0.0);

    if (j < n_wave_vectors) {
        Scalar a = d_fourier_mesh[j].x * d_fourier_mesh[j].x + d_fourier_mesh[j].y * d_fourier_mesh[j].y;
        Scalar3 k = d_k[j];
        max_q = make_scalar4(k.x,k.y,k.z,a);
        }

    sdata_max[tidx] = max_q;

   __syncthreads();

    // reduce the sum
    int offs = blockDim.x >> 1;
    while (offs > 0)
        {
        if (tidx < offs)
            {
            sdata_max[tidx] = (sdata_max[tidx].w > sdata_max[tidx + offs].w) ? sdata_max[tidx] : sdata_max[tidx + offs];
            }
        offs >>= 1;
        __syncthreads();
        }

    // write result to global memeory
    if (tidx == 0)
       max_partial[blockIdx.x] = sdata_max[0];
    }

__global__ void gpu_compute_qmax_final_kernel(Scalar4* max_partial,
                                       unsigned int nblocks,
                                       Scalar4 *q_max)
    {
    extern __shared__ Scalar4 sdata_max[];

    if (threadIdx.x == 0)
       *q_max = make_scalar4(0.0,0.0,0.0,0.0);

    for (int start = 0; start< nblocks; start += blockDim.x)
        {
        __syncthreads();
        if (start + threadIdx.x < nblocks)
            sdata_max[threadIdx.x] = max_partial[start + threadIdx.x];
        else
            sdata_max[threadIdx.x] = make_scalar4(0.0,0.0,0.0,0.0);

        __syncthreads();

        // reduce the sum
        int offs = blockDim.x >> 1;
        while (offs > 0)
            {
            if (threadIdx.x < offs)
                sdata_max[threadIdx.x] = (sdata_max[threadIdx.x].w > sdata_max[threadIdx.x + offs].w) ?
                                         sdata_max[threadIdx.x] : sdata_max[threadIdx.x + offs];
            offs >>= 1;
            __syncthreads();
            }

         if (threadIdx.x == 0)
            {
            Scalar4 old_qmax = *q_max;
            *q_max = (old_qmax.w > sdata_max[0].w) ? old_qmax : sdata_max[0];
            }
        }
    }

void gpu_compute_q_max(unsigned int n_wave_vectors,
                   Scalar4 *d_max_partial,
                   Scalar4 *d_q_max,
                   const Scalar3 *d_k,
                   const cufftComplex *d_fourier_mesh,
                   const unsigned int block_size)
    {
    unsigned int n_blocks = n_wave_vectors/block_size + 1;

    unsigned int shared_size = block_size * sizeof(Scalar4);
    gpu_compute_qmax_partial_kernel<<<n_blocks, block_size, shared_size>>>(
               n_wave_vectors,
               d_max_partial,
               d_k,
               d_fourier_mesh);

    // calculate final sum of mesh values
    const unsigned int final_block_size = 512;
    shared_size = final_block_size*sizeof(Scalar4);
    gpu_compute_qmax_final_kernel<<<1, final_block_size,shared_size>>>(d_max_partial,
                                                                n_blocks,
                                                                d_q_max);
    }


