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

__device__ uint3 find_cell(Scalar3 pos,
                           unsigned int nx,
                           unsigned int ny,
                           unsigned int nz,
                           const BoxDim box
                           )
    {
    // compute coordinates in units of the mesh size
    Scalar3 f = box.makeFraction(pos);
    Scalar3 reduced_pos = make_scalar3(f.x * (Scalar)nx,
                                       f.y * (Scalar)ny,
                                       f.z * (Scalar)nz);

    // find cell the particle is in
    unsigned int ix = reduced_pos.x;
    unsigned int iy = reduced_pos.y;
    unsigned int iz = reduced_pos.z;

    // handle particles on the boundary
    if (ix == nx)
        ix = 0;
    if (iy == ny)
        iy = 0;
    if (iz == nz) 
        iz = 0;

    return make_uint3(ix, iy, iz);
    }

//! Assignment of particles to mesh using three-point scheme (triangular shaped cloud)
/*! This is a second order accurate scheme with continuous value and continuous derivative
 */
__global__ void gpu_assign_particles_kernel(const unsigned int N,
                                       const Scalar4 *d_postype,
                                       cufftReal *d_mesh,
                                       const Index3D mesh_idx,
                                       const Scalar *d_mode,
                                       const BoxDim box)
    {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= N) return;

    int3 dim = make_int3(mesh_idx.getW(), mesh_idx.getH(), mesh_idx.getD());

    Scalar V_cell = box.getVolume()/(Scalar)mesh_idx.getNumElements();
 
    // inverse dimensions
    Scalar3 dim_inv = make_scalar3(Scalar(1.0)/(Scalar)dim.x,
                                   Scalar(1.0)/(Scalar)dim.y,
                                   Scalar(1.0)/(Scalar)dim.z);

    Scalar4 postype = d_postype[idx];

    Scalar3 pos = make_scalar3(postype.x, postype.y, postype.z);
    unsigned int type = __float_as_int(postype.w);
    Scalar mode = d_mode[type];

    // compute coordinates in units of the mesh size
    uint3 cell_coord = find_cell(pos, dim.x, dim.y, dim.z, box);

    // center of cell
    Scalar3 c = box.makeCoordinates(make_scalar3((Scalar)cell_coord.x+Scalar(0.5),
                                                 (Scalar)cell_coord.y+Scalar(0.5),
                                                 (Scalar)cell_coord.z+Scalar(0.5))*dim_inv);
    Scalar3 shift = box.minImage(pos - c);
    Scalar3 shift_frac = box.makeFraction(shift) - make_scalar3(0.5,0.5,0.5);
    shift_frac *= make_scalar3(dim.x,dim.y,dim.z);

    // assign particle to cell and next neighbors
    for (int i = -1; i <= 1 ; ++i)
    	for (int j = -1; j <= 1; ++j)
            for (int k = -1; k <= 1; ++k)
                {
                int neighi = cell_coord.x + i;
                int neighj = cell_coord.y + j;
                int neighk = cell_coord.z + k;

                if (neighi == dim.x)
                    neighi = 0;
                else if (neighi < 0)
                    neighi += dim.x;

                if (neighj == dim.y)
                    neighj = 0;
                else if (neighj < 0)
                    neighj += dim.y;

                if (neighk == dim.z)
                    neighk = 0;
                else if (neighk < 0)
                    neighk += dim.z;
            
                Scalar3 dx_frac = shift_frac - make_scalar3(i,j,k);
                
                // compute fraction of particle density assigned to cell
                Scalar density_fraction = assignTSC(dx_frac.x)*assignTSC(dx_frac.y)*assignTSC(dx_frac.z)/V_cell;
                unsigned int cell_idx = neighk + dim.z * (neighj + dim.y * neighi);

                atomicFloatAdd(&d_mesh[cell_idx], mode*density_fraction);
                }
                 
    }

__global__ void gpu_bin_particles_kernel(const unsigned int N,
                                         const Scalar4 *d_postype,
                                         Scalar4 *d_particle_bins,
                                         unsigned int *d_n_cell,
                                         unsigned int *d_overflow,
                                         const unsigned int maxn,
                                         const Index3D mesh_idx,
                                         const Scalar *d_mode,
                                         const BoxDim box)
    {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= N) return;

    int3 dim = make_int3(mesh_idx.getW(), mesh_idx.getH(), mesh_idx.getD());
 
    Scalar4 postype = d_postype[idx];

    Scalar3 pos = make_scalar3(postype.x, postype.y, postype.z);
    unsigned int type = __float_as_int(postype.w);
    Scalar mode = d_mode[type];

    // compute coordinates in units of the mesh size
    uint3 cell_coord = find_cell(pos, dim.x, dim.y, dim.z, box);

    unsigned int cell_idx = cell_coord.z + dim.z * (cell_coord.y + dim.y * cell_coord.x);

    unsigned int n = atomicInc(&d_n_cell[cell_idx], 0xffffffff);

    if (n >= maxn)
        {
        // overflow
        atomicMax(d_overflow, n+1);
        }
    else
        {
        // store distance to cell center in bin in units of cell dimensions
        Scalar3 f = box.makeFraction(pos);
        f = f*make_scalar3((Scalar)dim.x,(Scalar)dim.y,(Scalar)dim.z);
        Scalar3 c = make_scalar3((Scalar)cell_coord.x+Scalar(0.5),
                                 (Scalar)cell_coord.y+Scalar(0.5),
                                 (Scalar)cell_coord.z+Scalar(0.5));
        Scalar3 shift = f - c;
        uchar3 periodic = box.getPeriodic();

        if (periodic.x && shift.x > Scalar(1.0))
            shift.x -= (Scalar)dim.x;
        if (periodic.y && shift.y > Scalar(1.0))
            shift.y -= (Scalar)dim.y;
        if (periodic.z && shift.z > Scalar(1.0))
            shift.z -= (Scalar)dim.z;

        d_particle_bins[cell_idx*maxn+n] = make_scalar4(shift.x,shift.y,shift.z, mode);
        }
    }

void gpu_bin_particles(const unsigned int N,
                       const Scalar4 *d_postype,
                       Scalar4 *d_particle_bins,
                       unsigned int *d_n_cell,
                       unsigned int *d_overflow,
                       const unsigned int maxn,
                       const Index3D& mesh_idx,
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
                                                             d_mode,
                                                             box);
    }


texture<Scalar4, 1, cudaReadModeElementType> particle_bins_tex;

__global__ void gpu_assign_binned_particles_to_mesh_kernel(unsigned int nx,
                                                           unsigned int ny,
                                                           unsigned int nz,
                                                           const unsigned int *d_n_cell,
                                                           const unsigned int maxn,
                                                           cufftReal *d_mesh,
                                                           const BoxDim box)
    {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i>= (int)nx || j >= (int)ny || k >= (int)nz) return;

    unsigned int cell_idx = k + nz * (j + ny * i);

    Scalar V_cell = box.getVolume()/(Scalar)(nx*ny*nz);
    Scalar grid_val(0.0);

    // loop over particles in neighboring bins
    for (int l = -1; l <= 1 ; ++l)
    	for (int m = -1; m <= 1; ++m)
            for (int n = -1; n <= 1; ++n)
                {
                int neighi = i + l;
                int neighj = j + m;
                int neighk = k + n;

                if (neighi == nx)
                    neighi = 0;
                else if (neighi < 0)
                    neighi += nx; 

                if (neighj == ny)
                    neighj = 0;
                else if (neighj < 0)
                    neighj += ny;

                if (neighk == nz)
                    neighk = 0;
                else if (neighk < 0)
                    neighk += nz;

                    
                unsigned int neigh_cell_idx = neighk + nz * (neighj + ny * neighi);
                unsigned int n_cell = d_n_cell[neigh_cell_idx];
                Scalar3 cell_shift = make_scalar3(l,m,n);

                for (unsigned int neigh_idx = 0; neigh_idx < n_cell; neigh_idx++)
                    {
                    Scalar4 xyzm = tex1Dfetch(particle_bins_tex, maxn*neigh_cell_idx+neigh_idx);
                    Scalar3 shift_frac = make_scalar3(xyzm.x, xyzm.y, xyzm.z);

                    Scalar3 dx_frac = shift_frac + cell_shift;

                    // compute fraction of particle density assigned to cell
                    Scalar mode = xyzm.w;
                    grid_val += mode*assignTSC(dx_frac.x)*assignTSC(dx_frac.y)*assignTSC(dx_frac.z)/V_cell;
                    }
                } // end of loop over neighboring bins
  
    // write out mesh value
    d_mesh[cell_idx] = grid_val;
    }

void gpu_assign_binned_particles_to_mesh(const Index3D& mesh_idx,
                                         const Scalar4 *d_particle_bins,
                                         const unsigned int *d_n_cell,
                                         const unsigned int maxn,
                                         cufftReal *d_mesh,
                                         const BoxDim& box)
    {
    unsigned int num_cells = mesh_idx.getNumElements();

    particle_bins_tex.normalized = false;
    particle_bins_tex.filterMode = cudaFilterModePoint;
    cudaBindTexture(0, particle_bins_tex, d_particle_bins, sizeof(Scalar4)*num_cells*maxn);

    unsigned int block_size = 8;
    
    dim3 blockDim(block_size,block_size,block_size);
    dim3 gridDim((mesh_idx.getW() % block_size == 0) ? mesh_idx.getW()/block_size : mesh_idx.getW()/block_size+1,
                 (mesh_idx.getH() % block_size == 0) ? mesh_idx.getH()/block_size : mesh_idx.getH()/block_size+1,
                 (mesh_idx.getD() % block_size == 0) ? mesh_idx.getD()/block_size : mesh_idx.getW()/block_size+1);

    gpu_assign_binned_particles_to_mesh_kernel<<<gridDim,blockDim>>>(mesh_idx.getW(),
                                                                     mesh_idx.getH(),
                                                                     mesh_idx.getD(),
                                                                     d_n_cell,
                                                                     maxn,
                                                                     d_mesh,
                                                                     box);
    }

void gpu_assign_particles_30(const unsigned int N,
                          const Scalar4 *d_postype,
                          cufftReal *d_mesh,
                          const Index3D& mesh_idx,
                          const Scalar *d_mode,
                          const BoxDim& box)
    {

    unsigned int block_size = 512;

    gpu_assign_particles_kernel<<<N/block_size+1, block_size>>>(N,
                                                                d_postype,
                                                                d_mesh,
                                                                mesh_idx,
                                                                d_mode,
                                                                box);
    }

__global__ void gpu_compute_mesh_virial_kernel(const unsigned int n_wave_vectors,
                                         cufftComplex *d_fourier_mesh,
                                         cufftComplex *d_fourier_mesh_G,
                                         Scalar *d_virial_mesh,
                                         const Scalar3 *d_k,
                                         const Scalar qstarsq)
    {
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx >= n_wave_vectors) return;

    if (idx != 0)
        {
        // non-zero wave vector
        cufftComplex f_g = d_fourier_mesh_G[idx];
        cufftComplex f = d_fourier_mesh[idx];

        Scalar rhog = f_g.x * f.x + f_g.y * f.y;
        Scalar3 k = d_k[idx];
        Scalar ksq = dot(k,k);
        Scalar kfac = Scalar(2.0)*(Scalar(1.0)+Scalar(1.0/2.0)*ksq/qstarsq)/ksq;
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
                             const Scalar qstarsq)
    {
    const unsigned int block_size = 512;

    gpu_compute_mesh_virial_kernel<<<n_wave_vectors/block_size+1, block_size>>>(n_wave_vectors,
                                                                          d_fourier_mesh,
                                                                          d_fourier_mesh_G,
                                                                          d_virial_mesh,
                                                                          d_k,
                                                                          qstarsq);
    }
 
__global__ void gpu_update_meshes_kernel(const unsigned int n_wave_vectors,
                                         cufftComplex *d_fourier_mesh,
                                         cufftComplex *d_fourier_mesh_G,
                                         const Scalar *d_inf_f,
                                         const Scalar3 *d_k,
                                         const Scalar V_cell,
                                         cufftComplex *d_ifourier_mesh_force)
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
    d_ifourier_mesh_force[k].x = -fourier_G.y*kval.x;
    d_ifourier_mesh_force[k].y = fourier_G.x*kval.x;

    d_ifourier_mesh_force[k+n_wave_vectors].x = -fourier_G.y*kval.y;
    d_ifourier_mesh_force[k+n_wave_vectors].y = fourier_G.x*kval.y;

    d_ifourier_mesh_force[k+2*n_wave_vectors].x = -fourier_G.y*kval.z;
    d_ifourier_mesh_force[k+2*n_wave_vectors].y = fourier_G.x*kval.z;

    d_fourier_mesh[k] = f;
    d_fourier_mesh_G[k] = fourier_G;
    }

void gpu_update_meshes(const unsigned int n_wave_vectors,
                         cufftComplex *d_fourier_mesh,
                         cufftComplex *d_fourier_mesh_G,
                         const Scalar *d_inf_f,
                         const Scalar3 *d_k,
                         const Scalar V_cell,
                         cufftComplex *d_ifourier_mesh_force)
    {
    const unsigned int block_size = 512;

    gpu_update_meshes_kernel<<<n_wave_vectors/block_size+1, block_size>>>(n_wave_vectors,
                                                                          d_fourier_mesh,
                                                                          d_fourier_mesh_G,
                                                                          d_inf_f,
                                                                          d_k,
                                                                          V_cell,
                                                                          d_ifourier_mesh_force);
    }

//! Texture for reading particle positions
texture<Scalar4, 1, cudaReadModeElementType> force_mesh_tex;

__global__ void gpu_coalesce_forces_kernel(const unsigned int n_wave_vectors,
                                       const cufftReal *d_ifourier_mesh_force,
                                       Scalar4 *d_force_mesh)
    {
    unsigned int k = blockIdx.x*blockDim.x+threadIdx.x;

    if (k >= n_wave_vectors) return;

    d_force_mesh[k] = make_scalar4(d_ifourier_mesh_force[k],
                                   d_ifourier_mesh_force[k+n_wave_vectors],
                                   d_ifourier_mesh_force[k+2*n_wave_vectors],
                                   0.0);
    }

__global__ void gpu_interpolate_forces_kernel(const unsigned int N,
                                       const Scalar4 *d_postype,
                                       Scalar4 *d_force,
                                       const Scalar bias,
                                       const Index3D mesh_idx,
                                       const Scalar *d_mode,
                                       const BoxDim box)
    {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= N) return;

    Scalar V = box.getVolume();
 
    int3 dim = make_int3(mesh_idx.getW(), mesh_idx.getH(), mesh_idx.getD());

    // inverse dimensions
    Scalar3 dim_inv = make_scalar3(Scalar(1.0)/(Scalar)mesh_idx.getW(),
                                   Scalar(1.0)/(Scalar)mesh_idx.getH(),
                                   Scalar(1.0)/(Scalar)mesh_idx.getD());

    Scalar4 postype = d_postype[idx];

    Scalar3 pos = make_scalar3(postype.x, postype.y, postype.z);
    unsigned int type = __float_as_int(postype.w);
    Scalar mode = d_mode[type];

    // compute coordinates in units of the mesh size
    Scalar3 f = box.makeFraction(pos);
    Scalar3 reduced_pos = make_scalar3(f.x * (Scalar) dim.x,
                                       f.y * (Scalar) dim.y,
                                       f.z * (Scalar) dim.z);

    // find cell the particle is in
    unsigned int ix = reduced_pos.x;
    unsigned int iy = reduced_pos.y;
    unsigned int iz = reduced_pos.z;

    // handle particles on the boundary
    if (ix == mesh_idx.getW())
        ix = 0;
    if (iy == mesh_idx.getH())
        iy = 0;
    if (iz == mesh_idx.getD()) 
        iz = 0;

    // center of cell (in units of the mesh size)
    Scalar3 c = box.makeCoordinates(make_scalar3((Scalar)ix+Scalar(0.5),
                                                 (Scalar)iy+Scalar(0.5),
                                                 (Scalar)iz+Scalar(0.5))*dim_inv);
    Scalar3 shift = box.minImage(pos - c);
    Scalar3 shift_frac = box.makeFraction(shift) - make_scalar3(0.5,0.5,0.5);
    shift_frac *= make_scalar3(dim.x,dim.y,dim.z);

    Scalar3 force = make_scalar3(0.0,0.0,0.0);

    // assign particle to cell and next neighbors
    for (int i = -1; i <= 1 ; ++i)
    	for (int j = -1; j <= 1; ++j)
            for (int k = -1; k <= 1; ++k)
                {
                int neighi = ix + i;
                int neighj = iy + j;
                int neighk = iz + k;

                if (neighi == dim.x)
                    neighi = 0;
                else if (neighi < 0)
                    neighi += dim.x;

                if (neighj == dim.y)
                    neighj = 0;
                else if (neighj < 0)
                    neighj += dim.y;

                if (neighk == dim.z)
                    neighk = 0;
                else if (neighk < 0)
                    neighk += dim.z;
            
                Scalar3 dx_frac = shift_frac - make_scalar3(i,j,k);

                // compute fraction of particle density assigned to cell
                unsigned int cell_idx = neighk + dim.z * (neighj + dim.y * neighi);

                Scalar4 mesh_force = tex1Dfetch(force_mesh_tex,cell_idx);

                force += -assignTSC(dx_frac.x)*assignTSC(dx_frac.y)*assignTSC(dx_frac.z)*mode*
                        make_scalar3(mesh_force.x,mesh_force.y,mesh_force.z);
                }  

    // Multiply with bias potential derivative
    force *= bias/V;

    d_force[idx] = make_scalar4(force.x,force.y,force.z,0.0);
    }

void gpu_interpolate_forces(const unsigned int N,
                             const Scalar4 *d_postype,
                             Scalar4 *d_force,
                             const Scalar bias,
                             const cufftReal *d_ifourier_mesh_force,
                             Scalar4 *d_force_mesh,
                             const Index3D& mesh_idx,
                             const Scalar *d_mode,
                             const BoxDim& box)
    {
    const unsigned int block_size = 512;

    unsigned int num_cells = mesh_idx.getNumElements();

    gpu_coalesce_forces_kernel<<<num_cells/block_size+1, block_size>>>(num_cells,
                                                                 d_ifourier_mesh_force,
                                                                 d_force_mesh);
    force_mesh_tex.normalized = false;
    force_mesh_tex.filterMode = cudaFilterModePoint;
    cudaBindTexture(0, force_mesh_tex, d_force_mesh, sizeof(Scalar4)*num_cells);

    gpu_interpolate_forces_kernel<<<N/block_size+1,block_size>>>(N,
                                                                 d_postype,
                                                                 d_force,
                                                                 bias,
                                                                 mesh_idx,
                                                                 d_mode,
                                                                 box);
    }

__global__ void kernel_calculate_cv_partial(
            int n_wave_vectors,
            Scalar *sum_partial,
            const cufftComplex *d_fourier_mesh,
            const cufftComplex *d_fourier_mesh_G,
            const unsigned int dimz)
    {
    extern __shared__ Scalar sdata[];

    unsigned int tidx = threadIdx.x;

    int j = blockIdx.x * blockDim.x + threadIdx.x;

    Scalar mySum = Scalar(0.0);

    if (j < n_wave_vectors) {
        mySum = d_fourier_mesh[j].x * d_fourier_mesh_G[j].x + d_fourier_mesh[j].y * d_fourier_mesh_G[j].y;
        if (j % dimz) mySum *= Scalar(2.0);
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
                   const Index3D& mesh_idx)
    {
    unsigned int n_blocks = n_wave_vectors/block_size + 1;

    unsigned int shared_size = block_size * sizeof(Scalar);
    kernel_calculate_cv_partial<<<n_blocks, block_size, shared_size>>>(
               n_wave_vectors,
               d_sum_partial,
               d_fourier_mesh,
               d_fourier_mesh_G,
               mesh_idx.getD()/2+1
               );

    // calculate final S(q) values 
    const unsigned int final_block_size = 512;
    shared_size = final_block_size*sizeof(Scalar);
    kernel_final_reduce_cv<<<1, final_block_size,shared_size>>>(d_sum_partial,
                                                                  n_blocks,
                                                                  d_sum);
    }

__global__ void kernel_calculate_virial_partial(
            int n_wave_vectors,
            Scalar *sum_virial_partial,
            const Scalar *d_mesh_virial,
            const unsigned int dimz)
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

        if (j % dimz)
            {
            mySum_xx *= Scalar(2.0);
            mySum_xy *= Scalar(2.0);
            mySum_xz *= Scalar(2.0);
            mySum_yy *= Scalar(2.0);
            mySum_yz *= Scalar(2.0);
            mySum_zz *= Scalar(2.0);
            }
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
               d_mesh_virial,
               mesh_idx.getD()/2+1
               );

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

__global__ void gpu_compute_influence_function_kernel(const Index3D mesh_idx,
                                          const unsigned int N,
                                          Scalar *d_inf_f,
                                          Scalar3 *d_k,
                                          const Scalar3 b1,
                                          const Scalar3 b2,
                                          const Scalar3 b3,
                                          const BoxDim box,
                                          const Scalar qstarsq)
    {
    int l = blockIdx.x * blockDim.x + threadIdx.x;
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.z * blockDim.z + threadIdx.z;

    int3 dim = make_int3(mesh_idx.getW(), mesh_idx.getH(), mesh_idx.getD());
    if (l >= dim.x/2+1 || m >= dim.y/2+1 || n >= dim.z/2+1) return;

    Scalar3 dim_inv = make_scalar3(Scalar(1.0)/(Scalar)dim.x,
                                   Scalar(1.0)/(Scalar)dim.y,
                                   Scalar(1.0)/(Scalar)dim.z);

    Scalar V_box = box.getVolume();


    Scalar3 kval = (Scalar)l*b1+(Scalar)m*b2+(Scalar)n*b3;
    Scalar ksq = dot(kval,kval);

    Scalar Nsq = (Scalar)N*(Scalar)N;

    Scalar val = convolution_kernel(ksq,qstarsq)/Nsq/Nsq*V_box;

    int numi = (l > 0) ? 2 : 1;
    int numj = (m > 0) ? 2 : 1;

    for (int i = 0; i < numi; ++i)
        {
        for (int j = 0; j < numj; ++j)
            {
            // determine cell idx
            unsigned int ix, iy, iz;
            if (l < 0)
                ix = l + dim.x;
            else
                ix = l;

            if (m < 0)
                iy = m + dim.y;
            else
                iy = m;

            iz = n;
            
            unsigned int cell_idx = iz + (dim.z/2+1) * (iy + dim.y * ix);

            d_inf_f[cell_idx] = val;

            kval = (Scalar)l*b1+(Scalar)m*b2+(Scalar)n*b3;
            d_k[cell_idx] = kval;

            m *= -1;
            }
        l *= -1;
        }
    }

void gpu_compute_influence_function(const Index3D& mesh_idx,
                                    const unsigned int N,
                                    Scalar *d_inf_f,
                                    Scalar3 *d_k,
                                    const BoxDim& box,
                                    const Scalar qstarsq)
    { 
    // compute reciprocal lattice vectors
    Scalar3 a1 = box.getLatticeVector(0);
    Scalar3 a2 = box.getLatticeVector(1);
    Scalar3 a3 = box.getLatticeVector(2);

    Scalar V_box = box.getVolume();
    Scalar3 b1 = Scalar(2.0*M_PI)*make_scalar3(a2.y*a3.z-a2.z*a3.y, a2.z*a3.x-a2.x*a3.z, a2.x*a3.y-a2.y*a3.x)/V_box;
    Scalar3 b2 = Scalar(2.0*M_PI)*make_scalar3(a3.y*a1.z-a3.z*a1.y, a3.z*a1.x-a3.x*a1.z, a3.x*a1.y-a3.y*a1.x)/V_box;
    Scalar3 b3 = Scalar(2.0*M_PI)*make_scalar3(a1.y*a2.z-a1.z*a2.y, a1.z*a2.x-a1.x*a2.z, a1.x*a2.y-a1.y*a2.x)/V_box;

    unsigned int block_size = 4;
    
    dim3 blockDim(block_size,block_size,block_size);
    dim3 gridDim((mesh_idx.getW()/2+1)/block_size+1,
                 (mesh_idx.getH()/2+1)/block_size+1,
                 (mesh_idx.getD()/2+1)/block_size+1);
    gpu_compute_influence_function_kernel<<<gridDim,blockDim>>>(mesh_idx,
                                                                N,
                                                                d_inf_f,
                                                                d_k,
                                                                b1,
                                                                b2,
                                                                b3,
                                                                box,
                                                                qstarsq);
    } 
