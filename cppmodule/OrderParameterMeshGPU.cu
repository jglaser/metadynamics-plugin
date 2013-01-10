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
        // store distance to cell center in bin
        Scalar3 dim_inv = make_scalar3(Scalar(1.0)/(Scalar)dim.x,
                                       Scalar(1.0)/(Scalar)dim.y,
                                       Scalar(1.0)/(Scalar)dim.z);

        Scalar3 c = box.makeCoordinates(make_scalar3((Scalar)cell_coord.x+Scalar(0.5),
                                                 (Scalar)cell_coord.y+Scalar(0.5),
                                                 (Scalar)cell_coord.z+Scalar(0.5))*dim_inv);
        Scalar3 shift = box.minImage(pos-c);
        Scalar3 shift_frac = box.makeFraction(shift) - make_scalar3(0.5,0.5,0.5);
        shift_frac *= make_scalar3(dim.x,dim.y,dim.z);

        d_particle_bins[cell_idx*maxn+n] = make_scalar4(shift_frac.x,shift_frac.y,shift_frac.z, mode);
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
    extern __shared__ cufftReal assign_sdata[];

    unsigned int cell_idx = blockDim.y * blockIdx.y + threadIdx.y;

    int i = cell_idx/(ny*nz); 
    int j = (cell_idx - i*ny*nz)/nz;
    int k = cell_idx - i *ny*nz - j *nz;

    const unsigned int stride = blockDim.x;

    if (i>= (int)nx || j >= (int)ny || k >= (int)nz) return;

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

                for (unsigned int neigh_idx = threadIdx.x; neigh_idx < n_cell; neigh_idx+=stride)
                    {
                    Scalar4 xyzm = tex1Dfetch(particle_bins_tex, maxn*neigh_cell_idx+neigh_idx);
                    Scalar3 shift_frac = make_scalar3(xyzm.x, xyzm.y, xyzm.z);

                    Scalar3 dx_frac = shift_frac + make_scalar3(l,m,n);

                    // compute fraction of particle density assigned to cell
                    Scalar mode = xyzm.w;
                    grid_val += mode*assignTSC(dx_frac.x)*assignTSC(dx_frac.y)*assignTSC(dx_frac.z)/V_cell;
                    }

                } // end of loop over neighboring bins
  
    // write out mesh value to shared memory
    assign_sdata[threadIdx.y*blockDim.x+threadIdx.x] = grid_val;

    // no syncthreads necessary, we are within one warp
    if (threadIdx.x == 0)
        {
        cufftReal sum(0.0);
        for (unsigned int t = 0; t < blockDim.x; ++t)
            sum += assign_sdata[threadIdx.y*blockDim.x+t];

        d_mesh[cell_idx] = sum;
        }
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

    unsigned int block_size_x = 2;
    unsigned int block_size_y = 256;
    dim3 blockDim(block_size_x,block_size_y);
    dim3 gridDim(1,mesh_idx.getNumElements()/block_size_y+1,1);
    size_t sdata_size = sizeof(cufftReal)*block_size_y*block_size_x;
    gpu_assign_binned_particles_to_mesh_kernel<<<gridDim,blockDim,sdata_size>>>(mesh_idx.getW(),
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
