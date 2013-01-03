#include "OrderParameterMeshGPU.cuh"

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

//! Assignment of particles to mesh using three-point scheme (triangular shaped cloud)
/*! This is a second order accurate scheme with continuous value and continuous derivative
 */
__global__ void gpu_assign_particles_kernel(const unsigned int N,
                                       const Scalar4 *d_postype,
                                       cufftComplex *d_mesh,
                                       const Index3D mesh_idx,
                                       const Scalar *d_mode,
                                       const unsigned int *d_cadj,
                                       const Index2D cadji,
                                       const BoxDim box)
    {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= N) return;

    Scalar V_cell = box.getVolume()/(Scalar)mesh_idx.getNumElements();
 
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
    Scalar3 reduced_pos = make_scalar3(f.x * (Scalar) mesh_idx.getW(),
                                       f.y * (Scalar) mesh_idx.getH(),
                                       f.z * (Scalar) mesh_idx.getD());

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
    unsigned int my_cell = mesh_idx(ix,iy,iz);

    // assign particle to cell and next neighbors
    for (unsigned int k = 0; k < cadji.getW(); ++k)
        {
        unsigned int neigh_cell = d_cadj[cadji(k, my_cell)];

        uint3 cell_coord = mesh_idx.getTriple(neigh_cell); 

        Scalar3 neigh_frac = make_scalar3((Scalar) cell_coord.x + Scalar(0.5),
                                         (Scalar) cell_coord.y + Scalar(0.5),
                                         (Scalar) cell_coord.z + Scalar(0.5));
       
        // coordinates of the neighboring cell between 0..1 
        Scalar3 neigh_frac_box = neigh_frac * dim_inv;
        Scalar3 neigh_pos = box.makeCoordinates(neigh_frac_box);

        // compute distance between particle and cell center in fractional coordinates using minimum image
        Scalar3 dx = box.minImage(neigh_pos - pos);
        Scalar3 dx_frac_box = box.makeFraction(dx) - make_scalar3(0.5,0.5,0.5);
        Scalar3 dx_frac = dx_frac_box*make_scalar3(mesh_idx.getW(), mesh_idx.getH(), mesh_idx.getD());

        // compute fraction of particle density assigned to cell
        Scalar density_fraction = assignTSC(dx_frac.x)*assignTSC(dx_frac.y)*assignTSC(dx_frac.z)/V_cell;
        unsigned int cell_idx = cell_coord.z + mesh_idx.getD() * cell_coord.y
                                + mesh_idx.getD() * mesh_idx.getH() * cell_coord.x;

        d_mesh[cell_idx].x += mode*density_fraction;
        }
         
    }

void gpu_assign_particles(const unsigned int N,
                          const Scalar4 *d_postype,
                          cufftComplex *d_mesh,
                          const Index3D& mesh_idx,
                          const Scalar *d_mode,
                          const unsigned int *d_cadj,
                          const Index2D& cadji,
                          const BoxDim& box)
    {

    unsigned int block_size = 512;

    gpu_assign_particles_kernel<<<N/block_size+1, block_size>>>(N,
                                                                d_postype,
                                                                d_mesh,
                                                                mesh_idx,
                                                                d_mode,
                                                                d_cadj,
                                                                cadji,
                                                                box);
    }

__global__ void gpu_update_meshes_kernel(const unsigned int n_wave_vectors,
                                         cufftComplex *d_fourier_mesh,
                                         cufftComplex *d_fourier_mesh_G,
                                         const Scalar *d_inf_f,
                                         const Scalar3 *d_k,
                                         const Scalar V_cell,
                                         cufftComplex *d_fourier_mesh_x,
                                         cufftComplex *d_fourier_mesh_y,
                                         cufftComplex *d_fourier_mesh_z)
    {
    unsigned int k = blockDim.x * blockIdx.x + threadIdx.x;

    if (k >= n_wave_vectors) return;

    d_fourier_mesh[k].x *= V_cell;
    d_fourier_mesh[k].y *= V_cell;

    cufftComplex fourier_G;
    fourier_G.x =d_fourier_mesh[k].x * d_inf_f[k];
    fourier_G.y =d_fourier_mesh[k].y * d_inf_f[k];

    Scalar3 kval = d_k[k];
    d_fourier_mesh_x[k].x = -fourier_G.y*kval.x;
    d_fourier_mesh_x[k].y = fourier_G.x*kval.x;

    d_fourier_mesh_y[k].x = -fourier_G.y*kval.y;
    d_fourier_mesh_y[k].y = fourier_G.x*kval.y;

    d_fourier_mesh_z[k].x = -fourier_G.y*kval.z;
    d_fourier_mesh_z[k].y = fourier_G.x*kval.z;

    d_fourier_mesh_G[k] = fourier_G;
    }

void gpu_update_meshes(const unsigned int n_wave_vectors,
                         cufftComplex *d_fourier_mesh,
                         cufftComplex *d_fourier_mesh_G,
                         const Scalar *d_inf_f,
                         const Scalar3 *d_k,
                         const Scalar V_cell,
                         cufftComplex *d_fourier_mesh_x,
                         cufftComplex *d_fourier_mesh_y,
                         cufftComplex *d_fourier_mesh_z)
    {
    const unsigned int block_size = 512;

    gpu_update_meshes_kernel<<<n_wave_vectors/block_size+1, block_size>>>(n_wave_vectors,
                                                                          d_fourier_mesh,
                                                                          d_fourier_mesh_G,
                                                                          d_inf_f,
                                                                          d_k,
                                                                          V_cell,
                                                                          d_fourier_mesh_x,
                                                                          d_fourier_mesh_y,
                                                                          d_fourier_mesh_z);
    }

__global__ void gpu_interpolate_forces_kernel(const unsigned int N,
                                       const Scalar4 *d_postype,
                                       Scalar4 *d_force,
                                       const Scalar bias,
                                       const cufftComplex *d_force_mesh_x,
                                       const cufftComplex *d_force_mesh_y,
                                       const cufftComplex *d_force_mesh_z,
                                       const Index3D mesh_idx,
                                       const Scalar *d_mode,
                                       const unsigned int *d_cadj,
                                       const Index2D cadji,
                                       const BoxDim box)
    {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= N) return;

    Scalar V = box.getVolume();
 
    // inverse dimensions
    Scalar3 dim_inv = make_scalar3(Scalar(1.0)/(Scalar)mesh_idx.getW(),
                                   Scalar(1.0)/(Scalar)mesh_idx.getH(),
                                   Scalar(1.0)/(Scalar)mesh_idx.getD());

    Scalar4 postype = d_postype[idx];

    Scalar3 pos = make_scalar3(postype.x, postype.y, postype.z);
    unsigned int type = __float_as_int(postype.w);

    // compute coordinates in units of the mesh size
    Scalar3 f = box.makeFraction(pos);
    Scalar3 reduced_pos = make_scalar3(f.x * (Scalar) mesh_idx.getW(),
                                       f.y * (Scalar) mesh_idx.getH(),
                                       f.z * (Scalar) mesh_idx.getD());

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
    unsigned int my_cell = mesh_idx(ix,iy,iz);

    Scalar3 force = make_scalar3(0.0,0.0,0.0);

    // assign particle to cell and next neighbors
    for (unsigned int k = 0; k < cadji.getW(); ++k)
        {
        unsigned int neigh_cell = d_cadj[cadji(k, my_cell)];

        uint3 cell_coord = mesh_idx.getTriple(neigh_cell); 

        Scalar3 neigh_frac = make_scalar3((Scalar) cell_coord.x + Scalar(0.5),
                                         (Scalar) cell_coord.y + Scalar(0.5),
                                         (Scalar) cell_coord.z + Scalar(0.5));
       
        // coordinates of the neighboring cell between 0..1 
        Scalar3 neigh_frac_box = neigh_frac * dim_inv;
        Scalar3 neigh_pos = box.makeCoordinates(neigh_frac_box);

        // compute distance between particle and cell center in fractional coordinates using minimum image
        Scalar3 dx = box.minImage(neigh_pos - pos);
        Scalar3 dx_frac_box = box.makeFraction(dx) - make_scalar3(0.5,0.5,0.5);
        Scalar3 dx_frac = dx_frac_box*make_scalar3(mesh_idx.getW(), mesh_idx.getH(), mesh_idx.getD());

        unsigned int cell_idx = cell_coord.z + mesh_idx.getD() * cell_coord.y
                                + mesh_idx.getD() * mesh_idx.getH() * cell_coord.x;

        force += -assignTSC(dx_frac.x)*assignTSC(dx_frac.y)*assignTSC(dx_frac.z)*d_mode[type]
                            *make_scalar3(d_force_mesh_x[cell_idx].x,
                                          d_force_mesh_y[cell_idx].x,
                                          d_force_mesh_z[cell_idx].x)/V;
        }  

    // Multiply with bias potential derivative
    force *= bias;

    d_force[idx] = make_scalar4(force.x,force.y,force.z,0.0);
    }

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
                             const BoxDim& box)
    {
    const unsigned int block_size = 512;

    gpu_interpolate_forces_kernel<<<N/block_size+1,block_size>>>(N,
                                                                 d_postype,
                                                                 d_force,
                                                                 bias,
                                                                 d_force_mesh_x,
                                                                 d_force_mesh_y,
                                                                 d_force_mesh_z,
                                                                 mesh_idx,
                                                                 d_mode,
                                                                 d_cadj,
                                                                 cadji,
                                                                 box);
    }

__global__ void kernel_calculate_cv_partial(
            int n_wave_vectors,
            Scalar *sum_partial,
            cufftComplex *d_fourier_mesh,
            cufftComplex *d_fourier_mesh_G)
    {
    extern __shared__ Scalar sdata[];

    unsigned int tidx = threadIdx.x;

    int j = blockIdx.x * blockDim.x + threadIdx.x;

    Scalar mySum = Scalar(0.0);

    if (j < n_wave_vectors) {
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
                           cufftComplex *d_fourier_mesh,
                           cufftComplex *d_fourier_mesh_G,
                           const unsigned int block_size)
    {
    unsigned int n_blocks = n_wave_vectors/block_size + 1;

    unsigned int shared_size = block_size * sizeof(Scalar);
    kernel_calculate_cv_partial<<<n_blocks, block_size, shared_size>>>(
               n_wave_vectors,
               d_sum_partial,
               d_fourier_mesh,
               d_fourier_mesh_G);

    // calculate final S(q) values 
    const unsigned int final_block_size = 512;
    shared_size = final_block_size*sizeof(Scalar);
    kernel_final_reduce_cv<<<1, final_block_size,shared_size>>>(d_sum_partial,
                                                                  n_blocks,
                                                                  d_sum);
    }
 
