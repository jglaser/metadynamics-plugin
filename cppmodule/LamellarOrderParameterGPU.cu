/*! \file LamellarOrderParameterGPU.cu
    \brief CUDA implementation of LamellarOrderParameter GPU routines
 */
#include <cuda.h>

#include "LamellarOrderParameterGPU.cuh"

__global__ void kernel_calculate_sq_partial(
            int n_particles,
            Scalar2 *fourier_mode_partial,
            Scalar4 *postype,
            int n_wave,
            const int3 *lattice_vectors,
            Scalar *d_modes,
            const Scalar3 L)
    {
    extern __shared__ Scalar2 sdata[];

    unsigned int tidx = threadIdx.x;

    int j = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned int i = blockIdx.y;

    Scalar3 q = make_scalar3(lattice_vectors[i].x, lattice_vectors[i].y, lattice_vectors[i].z);
    q = Scalar(2.0*M_PI)*make_scalar3(q.x/L.x,q.y/L.y,q.z/L.z);

    Scalar2 mySum = make_scalar2(0.0,0.0);

    if (j < n_particles) {
        Scalar3 p = make_scalar3(postype[j].x, postype[j].y, postype[j].z);
        Scalar dotproduct = dot(q,p);
        unsigned int type = __float_as_int(postype[j].w);
        Scalar mode = d_modes[type];
        mySum.x = mode*fast::cos(dotproduct);
        mySum.y = mode*fast::sin(dotproduct);
        }
    sdata[tidx] = mySum;

   __syncthreads();

    // reduce the sum
    int offs = blockDim.x >> 1;
    while (offs > 0)
        {
        if (tidx < offs)
            {
            sdata[tidx].x += sdata[tidx + offs].x;
            sdata[tidx].y += sdata[tidx + offs].y;
            }
        offs >>= 1;
        __syncthreads();
        }

    // write result to global memeory
    if (tidx == 0)
       fourier_mode_partial[blockIdx.x + gridDim.x*i] = sdata[0];
    } 

__global__ void kernel_final_reduce_fourier_modes(Scalar2* fourier_mode_partial,
                                       unsigned int nblocks,
                                       Scalar2 *fourier_modes,
                                       unsigned int n_wave)
    {
    extern __shared__ Scalar2 smem[];

    unsigned int j = blockIdx.x;

    if (threadIdx.x == 0)
       fourier_modes[j] =make_scalar2(0.0,0.0);

    for (int start = 0; start< nblocks; start += blockDim.x)
        {
        __syncthreads();
        if (start + threadIdx.x < nblocks)
            smem[threadIdx.x] = fourier_mode_partial[j*nblocks+start + threadIdx.x];
        else
            smem[threadIdx.x] = make_scalar2(0.0,0.0);

        __syncthreads();

        // reduce the sum
        int offs = blockDim.x >> 1;
        while (offs > 0)
            {
            if (threadIdx.x < offs)
                {
                smem[threadIdx.x].x += smem[threadIdx.x + offs].x;
                smem[threadIdx.x].y += smem[threadIdx.x + offs].y;
                }
            offs >>= 1;
            __syncthreads();
            }

         if (threadIdx.x == 0)
            {
            fourier_modes[j].x += smem[0].x;
            fourier_modes[j].y += smem[0].y;
            }
        }
    }

cudaError_t gpu_calculate_fourier_modes(unsigned int n_wave,
                                 const int3 *d_lattice_vectors,
                                 unsigned int n_particles,
                                 Scalar4 *d_postype,
                                 Scalar *d_mode,
                                 Scalar2 *d_fourier_modes,
                                 unsigned int block_size,
                                 Scalar2 *d_fourier_mode_partial,
                                 const BoxDim& global_box
                                 )
    {
    unsigned int n_blocks = n_particles/block_size + 1;
    dim3 grid_dim(n_blocks, n_wave,1);
    dim3 block_dim(block_size);

    unsigned int shared_size = block_size * sizeof(Scalar2);
    kernel_calculate_sq_partial<<<grid_dim, block_dim, shared_size,0>>>(
               n_particles,
               d_fourier_mode_partial,
               d_postype,
               n_wave,
               d_lattice_vectors,
               d_mode,
               global_box.getL());

    // calculate final S(q) values
    const unsigned int final_block_size = 512;
    shared_size = final_block_size*sizeof(Scalar2);
    kernel_final_reduce_fourier_modes<<<n_wave, final_block_size,shared_size,0>>>(d_fourier_mode_partial,
                                                                  n_blocks,
                                                                  d_fourier_modes,
                                                                  n_wave);
    return cudaSuccess;
    }

__global__ void kernel_compute_sq_forces(unsigned int N,
                                  Scalar4 *postype,
                                  Scalar4 *force,
                                  unsigned int n_wave,
                                  const int3 *lattice_vectors,
                                  Scalar *mode,
                                  unsigned int n_global,
                                  Scalar bias,
                                  Scalar cv,
                                  const Scalar3 L)
    {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N)
        return;

    Scalar4 p = postype[idx];
    Scalar3 pos = make_scalar3(p.x, p.y, p.z);
    unsigned int type = __float_as_int(p.w);

    Scalar m = mode[type];

    Scalar4 force_energy = make_scalar4(0.0f,0.0f,0.0f,0.0f);

    Scalar denom = (Scalar)n_global;
    for (unsigned int k = 0; k < n_wave; k++)
        {
        Scalar3 q = make_scalar3(lattice_vectors[k].x, lattice_vectors[k].y, lattice_vectors[k].z);
        q = Scalar(2.0*M_PI)*make_scalar3(q.x/L.x,q.y/L.y,q.z/L.z);
        Scalar dotproduct = dot(pos,q);

        Scalar f = Scalar(2.0)*m*fast::sin(dotproduct);

        force_energy.x += q.x*f;
        force_energy.y += q.y*f;
        force_energy.z += q.z*f;
        }

    force_energy.x /= denom;
    force_energy.y /= denom;
    force_energy.z /= denom;

    force_energy.x *= bias;
    force_energy.y *= bias;
    force_energy.z *= bias;

    force[idx] = force_energy;
    }

cudaError_t gpu_compute_sq_forces(unsigned int N,
                                  Scalar4 *d_postype,
                                  Scalar4 *d_force,
                                  unsigned int n_wave,
                                  const int3 *d_lattice_vectors,
                                  Scalar *d_mode,
                                  unsigned int n_global,
                                  Scalar bias,
                                  Scalar cv_val,
                                  const BoxDim& global_box)
    {
    cudaError_t cudaStatus;
    const unsigned int block_size = 512;

    kernel_compute_sq_forces<<<N/block_size + 1, block_size,0,0>>>(N,
                                                               d_postype,
                                                               d_force,
                                                               n_wave,
                                                               d_lattice_vectors,
                                                               d_mode,
                                                               n_global,
                                                               bias,
                                                               cv_val,
                                                               global_box.getL());

    cudaStatus = cudaGetLastError();
    return cudaStatus;
    }
