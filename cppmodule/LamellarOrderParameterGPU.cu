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
            Scalar3 *wave_vectors,
            Scalar *d_modes,
            Scalar *phases)
    {
    extern __shared__ Scalar2 sdata[];

    unsigned int tidx = threadIdx.x;

    int j = blockIdx.x * blockDim.x + threadIdx.x;

    for (unsigned int i = 0; i < n_wave; i++) {
        Scalar3 q = wave_vectors[i];
        Scalar phi = phases[i];

        Scalar2 mySum = make_scalar2(0.0f,0.0f);

        if (j < n_particles) {

            Scalar3 p = make_scalar3(postype[j].x, postype[j].y, postype[j].z);
            Scalar dotproduct = q.x * p.x + q.y * p.y + q.z * p.z;
            unsigned int type = __float_as_int(postype[j].w);
            Scalar mode = d_modes[type];
            Scalar2 exponential = make_scalar2(mode*cosf(dotproduct+phi),
                                                   mode*sinf(dotproduct+phi));
            mySum.x += exponential.x;
            mySum.y += exponential.y;
        }
        sdata[tidx] = mySum;

       __syncthreads();
        // reduce in shared memory
        if (blockDim.x >= 512)
            {
            if (tidx < 256)
                {
                mySum.x += sdata[tidx+256].x;
                mySum.y += sdata[tidx+256].y;
                sdata[tidx] = mySum;
                }
            __syncthreads();
            }

        if (blockDim.x >= 256) {
            if (tidx < 128)
                {
                mySum.x += sdata[tidx+128].x;
                mySum.y += sdata[tidx+128].y;
                sdata[tidx] = mySum;
                }
            __syncthreads();
            }

        if (blockDim.x >= 128)
            {
            if (tidx < 64)
                {
                mySum.x += sdata[tidx+64].x;
                mySum.y += sdata[tidx+64].y;
                sdata[tidx] = mySum;
                }
           __syncthreads();
            }

        if (tidx < 32) {
            volatile Scalar2* smem = sdata;
            if (blockDim.x >= 64)
                {
                Scalar2 rhs;
                rhs.x = mySum.x + smem[tidx + 32].x;
                rhs.y = mySum.y + smem[tidx + 32].y;
                smem[tidx].x = rhs.x;
                smem[tidx].y = rhs.y;
                mySum = rhs;
                }
            if (blockDim.x >= 32)
                {
                Scalar2 rhs;
                rhs.x = mySum.x + smem[tidx + 16].x;
                rhs.y = mySum.y + smem[tidx + 16].y;
                smem[tidx].x = rhs.x;
                smem[tidx].y = rhs.y;
                mySum = rhs;
                }
            if (blockDim.x >= 16)
                {
                Scalar2 rhs;
                rhs.x = mySum.x + smem[tidx + 8].x;
                rhs.y = mySum.y + smem[tidx + 8].y;
                smem[tidx].x = rhs.x;
                smem[tidx].y = rhs.y;
                mySum = rhs;
                }
            if (blockDim.x >=  8)
                {
                Scalar2 rhs;
                rhs.x = mySum.x + smem[tidx + 4].x;
                rhs.y = mySum.y + smem[tidx + 4].y;
                smem[tidx].x = rhs.x;
                smem[tidx].y = rhs.y;
                mySum = rhs;
                }
            if (blockDim.x >=  4)
                {
                Scalar2 rhs;
                rhs.x = mySum.x + smem[tidx + 2].x;
                rhs.y = mySum.y + smem[tidx + 2].y;
                smem[tidx].x = rhs.x;
                smem[tidx].y = rhs.y;
                mySum = rhs;
                }
            if (blockDim.x >=  2)
                { 
                Scalar2 rhs;
                rhs.x = mySum.x + smem[tidx + 1].x;
                rhs.y = mySum.y + smem[tidx + 1].y;
                smem[tidx].x = rhs.x;
                smem[tidx].y = rhs.y;
                mySum = rhs;
                }
            }

        // write result to global memeory
        if (tidx == 0)
           fourier_mode_partial[blockIdx.x + gridDim.x*i] = sdata[0];
        } // end loop over wave vectors
    } 

__global__ void kernel_final_reduce_fourier_modes(Scalar2* fourier_mode_partial,
                                       unsigned int nblocks,
                                       Scalar2 *fourier_modes,
                                       unsigned int n_wave)
    {
    extern __shared__ volatile Scalar2 smem[];

    if (threadIdx.x == 0)
        {
        for (unsigned int j = 0; j < n_wave; j++)
            fourier_modes[j] = make_scalar2(0.0,0.0);
        }

    for (int start = 0; start< nblocks; start += blockDim.x)
        {
        __syncthreads();
        if (start + threadIdx.x < nblocks)
            {
            for (unsigned int j = 0; j < n_wave; ++j)
                {
                smem[j*blockDim.x + threadIdx.x].x = fourier_mode_partial[j*nblocks+start + threadIdx.x].x;
                smem[j*blockDim.x + threadIdx.x].y = fourier_mode_partial[j*nblocks+start + threadIdx.x].y;
                }
            }
        else
            {
            for (unsigned int j = 0; j < n_wave; ++j)
                {
                smem[j*blockDim.x + threadIdx.x].x = Scalar(0.0);
                smem[j*blockDim.x + threadIdx.x].y = Scalar(0.0);
                }
            }

        __syncthreads();

        // reduce the sum
        int offs = blockDim.x >> 1;
        while (offs > 0)
            {
            if (threadIdx.x < offs)
                {
                for (unsigned int j = 0; j < n_wave; ++j)
                    {
                    smem[j*blockDim.x + threadIdx.x].x += smem[j*blockDim.x + threadIdx.x + offs].x;
                    smem[j*blockDim.x + threadIdx.x].y += smem[j*blockDim.x + threadIdx.x + offs].y;
                    }
                }
            offs >>= 1;
            __syncthreads();
            }

         if (threadIdx.x == 0)
            {
            for (unsigned int j = 0; j < n_wave; ++j)
                {
                fourier_modes[j].x += smem[j*blockDim.x].x;
                fourier_modes[j].y += smem[j*blockDim.x].y;
                }
            }
        }
    }

cudaError_t gpu_calculate_fourier_modes(unsigned int n_wave,
                                 Scalar3 *d_wave_vectors,
                                 unsigned int n_particles,
                                 Scalar4 *d_postype,
                                 Scalar *d_mode,
                                 Scalar2 *d_fourier_modes,
                                 Scalar *d_phases,
                                 unsigned int block_size,
                                 Scalar2 *d_fourier_mode_partial
                                 )
    {
    cudaError_t cudaStatus;

    unsigned int n_blocks = n_particles/block_size + 1;

    unsigned int shared_size = block_size * sizeof(Scalar2);
    kernel_calculate_sq_partial<<<n_blocks, block_size, shared_size>>>(
               n_particles,
               d_fourier_mode_partial,
               d_postype,
               n_wave,
               d_wave_vectors,
               d_mode,
               d_phases);

    if (cudaStatus = cudaGetLastError()) 
           return cudaStatus;

    // calculate final S(q) values 
    const unsigned int final_block_size = 512;
    shared_size = final_block_size*n_wave*sizeof(Scalar2);
    kernel_final_reduce_fourier_modes<<<1, final_block_size,shared_size>>>(d_fourier_mode_partial,
                                                                  n_blocks,
                                                                  d_fourier_modes,
                                                                  n_wave);
                                                                  

    if (cudaStatus = cudaGetLastError())
        return cudaStatus;

    return cudaSuccess;
    }

__global__ void kernel_compute_sq_forces(unsigned int N,
                                  Scalar4 *postype,
                                  Scalar4 *force,
                                  unsigned int n_wave,
                                  Scalar3 *wave_vectors,
                                  Scalar *mode,
                                  Scalar n_global,
                                  Scalar bias,
                                  Scalar *phases)
    {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N)
        return;

    Scalar4 p = postype[idx];
    Scalar3 pos = make_scalar3(p.x, p.y, p.z);
    unsigned int type = __float_as_int(p.w);

    Scalar m = mode[type];

    Scalar4 force_energy = make_scalar4(0.0f,0.0f,0.0f,0.0f);

    for (unsigned int k = 0; k < n_wave; k++)
        {
        Scalar3 q = wave_vectors[k];
        Scalar dotproduct = dot(pos,q);

        Scalar f = m*sinf(dotproduct + phases[k]);
        
        force_energy.x += q.x*f;
        force_energy.y += q.y*f;
        force_energy.z += q.z*f;
        }

    force_energy.x /= n_global;
    force_energy.y /= n_global;
    force_energy.z /= n_global;

    force_energy.x *= bias;
    force_energy.y *= bias;
    force_energy.z *= bias;

    force[idx] = force_energy;
    }

cudaError_t gpu_compute_sq_forces(unsigned int N,
                                  Scalar4 *d_postype,
                                  Scalar4 *d_force,
                                  unsigned int n_wave,
                                  Scalar3 *d_wave_vectors,
                                  Scalar *d_mode,
                                  unsigned int n_global,
                                  Scalar bias,
                                  Scalar *d_phases)
    {
    cudaError_t cudaStatus;
    const unsigned int block_size = 512;

    kernel_compute_sq_forces<<<N/block_size + 1, block_size>>>(N,
                                                               d_postype,
                                                               d_force,
                                                               n_wave,
                                                               d_wave_vectors,
                                                               d_mode,
                                                               n_global,
                                                               bias,
                                                               d_phases);

    cudaStatus = cudaGetLastError();
    return cudaStatus;
    }
