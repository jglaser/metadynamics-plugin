#include <cuda.h>
#include <cuComplex.h>

#include "LamellarOrderParameterGPU.cuh"

__global__ void kernel_calculate_sq_partial(
            int n_particles,
            cuComplex *fourier_mode_partial,
            Scalar4 *postype,
            int n_wave,
            Scalar3 *wave_vectors,
            Scalar *d_modes)
    {
    extern __shared__ cuComplex sdata[];

    unsigned int tidx = threadIdx.x;

    int j = blockIdx.x * blockDim.x + threadIdx.x;

    for (unsigned int i = 0; i < n_wave; i++) {
        Scalar3 q = wave_vectors[i];

        cuComplex mySum = make_cuComplex(0.0f,0.0f);

        if (j < n_particles) {

            Scalar3 p = make_scalar3(postype[j].x, postype[j].y, postype[j].z);
            Scalar dotproduct = q.x * p.x + q.y * p.y + q.z * p.z;
            unsigned int type = __float_as_int(postype[j].w);
            Scalar mode = d_modes[type];
            cuComplex exponential = make_cuComplex(mode*cosf(dotproduct),
                                                   mode*sinf(dotproduct));
            mySum = cuCaddf(mySum,exponential);
        }
        sdata[tidx] = mySum;

       __syncthreads();
        // reduce in shared memory
        if (blockDim.x >= 512) {
           if (tidx < 256) {sdata[tidx] = mySum = cuCaddf(mySum,sdata[tidx+256]); }
            __syncthreads();
        }

        if (blockDim.x >= 256) {
           if (tidx < 128) {sdata[tidx] = mySum = cuCaddf(mySum, sdata[tidx + 128]); }
           __syncthreads();
        }

        if (blockDim.x >= 128) {
           if (tidx <  64) {sdata[tidx] = mySum = cuCaddf(mySum, sdata[tidx +  64]); }
           __syncthreads();
        }

        if (tidx < 32) {
            volatile cuComplex* smem = sdata;
            if (blockDim.x >= 64) {
                cuComplex rhs = cuCaddf(mySum, smem[tidx + 32]);
                smem[tidx].x = rhs.x;
                smem[tidx].y = rhs.y;
                mySum = rhs;
            }
            if (blockDim.x >= 32) {
                cuComplex rhs = cuCaddf(mySum, smem[tidx + 16]);
                smem[tidx].x = rhs.x;
                smem[tidx].y = rhs.y;
                mySum = rhs;
            }
            if (blockDim.x >= 16) {
                cuComplex rhs = cuCaddf(mySum, smem[tidx + 8]);
                smem[tidx].x = rhs.x;
                smem[tidx].y = rhs.y;
                mySum = rhs;
            }
             if (blockDim.x >=  8) {
                cuComplex rhs = cuCaddf(mySum, smem[tidx + 4]);
                smem[tidx].x = rhs.x;
                smem[tidx].y = rhs.y;
                mySum = rhs;
            }
            if (blockDim.x >=  4) {
                cuComplex rhs = cuCaddf(mySum, smem[tidx + 2]);
                smem[tidx].x = rhs.x;
                smem[tidx].y = rhs.y;
                mySum = rhs;
            }
            if (blockDim.x >=  2) {
                cuComplex rhs = cuCaddf(mySum, smem[tidx + 1]);
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

__global__ void kernel_final_reduce_fourier_modes(cuComplex* fourier_mode_partial,
                                       unsigned int nblocks,
                                       Scalar2 *fourier_modes,
                                       unsigned int n_wave)
    {
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n_wave)
        return;

    // do final reduction of fourier mode
    cuComplex fourier_mode = make_cuComplex(0.0f,0.0f);
    for (unsigned int j = 0; j < nblocks; j++)
       fourier_mode = cuCaddf(fourier_mode, fourier_mode_partial[j + i*nblocks]);

    fourier_modes[i] = make_scalar2(fourier_mode.x, fourier_mode.y); 
    }

cudaError_t gpu_calculate_fourier_modes(unsigned int n_wave,
                                 Scalar3 *d_wave_vectors,
                                 unsigned int n_particles,
                                 Scalar4 *d_postype,
                                 Scalar *d_mode,
                                 Scalar2 *d_fourier_modes
                                 )
    {
    cuComplex *d_fourier_mode_partial;

    cudaError_t cudaStatus;

    const unsigned int block_size_x = 256;
    unsigned int n_blocks_x = n_particles/block_size_x + 1;

    cudaMalloc(&d_fourier_mode_partial, sizeof(cuComplex)*n_wave*n_blocks_x);

    unsigned int shared_size = block_size_x * sizeof(cuComplex);
    kernel_calculate_sq_partial<<<n_blocks_x, block_size_x, shared_size>>>(
               n_particles,
               d_fourier_mode_partial,
               d_postype,
               n_wave,
               d_wave_vectors,
               d_mode);

    if (cudaStatus = cudaGetLastError()) 
           return cudaStatus;

    // calculate final S(q) values 
    const unsigned int block_size = 512;
    kernel_final_reduce_fourier_modes<<<n_wave/block_size + 1, block_size>>>(d_fourier_mode_partial,
                                                                  n_blocks_x,
                                                                  d_fourier_modes,
                                                                  n_wave);
                                                                  

    if (cudaStatus = cudaGetLastError())
        return cudaStatus;

    cudaFree(d_fourier_mode_partial);

    return cudaSuccess;
    }

__global__ void kernel_compute_sq_forces(unsigned int N,
                                  Scalar4 *postype,
                                  Scalar4 *force,
                                  Scalar *virial,
                                  unsigned int n_wave,
                                  Scalar3 *wave_vectors,
                                  Scalar2 *fourier_modes,
                                  Scalar *mode,
                                  Scalar V,
                                  Scalar bias)
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

        Scalar2 exponential;
        exponential.x = m*cosf(dotproduct);
        exponential.y = -m*sinf(dotproduct);
        
        Scalar2 fourier_mode = fourier_modes[k];
        Scalar im = - (exponential.x*fourier_mode.y + exponential.y*fourier_mode.x);

        force_energy.x += Scalar(2.0)*q.x*im;
        force_energy.y += Scalar(2.0)*q.y*im;
        force_energy.z += Scalar(2.0)*q.z*im;
        }

    force_energy.x /= (Scalar)n_wave*V;
    force_energy.y /= (Scalar)n_wave*V;
    force_energy.z /= (Scalar)n_wave*V;

    force_energy.x *= bias;
    force_energy.y *= bias;
    force_energy.z *= bias;

    force[idx] = force_energy;
    }

cudaError_t gpu_compute_sq_forces(unsigned int N,
                                  Scalar4 *d_postype,
                                  Scalar4 *d_force,
                                  Scalar *d_virial,
                                  unsigned int n_wave,
                                  Scalar3 *d_wave_vectors,
                                  Scalar2 *d_fourier_modes,
                                  Scalar *d_mode,
                                  const BoxDim global_box,
                                  Scalar bias)
    {
    cudaError_t cudaStatus;
    const unsigned int block_size = 512;

    Scalar3 L = global_box.getL();
    Scalar V = L.x*L.y*L.z;

    kernel_compute_sq_forces<<<N/block_size + 1, block_size>>>(N,
                                                               d_postype,
                                                               d_force,
                                                               d_virial,
                                                               n_wave,
                                                               d_wave_vectors,
                                                               d_fourier_modes,
                                                               d_mode,
                                                               V,
                                                               bias);

    cudaStatus = cudaGetLastError();
    return cudaStatus;
    }
