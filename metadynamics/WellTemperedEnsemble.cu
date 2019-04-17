#include <hoomd/ParticleData.cuh>
#include "WellTemperedEnsemble.cuh"

// pre-pascal atomicAdd for doubles
__device__ double atomicAdd_double(double* address, double val) { 

    unsigned long long int* address_as_ull = (unsigned long long int*)address; 
    unsigned long long int old = *address_as_ull, assumed; 

    do { 
        assumed = old; 
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed))); 
        // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
        } 
    while (assumed != old); 
    return __longlong_as_double(old); 
    } 

__global__ void gpu_scale_netforce_kernel(Scalar4 *d_net_force,
    Scalar4 *d_net_torque,
    Scalar *d_net_virial,
    unsigned int net_virial_pitch,
    Scalar fac,
    const unsigned int nwork,
    const unsigned int offset)
    {
    unsigned int idx = blockIdx.x*blockDim.x+threadIdx.x;

    if (idx>=nwork) return;
    idx += offset;

    Scalar4 net_force = d_net_force[idx];
    net_force.x *= fac;
    net_force.y *= fac;
    net_force.z *= fac;
//    net_force.w *= fac;
    d_net_force[idx] = net_force;

    Scalar4 net_torque = d_net_torque[idx];
    net_torque.x *= fac;
    net_torque.y *= fac;
    net_torque.z *= fac;
    d_net_torque[idx] = net_torque;

    d_net_virial[0*net_virial_pitch+idx] *= fac;
    d_net_virial[1*net_virial_pitch+idx] *= fac;
    d_net_virial[2*net_virial_pitch+idx] *= fac;
    d_net_virial[3*net_virial_pitch+idx] *= fac;
    d_net_virial[4*net_virial_pitch+idx] *= fac;
    d_net_virial[5*net_virial_pitch+idx] *= fac;
    }

void gpu_scale_netforce(Scalar4 *d_net_force,
    Scalar4 *d_net_torque,
    Scalar *d_net_virial,
    unsigned int net_virial_pitch,
    Scalar fac,
    const GPUPartition& gpu_partition,
    const unsigned int nghost,
    const unsigned int block_size)
    {
    static unsigned int max_block_size = UINT_MAX;
    if (max_block_size == UINT_MAX)
        {
        cudaFuncAttributes attr;
        cudaFuncGetAttributes(&attr, (const void *)gpu_scale_netforce_kernel);
        max_block_size = attr.maxThreadsPerBlock;
        }

    unsigned int run_block_size = min(block_size, max_block_size);

    // iterate over active GPUs in reverse, to end up on first GPU when returning from this function
    for (int idev = gpu_partition.getNumActiveGPUs() - 1; idev >= 0; --idev)
        {
        auto range = gpu_partition.getRangeAndSetGPU(idev);

        unsigned int nwork = range.second - range.first;

        // process ghosts in final range
        if (idev == (int)gpu_partition.getNumActiveGPUs()-1)
            nwork += nghost;

        // setup the grid to run the kernel
        dim3 grid( (nwork/run_block_size) + 1, 1, 1);
        dim3 threads(run_block_size, 1, 1);

        gpu_scale_netforce_kernel<<<grid, threads>>>(d_net_force, d_net_torque, d_net_virial, net_virial_pitch, fac, nwork, range.first);
        }
    }

//! Shared memory used in reducing the sums
extern __shared__ Scalar compute_pe_sdata[];

///! Shared memory used in final reduction
extern __shared__ Scalar compute_pe_final_sdata[];

__global__ void gpu_compute_potential_energy_partial(Scalar *d_scratch,
                                                Scalar4 *d_net_force,
                                                const unsigned int nwork,
                                                const unsigned int offset,
                                                bool zero_energy)
    {
    // determine which particle this thread works on
    int work_idx = blockIdx.x * blockDim.x + threadIdx.x;

    Scalar my_element; // element of scratch space read in
    if (work_idx < nwork)
        {
        const unsigned int idx = work_idx + offset;
        Scalar4 net_force = d_net_force[idx];

        // compute our contribution to the sum
        my_element = net_force.w;

        if (zero_energy) d_net_force[idx].w = Scalar(0.0);
        }
    else
        {
        // non-participating thread: contribute 0 to the sum
        my_element = Scalar(0.0);
        }

    compute_pe_sdata[threadIdx.x] = my_element;
    __syncthreads();

    // reduce the sum in parallel
    int offs = blockDim.x >> 1;
    while (offs > 0)
        {
        if (threadIdx.x < offs)
            {
            compute_pe_sdata[threadIdx.x] += compute_pe_sdata[threadIdx.x + offs];
            }
        offs >>= 1;
        __syncthreads();
        }

    // write out our partial sum
    if (threadIdx.x == 0)
        {
        Scalar res = compute_pe_sdata[0];
        d_scratch[blockIdx.x] = res;
        }
    }

__global__ void gpu_compute_potential_energy_final_sums(Scalar *d_scratch,
                                              unsigned int num_partial_sums,
                                              Scalar *d_sum
                                              )
    {
    Scalar final_sum(0.0);

    // sum up the values in the partial sum via a sliding window
    for (int start = 0; start < num_partial_sums; start += blockDim.x)
        {
        __syncthreads();
        if (start + threadIdx.x < num_partial_sums)
            {
            Scalar scratch = d_scratch[start + threadIdx.x];

            compute_pe_final_sdata[threadIdx.x] = scratch;
            }
        else
            compute_pe_final_sdata[threadIdx.x] = Scalar(0.0);
        __syncthreads();

        // reduce the sum in parallel
        int offs = blockDim.x >> 1;
        while (offs > 0)
            {
            if (threadIdx.x < offs)
                {
                compute_pe_final_sdata[threadIdx.x] += compute_pe_final_sdata[threadIdx.x + offs];
                }
            offs >>= 1;
            __syncthreads();
            }

        if (threadIdx.x == 0)
            {
            final_sum += compute_pe_final_sdata[0];
            }
        }

    if (threadIdx.x == 0)
        {
        #if (__CUDA_ARCH >= 600)
        atomicAdd_system(d_sum,final_sum);
        #else
        atomicAdd_double(d_sum,final_sum);
        #endif
        }
    }

void gpu_reduce_potential_energy(Scalar *d_scratch,
    Scalar4 *d_net_force,
    const GPUPartition& gpu_partition,
    const unsigned int nghost,
    const unsigned int scratch_size,
    Scalar *d_sum,
    bool zero_energy,
    const unsigned int block_size)
    {
    static unsigned int max_block_size = UINT_MAX;
    if (max_block_size == UINT_MAX)
        {
        cudaFuncAttributes attr;
        cudaFuncGetAttributes(&attr, (const void *)gpu_compute_potential_energy_partial);
        max_block_size = attr.maxThreadsPerBlock;
        }

    unsigned int run_block_size = min(block_size, max_block_size);

    // iterate over active GPUs in reverse, to end up on first GPU when returning from this function
    for (int idev = gpu_partition.getNumActiveGPUs() - 1; idev >= 0; --idev)
        {
        auto range = gpu_partition.getRangeAndSetGPU(idev);

        unsigned int nwork = range.second - range.first;

        // process ghosts in final range
        if (idev == (int)gpu_partition.getNumActiveGPUs()-1)
            nwork += nghost;

        // setup the grid to run the kernel
        unsigned int n_blocks = (nwork/run_block_size) + 1;
        dim3 grid(n_blocks,  1, 1);
        dim3 threads(run_block_size, 1, 1);

        unsigned int shared_bytes = sizeof(Scalar)*run_block_size;
        gpu_compute_potential_energy_partial<<<grid, threads, shared_bytes>>>(d_scratch+idev*scratch_size, d_net_force, nwork, range.first, zero_energy);

        int final_block_size = 512;
        grid = dim3(1, 1, 1);
        threads = dim3(final_block_size, 1, 1);
        shared_bytes = sizeof(Scalar)*final_block_size;

        // run the kernel
        gpu_compute_potential_energy_final_sums<<<grid, threads, shared_bytes>>>(d_scratch+idev*scratch_size,
                                                                       n_blocks,
                                                                       d_sum);
        }
    }
