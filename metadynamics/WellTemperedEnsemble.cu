#include <hoomd/ParticleData.cuh>

__global__ void gpu_scale_netforce_kernel(Scalar4 *d_net_force,
    Scalar4 *d_net_torque,
    Scalar *d_net_virial,
    unsigned int net_virial_pitch,
    Scalar fac,
    unsigned int N)
    {
    unsigned int idx = blockIdx.x*blockDim.x+threadIdx.x;

    if (idx>=N) return;

    Scalar4 net_force = d_net_force[idx];
    net_force.x *= fac;
    net_force.y *= fac;
    net_force.z *= fac;
    net_force.w *= fac;
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
    unsigned int N)
    {
    unsigned int block_size = 256;
    gpu_scale_netforce_kernel<<<(N/block_size+1),block_size>>>(d_net_force, d_net_torque, d_net_virial, net_virial_pitch, fac, N);
    }

//! Shared memory used in reducing the sums
extern __shared__ Scalar compute_pe_sdata[];

///! Shared memory used in final reduction
extern __shared__ Scalar compute_pe_final_sdata[];

__global__ void gpu_compute_potential_energy_partial(Scalar *d_scratch,
                                                const Scalar4 *d_net_force,
                                                unsigned int N)
    {
    // determine which particle this thread works on
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    Scalar my_element; // element of scratch space read in
    if (idx < N)
        {
        // update positions to the next timestep and update velocities to the next half step
        Scalar4 net_force = d_net_force[idx];

        // compute our contribution to the sum
        my_element = net_force.w;
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
        *d_sum = final_sum;
        }
    }

void gpu_reduce_potential_energy(Scalar *d_scratch,
    const Scalar4 *d_net_force,
    unsigned int N,
    Scalar *d_sum,
    unsigned int n_blocks,
    unsigned int block_size)
    {
    dim3 grid(n_blocks, 1, 1);
    dim3 threads(block_size, 1, 1);
    unsigned int shared_bytes = sizeof(Scalar)*block_size;

    gpu_compute_potential_energy_partial<<<grid, threads, shared_bytes>>>(d_scratch, d_net_force, N);

    int final_block_size = 512;
    grid = dim3(1, 1, 1);
    threads = dim3(final_block_size, 1, 1);

    shared_bytes = sizeof(Scalar)*final_block_size;

    // run the kernel
    gpu_compute_potential_energy_final_sums<<<grid, threads, shared_bytes>>>(d_scratch,
                                                                   n_blocks,
                                                                   d_sum);
    }
