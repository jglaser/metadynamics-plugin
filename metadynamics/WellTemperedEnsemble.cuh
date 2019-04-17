#include "hoomd/GPUPartition.cuh"

void gpu_scale_netforce(Scalar4 *d_net_force,
    Scalar4 *d_net_torque,
    Scalar *d_net_virial,
    unsigned int net_virial_pitch,
    Scalar fac,
    const GPUPartition& gpu_partition,
    const unsigned int nghost,
    const unsigned int block_size);

void gpu_reduce_potential_energy(Scalar *d_scratch,
    Scalar4 *d_net_force,
    const GPUPartition& gpu_partition,
    const unsigned int nghost,
    const unsigned int scratch_size,
    Scalar *d_sum,
    bool zero_energy,
    const unsigned int block_size);
