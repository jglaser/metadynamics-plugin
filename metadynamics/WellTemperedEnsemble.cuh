void gpu_scale_netforce(Scalar4 *d_net_force,
    Scalar4 *d_net_torque,
    Scalar *d_net_virial,
    unsigned int net_virial_pitch,
    Scalar fac,
    unsigned int N);

void gpu_reduce_potential_energy(Scalar *d_scratch,
    const Scalar4 *d_net_force,
    unsigned int N,
    Scalar *d_sum,
    unsigned int n_blocks,
    unsigned int block_size);
