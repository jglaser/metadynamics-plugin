void gpu_scale_netforce(Scalar4 *d_net_force,
    Scalar fac,
    unsigned int N);

void gpu_reduce_potential_energy(Scalar *d_scratch,
    const Scalar4 *d_net_force,
    unsigned int N,
    Scalar *d_sum,
    unsigned int n_blocks,
    unsigned int block_size);
