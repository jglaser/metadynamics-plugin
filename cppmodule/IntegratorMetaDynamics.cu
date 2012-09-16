#include <hoomd/hoomd_config.h>
#include <hoomd/HOOMDMath.h>

extern __shared__ unsigned int coords[];

__global__ void gpu_update_grid_kernel(unsigned int num_elements,
                                       unsigned int *lengths,
                                       unsigned int dim,
                                       Scalar *current_val,
                                       Scalar *grid,
                                       Scalar *cv_min,
                                       Scalar *cv_max,
                                       Scalar *cv_sigma,
                                       Scalar scal,
                                       Scalar W)
    {
    unsigned int grid_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (grid_idx >= num_elements) return;

    // obtain d-dimensional coordinates
    unsigned int factor = 1;
    for (int j = 1; j < dim; j++)
        factor *= lengths[j-1];
 
    unsigned int rest = grid_idx;
    for (int i = dim-1; i >= 0; i--)
        {
        unsigned int c = rest/factor;
        coords[i+dim*threadIdx.x] = c;
        rest -= c*factor;
        if (i >0) factor /= lengths[i-1];
        }

    Scalar gauss_exp = Scalar(0.0);

    // evaluate Gaussian on grid point
    for (unsigned int cv_idx = 0; cv_idx < dim; ++cv_idx)
        {
        Scalar min = cv_min[cv_idx];
        Scalar max = cv_max[cv_idx];
        Scalar delta = (max - min)/(Scalar)(lengths[cv_idx] - 1);
        Scalar val = min + (Scalar)coords[cv_idx+dim*threadIdx.x]*delta;
        Scalar sigma = cv_sigma[cv_idx];
        Scalar d = val - current_val[cv_idx];
        gauss_exp -= d*d/Scalar(2.0)/sigma/sigma;
        }

    Scalar gauss = expf(gauss_exp);

    // add Gaussian to grid
    grid[grid_idx] += W*scal*gauss;
    }

cudaError_t gpu_update_grid(unsigned int num_elements,
                     unsigned int *d_lengths,
                     unsigned int dim,
                     Scalar *d_current_val,
                     Scalar *d_grid,
                     Scalar *d_cv_min,
                     Scalar *d_cv_max,
                     Scalar *d_cv_sigma,
                     Scalar scal,
                     Scalar W)
    {
    unsigned int block_size = 512;
    unsigned int smem_size = dim*sizeof(unsigned int)*block_size; 
    gpu_update_grid_kernel<<<num_elements/block_size+1, block_size, smem_size>>>(num_elements,
                                                                                 d_lengths,
                                                                                 dim,
                                                                                 d_current_val,
                                                                                 d_grid,
                                                                                 d_cv_min,
                                                                                 d_cv_max,
                                                                                 d_cv_sigma,
                                                                                 scal,
                                                                                 W);
    return cudaSuccess;
    }
