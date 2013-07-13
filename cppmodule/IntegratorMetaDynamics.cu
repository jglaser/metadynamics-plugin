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
                                       Scalar *cv_sigma_inv,
                                       Scalar scal,
                                       Scalar W,
                                       Scalar T)
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
    for (unsigned int cv_i = 0; cv_i < dim; ++cv_i)
        {
        Scalar min_i = cv_min[cv_i];
        Scalar max_i = cv_max[cv_i];
        Scalar delta_i = (max_i - min_i)/(Scalar)(lengths[cv_i] - 1);
        Scalar val_i = min_i + (Scalar)coords[cv_i+dim*threadIdx.x]*delta_i;
        Scalar d_i = val_i - current_val[cv_i];

        for (unsigned int cv_j = 0; cv_j < dim; ++cv_j)
            {
            Scalar min_j = cv_min[cv_j];
            Scalar max_j = cv_max[cv_j];
            Scalar delta_j = (max_j - min_j)/(Scalar)(lengths[cv_j] - 1);
            Scalar val_j = min_j + (Scalar)coords[cv_j+dim*threadIdx.x]*delta_j;
            Scalar d_j = val_j - current_val[cv_j];

            Scalar sigma_inv_ij = cv_sigma_inv[cv_i*dim+cv_j];

            gauss_exp -= d_i*d_j*Scalar(1.0/2.0)*(sigma_inv_ij*sigma_inv_ij);
            }
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
                     Scalar *d_cv_sigma_inv,
                     Scalar scal,
                     Scalar W,
                     Scalar T)
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
                                                                                 d_cv_sigma_inv,
                                                                                 scal,
                                                                                 W,
                                                                                 T);
    return cudaSuccess;
    }

__global__ void gpu_update_histograms_kernel(Scalar val,
                                             Scalar cv_min,
                                             Scalar delta,
                                             unsigned int num_points,
                                             Scalar sigma,
                                             bool state,
                                             Scalar *histogram,
                                             Scalar *histogram_plus)

    {

    unsigned int grid_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (grid_idx >= num_points) return;

    Scalar norm = Scalar(1.0)/sqrtf(Scalar(2.0*M_PI)*sigma*sigma);
    Scalar val_grid = cv_min + grid_idx*delta;
    Scalar d = val - val_grid;
    Scalar gauss_exp = d*d/Scalar(2.0)/sigma/sigma;
    Scalar gauss = norm*expf(-gauss_exp);

    // add Gaussian to grid
    histogram[grid_idx] += gauss;

    if (state == true)
        histogram_plus[grid_idx] += gauss;
    }

cudaError_t gpu_update_histograms(Scalar val,
                                  Scalar cv_min,
                                  Scalar delta,
                                  unsigned int num_points,
                                  Scalar sigma,
                                  bool state,
                                  Scalar *d_histogram,
                                  Scalar *d_histogram_plus)

    {
    unsigned int block_size = 512;
    gpu_update_histograms_kernel<<<num_points/block_size+1, block_size>>>(val,
                                                                            cv_min,
                                                                            delta,
                                                                            num_points,
                                                                            sigma,
                                                                            state,
                                                                            d_histogram,
                                                                            d_histogram_plus);
    return cudaSuccess;
    }


