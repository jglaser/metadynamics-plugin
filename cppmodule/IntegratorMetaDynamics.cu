#include <hoomd/hoomd_config.h>
#include <hoomd/HOOMDMath.h>

extern __shared__ unsigned int coords[];

__device__ Scalar gpu_interpolate_fraction(Scalar min,
                                      Scalar delta,
                                      unsigned int num_points,
                                      Scalar val,
                                      Scalar *histogram,
                                      Scalar *histogram_plus)
    {
    int lower_bin = (int) ((val - min)/delta);
    unsigned int upper_bin = lower_bin+1;

    // need to handly boundary case
    if (lower_bin < 0 || upper_bin >= num_points)
        return 0;

    Scalar rel_delta = (val - (Scalar)lower_bin*delta)/delta;

    Scalar lower_val = histogram_plus[lower_bin]/histogram[lower_bin];
    Scalar upper_val = histogram_plus[upper_bin]/histogram[upper_bin];

    return lower_val + rel_delta*(upper_val - lower_val);
    }

__device__ Scalar gpu_fraction_derivative(Scalar min,
                               Scalar max,
                               Scalar delta,
                               unsigned int num_points,
                               Scalar val,
                               Scalar *histogram,
                               Scalar *histogram_plus)
    {

    if (val - delta < min) 
        {
        // forward difference
        Scalar val2 = val + delta;

        Scalar y2 = gpu_interpolate_fraction(min,
                                             delta,
                                             num_points,
                                             val2,
                                             histogram,
                                             histogram_plus);
        Scalar y1 = gpu_interpolate_fraction(min,
                                             delta,
                                             num_points,
                                             val,
                                             histogram,
                                             histogram_plus);
        return (y2-y1)/delta;
        }
    else if (val + delta > max)
        {
        // backward difference
        Scalar val2 = val - delta;
        Scalar y1 = gpu_interpolate_fraction(min,
                                             delta,
                                             num_points,
                                             val2,
                                             histogram,
                                             histogram_plus);
        Scalar y2 = gpu_interpolate_fraction(min,
                                             delta,
                                             num_points,
                                             val,
                                             histogram,
                                             histogram_plus);
        return (y2-y1)/delta;
        }
    else
        {
        // central difference
        Scalar val2 = val + delta;
        Scalar val1 = val - delta;
        Scalar y1 = gpu_interpolate_fraction(min,
                                             delta,
                                             num_points,
                                             val1,
                                             histogram,
                                             histogram_plus);
        Scalar y2 = gpu_interpolate_fraction(min,
                                             delta,
                                             num_points,
                                             val2,
                                             histogram,
                                             histogram_plus);
        return (y2 - y1)/(Scalar(2.0)*delta);
        }
    } 

__global__ void gpu_update_grid_kernel(unsigned int num_elements,
                                       unsigned int *lengths,
                                       unsigned int dim,
                                       Scalar *current_val,
                                       Scalar *grid,
                                       Scalar *cv_min,
                                       Scalar *cv_max,
                                       Scalar *cv_sigma,
                                       Scalar scal,
                                       Scalar W,
                                       bool flux_tempered,
                                       Scalar T,
                                       Scalar *histogram,
                                       Scalar *histogram_plus,
                                       unsigned int num_histogram_entries,
                                       Scalar ftm_min,
                                       Scalar ftm_max)
    {
    unsigned int grid_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (grid_idx >= num_elements) return;

    if (! flux_tempered)
        {
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
    else // flux-tempered
        {
        Scalar min = cv_min[0];
        Scalar max = cv_max[0];
        unsigned int num_points = lengths[0];
        Scalar grid_delta = (max - min)/ (Scalar)(num_points -1);
        Scalar val = min + grid_idx*grid_delta;
     
        Scalar dfds = gpu_fraction_derivative(min,
                                              max,
                                              grid_delta,
                                              num_points,
                                              val,
                                              histogram,
                                              histogram_plus);
                                               
        Scalar hist = histogram[grid_idx];

        // normalize histogram
        hist /= num_histogram_entries; 

        Scalar del = -Scalar(1.0/2.0)*T*(logf(fabsf(dfds)) - logf(hist));
        grid[grid_idx] += del;
        } 
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
                     Scalar W,
                     bool flux_tempered,
                     Scalar T,
                     Scalar *d_histogram,
                     Scalar *d_histogram_plus,
                     unsigned int num_histogram_entries,
                     Scalar ftm_min,
                     Scalar ftm_max)
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
                                                                                 W,
                                                                                 flux_tempered,
                                                                                 T,
                                                                                 d_histogram,
                                                                                 d_histogram_plus,
                                                                                 num_histogram_entries,
                                                                                 ftm_min,
                                                                                 ftm_max);
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


