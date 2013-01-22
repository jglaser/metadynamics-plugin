#include <hoomd/hoomd_config.h>
#include <hoomd/HOOMDMath.h>

#include <cufft.h>


void gpu_sande_tukey(unsigned int n_cells,
                 cufftComplex *d_combine_buf,
                 const cufftComplex *d_stage_buf,
                 const bool sw,
                 const unsigned int n_current_dir,
                 const unsigned int offset,
                 const unsigned int stride);

void gpu_cooley_tukey(unsigned int n_cells,
                     cufftComplex *d_combine_buf,
                     const cufftComplex *d_stage_buf,
                     const bool sw,
                     const unsigned int n_current_dir,
                     const unsigned int offset,
                     const unsigned int stride); 

void gpu_rotate_buf_z_y(unsigned int nx,
                        unsigned int ny,
                        unsigned int nz,
                        const cufftComplex *d_combine_buf,
                        cufftComplex *d_work_buf);

void gpu_rotate_buf_y_x(unsigned int nx,
                        unsigned int ny,
                        unsigned int nz,
                        const cufftComplex *d_combine_buf,
                        cufftComplex *d_work_buf);

void gpu_rotate_buf_x_y(unsigned int nx,
                        unsigned int ny,
                        unsigned int nz,
                        const cufftComplex *d_combine_buf,
                        cufftComplex *d_work_buf);

void gpu_rotate_buf_y_z(unsigned int nx,
                        unsigned int ny,
                        unsigned int nz,
                        const cufftComplex *d_combine_buf,
                        cufftComplex *d_work_buf);

void gpu_partial_dft(const unsigned int long_idx,
                     const unsigned int long_idx_remote,
                     const unsigned int offset,
                     const unsigned int L,
                     const unsigned int nx,
                     const unsigned int ny,
                     const unsigned int nz,
                     const unsigned int dir,
                     const unsigned int N,
                     const unsigned int stride,
                     const bool inverse,
                     cufftComplex *d_combine_buf,
                     const cufftComplex *d_stage_buf);
 
