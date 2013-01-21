#include "DAFTGPU.cuh"

__device__ uint3 get_grid_pos(unsigned int grid_idx,
                              unsigned int nx,
                              unsigned int ny,
                              unsigned int nz)
    {
    uint3 pos;
    pos.x = grid_idx/ny/nz;
    pos.y = (grid_idx - pos.x*ny*nz)/nz;
    pos.z = (grid_idx - pos.x*ny*nz - pos.y*nz);
    return pos;
    }

__device__ unsigned int get_grid_idx(uint3 grid_pos,
                              unsigned int nx,
                              unsigned int ny,
                              unsigned int nz)
    {
    unsigned int idx = grid_pos.z + nz * (grid_pos.y + ny * grid_pos.x);
    return idx;
    }

__global__ void gpu_combine_buf_kernel(unsigned int n_cells,
                          cufftComplex *d_combine_buf,
                          const cufftComplex *d_stage_buf,
                          const bool sw,
                          const unsigned int n_current_dir,
                          const unsigned int stride)
    {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n_cells) return;

    if (sw)
        {
        cufftComplex local = d_combine_buf[idx];
        cufftComplex remote = d_stage_buf[idx];
        local.x += remote.x;
        local.y += remote.y;
        d_combine_buf[idx] = local;
        }
    else
        {
        cufftComplex local = d_combine_buf[idx];
        local.x *= Scalar(-1.0);
        local.y *= Scalar(-1.0);
        cufftComplex remote = d_stage_buf[idx];
        cufftComplex exp_fac, out;
        unsigned int grid_idx = idx % stride;
        exp_fac.x = cosf(Scalar(2.0*M_PI)*(Scalar)grid_idx/(Scalar)n_current_dir);
        exp_fac.y = sinf(Scalar(2.0*M_PI)*(Scalar)grid_idx/(Scalar)n_current_dir);
        out.x = exp_fac.x * (local.x + remote.x) - exp_fac.y * (local.y + remote.y);
        out.y = exp_fac.x * (local.y + remote.y) + exp_fac.y * (local.x + remote.x);
        d_combine_buf[idx] = out;
        } 
    }

void gpu_combine_buf(unsigned int n_cells,
                     cufftComplex *d_combine_buf,
                     const cufftComplex *d_stage_buf,
                     const bool sw,
                     const unsigned int n_current_dir,
                     const unsigned int stride)
    {
    unsigned int block_size = 512;
    unsigned int n_blocks = n_cells/block_size;
    if (n_cells % block_size) n_blocks+=1;
    gpu_combine_buf_kernel<<<n_blocks, block_size>>>(n_cells,  d_combine_buf, d_stage_buf, sw, n_current_dir, stride);
    }

__global__ void gpu_rotate_buf_kernel_z_y(unsigned int nx,
                                      unsigned int ny,
                                      unsigned int nz,
                                      const cufftComplex *d_combine_buf,
                                      cufftComplex *d_work_buf)
    {
    unsigned int idx = blockIdx.x * blockDim. x + threadIdx.x;

    if (idx >= nx*ny*nz) return;

    uint3 grid_pos = get_grid_pos(idx, nx, ny, nz);
    d_work_buf[grid_pos.y + ny * (grid_pos.x + nx * grid_pos.z)] =
        d_combine_buf[grid_pos.z + nz * (grid_pos.y + ny * grid_pos.x)];
    }

__global__ void gpu_rotate_buf_kernel_y_x(unsigned int nx,
                                      unsigned int ny,
                                      unsigned int nz,
                                      const cufftComplex *d_combine_buf,
                                      cufftComplex *d_work_buf)
    {
    unsigned int idx = blockIdx.x * blockDim. x + threadIdx.x;

    if (idx >= nx*ny*nz) return;

    uint3 grid_pos = get_grid_pos(idx, nx, ny, nz);
    d_work_buf[grid_pos.x + nx * (grid_pos.z + nz * grid_pos.y)] =
        d_combine_buf[grid_pos.y + ny * (grid_pos.x + nx * grid_pos.z)];
    } 

void gpu_rotate_buf_z_y(unsigned int nx,
                        unsigned int ny,
                        unsigned int nz,
                        const cufftComplex *d_combine_buf,
                        cufftComplex *d_work_buf)
    {
    unsigned int block_size = 512;
    unsigned int n_cells = nx*ny*nz;
    unsigned int n_blocks = n_cells/block_size;
    if (n_cells % block_size) n_blocks+=1;

    gpu_rotate_buf_kernel_z_y<<<n_blocks, block_size>>>(nx, ny, nz, d_combine_buf, d_work_buf);
    }

void gpu_rotate_buf_y_x(unsigned int nx,
                        unsigned int ny,
                        unsigned int nz,
                        const cufftComplex *d_combine_buf,
                        cufftComplex *d_work_buf)
    {
    unsigned int block_size = 512;
    unsigned int n_cells = nx*ny*nz;
    unsigned int n_blocks = n_cells/block_size;
    if (n_cells % block_size) n_blocks+=1;

    gpu_rotate_buf_kernel_y_x<<<n_blocks, block_size>>>(nx, ny, nz, d_combine_buf, d_work_buf);
    } 
