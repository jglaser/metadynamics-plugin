#include "DAFTGPU.h"
#include "DAFTGPU.cuh"

DAFTGPU::DAFTGPU(boost::shared_ptr<const ExecutionConfiguration> exec_conf,
        boost::shared_ptr<DomainDecomposition> decomposition,
        unsigned int nx,
        unsigned int ny,
        unsigned int nz
        )
    : m_exec_conf(exec_conf),
      m_decomposition(decomposition),
      m_nx(nx),
      m_ny(ny),
      m_nz(nz)
    {
    unsigned int buf_size = nx*ny*nz;

    GPUArray<cufftComplex> stage_buf(buf_size, m_exec_conf);
    m_stage_buf.swap(stage_buf);

    GPUArray<cufftComplex> combine_buf(buf_size, m_exec_conf);
    m_combine_buf.swap(combine_buf);

    GPUArray<cufftComplex> work_buf(buf_size, m_exec_conf);
    m_work_buf.swap(work_buf);

    cufftPlan1d(&m_cufft_plan_x, nx, CUFFT_C2C, ny*nz);
    cufftPlan1d(&m_cufft_plan_y, ny, CUFFT_C2C, nx*nz);
    cufftPlan1d(&m_cufft_plan_z, nz, CUFFT_C2C, nx*ny);
    }

DAFTGPU::~DAFTGPU()
    {
    cufftDestroy(m_cufft_plan_x);
    cufftDestroy(m_cufft_plan_y);
    cufftDestroy(m_cufft_plan_z);
    }

void DAFTGPU::forwardFFT3D(const GPUArray<cufftComplex>& in, const GPUArray<cufftComplex>& out)
    {
    Index3D idx;
    if (m_decomposition)
        idx = m_decomposition->getDomainIndexer();
    else
        idx = Index3D(1,1,1);


    uint3 grid_pos = idx.getTriple(m_exec_conf->getRank());

    unsigned int local_size = m_nx*m_ny*m_nz;
    unsigned int grid_idx;
    unsigned int n_current_dir;
    unsigned int stride;
    cufftHandle plan;
    unsigned int div;

    for (int i = 2; i >= 0 ; --i)
        {
        switch(i)
            {
            case 0:
                div = idx.getW();
                plan = m_cufft_plan_x;
                grid_idx = grid_pos.x;
                n_current_dir = m_nx;
                stride = m_ny*m_nz;
                break;
            case 1:
                div = idx.getH();
                plan = m_cufft_plan_y;
                grid_idx = grid_pos.y;
                n_current_dir = m_ny;
                stride = m_nx*m_nz;
                break;
            case 2:
                div = idx.getD();
                plan = m_cufft_plan_z;
                grid_idx = grid_pos.z;
                n_current_dir = m_nz;
                stride = m_nx*m_ny;
                break;
            }

        while (div >= 1)
            {
            if (div == 1)
                {
                // do local FFT
                    {
                    switch(i)
                        {
                        case 2:
                            // do nothing
                            break;
                        case 1:
                            {
                            ArrayHandle<cufftComplex> d_combine_buf(m_combine_buf, access_location::device, access_mode::read);
                            ArrayHandle<cufftComplex> d_work_buf(m_work_buf, access_location::device, access_mode::overwrite);
                            // rotate to column-major in y coordinate
                            gpu_rotate_buf_z_y(m_nx, m_ny, m_nz, d_combine_buf.data, d_work_buf.data);
                            break;
                            }
                        case 0:
                            {
                            ArrayHandle<cufftComplex> d_combine_buf(m_combine_buf, access_location::device, access_mode::read);
                            ArrayHandle<cufftComplex> d_work_buf(m_work_buf, access_location::device, access_mode::overwrite);
                            // rotate to column-major in x coordinate
                            gpu_rotate_buf_y_x(m_nx, m_ny, m_nz, d_combine_buf.data, d_work_buf.data);
                            break;
                            }
                        }
                    }

                    {
                    const GPUArray<cufftComplex>& src_array = (i == 2) ? ( (idx.getD() == 1) ? in : m_combine_buf): m_work_buf;
                    const GPUArray<cufftComplex>& dest_array = (i == 0 && div == 1) ? out : m_combine_buf;
                    ArrayHandle<cufftComplex> d_src_buf(src_array, access_location::device, access_mode::read);
                    ArrayHandle<cufftComplex> d_out_buf(dest_array, access_location::device, access_mode::overwrite);

                    // call FFT library

                    // handle case where one dimension is unity
                    if (n_current_dir > 1)
                        cufftExecC2C(plan, d_src_buf.data, d_out_buf.data, CUFFT_FORWARD);
                    else
                        cudaMemcpy(d_out_buf.data, d_src_buf.data, stride*sizeof(cufftComplex),cudaMemcpyDeviceToDevice);
                    }

                }
            else
                {
                // combine data from different processors
                int dir;
                if ((grid_idx % div) / (div/2))
                    dir = -1;
                else
                    dir = 1;

                const GPUArray<cufftComplex>& src_array = (div == idx.getW()) ? in : m_combine_buf;
                    {
                    // send all cells to processor N/2 domains away in direction dir
                    #ifdef ENABLE_MPI_CUDA
                    ArrayHandle<cufftComplex> in_handle(src_array, access_location::device, access_mode::read);
                    ArrayHandle<cufftComplex> stage_buf_handle(m_stage_buf, access_location::device, access_mode::overwrite);
                    #else
                    ArrayHandle<cufftComplex> in_handle(src_array, access_location::host, access_mode::read);
                    ArrayHandle<cufftComplex> stage_buf_handle(m_stage_buf, access_location::host, access_mode::overwrite);
                    #endif

                    int pt_grid_idx = (int) grid_idx + dir * (int)div/2;
                    uint3 pt_grid_pos = grid_pos;

                    switch(i)
                        {
                        case 0:
                            pt_grid_pos.x = pt_grid_idx;
                            break;
                        case 1:
                            pt_grid_pos.y = pt_grid_idx;
                            break;
                        case 2:
                            pt_grid_pos.z = pt_grid_idx;
                            break;
                        }

                    unsigned int pt_rank = idx(pt_grid_pos.x, pt_grid_pos.y, pt_grid_pos.z);

                    MPI_Request req[2];
                    MPI_Isend(in_handle.data,
                              local_size*sizeof(cufftComplex),
                              MPI_BYTE,
                              pt_rank,
                              0,
                              m_exec_conf->getMPICommunicator(),
                              &req[0]);
                    MPI_Irecv(stage_buf_handle.data,
                              local_size*sizeof(cufftComplex),
                              MPI_BYTE,
                              pt_rank,
                              0,
                              m_exec_conf->getMPICommunicator(),
                              &req[1]);

                    MPI_Status stat[2];
                    MPI_Waitall(2, req, stat);
                    }
                   
                    {
                    // combine data sets
                    ArrayHandle<cufftComplex> d_combine_buf(m_combine_buf, access_location::device, access_mode::readwrite);

                    ArrayHandle<cufftComplex> d_stage_buf(m_stage_buf, access_location::device, access_mode::read);

                    gpu_combine_buf(local_size,
                                    d_combine_buf.data,
                                    d_stage_buf.data,
                                    dir == 1,
                                    n_current_dir,
                                    stride);
                    }

                }
            div /= 2;
            }
        }
    }

