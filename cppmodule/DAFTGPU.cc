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
    unsigned int div,len;

    bool input_read = false;

    for (int i = 2; i >= 0 ; --i)
        {
        switch(i)
            {
            case 0:
                len = idx.getW();
                plan = m_cufft_plan_x;
                grid_idx = grid_pos.x;
                n_current_dir = m_nx;
                stride = m_ny*m_nz;
                break;
            case 1:
                len = idx.getH();
                plan = m_cufft_plan_y;
                grid_idx = grid_pos.y;
                n_current_dir = m_ny;
                stride = m_nx*m_nz;
                break;
            case 2:
                len = idx.getD();
                plan = m_cufft_plan_z;
                grid_idx = grid_pos.z;
                n_current_dir = m_nz;
                stride = m_nx*m_ny;
                break;
            }

        // Determine factorization of number of processors into S*L where S=2^i
        div = 1;
        while ((div < len)&& (len% (div*2) == 0)) div*=2;
        unsigned int L = len/div;

        /*
         * Perform DFT on long factors
         */
        
        if (L > 1)
            {
            for (unsigned int k = 0; k < L; ++k)
                {
                if (k == 0)
                    {
                    // initialize work buffer with local data
                    ArrayHandle<cufftComplex> d_work_buf(m_work_buf, access_location::device, access_mode::overwrite);

                    const GPUArray<cufftComplex>& src_array = ((i == 2) ? in : m_combine_buf);
                    if (i == 2) input_read = true;

                    ArrayHandle<cufftComplex> d_src_buf(src_array, access_location::device, access_mode::read);
                    cudaMemcpy(d_work_buf.data,d_src_buf.data, sizeof(cufftComplex)*local_size, cudaMemcpyDeviceToDevice);
                    }
                else
                    {
                    // initialize work buffer with remote data
                    ArrayHandle<cufftComplex> d_work_buf(m_work_buf, access_location::device, access_mode::overwrite);
                    ArrayHandle<cufftComplex> d_stage_buf(m_stage_buf, access_location::device, access_mode::read);
                    cudaMemcpy(d_work_buf.data,d_stage_buf.data, sizeof(cufftComplex)*local_size, cudaMemcpyDeviceToDevice);
                    }

                // receive from below and send to above using a systolic ring
                int down = grid_idx - div;
                if (down < 0) down += L*div;

                int up = grid_idx + div;
                if (up >= (int)(L*div)) up -= L*div;

                uint3 up_grid_pos = grid_pos;
                uint3 down_grid_pos = grid_pos;

                switch (i)
                    {
                    case 0:
                        up_grid_pos.x = up;
                        down_grid_pos.x = down;
                        break;
                    case 1:
                        up_grid_pos.y = up;
                        down_grid_pos.y = down;
                        break;
                    case 2:
                        up_grid_pos.z = up;
                        down_grid_pos.z = down;
                        break;
                    }

                unsigned int up_rank = idx(up_grid_pos.x, up_grid_pos.y, up_grid_pos.z);
                unsigned int down_rank = idx(down_grid_pos.x, down_grid_pos.y, down_grid_pos.z);

                    {
                    #ifdef ENABLE_MPI_CUDA
                    ArrayHandle<cufftComplex> work_buf_handle(m_work_buf, access_location::device, access_mode::read);
                    ArrayHandle<cufftComplex> stage_buf_handle(m_stage_buf, access_location::device, access_mode::overwrite);
                    #else
                    ArrayHandle<cufftComplex> work_buf_handle(m_work_buf, access_location::host, access_mode::read);
                    ArrayHandle<cufftComplex> stage_buf_handle(m_stage_buf, access_location::host, access_mode::overwrite);
                    #endif

                    MPI_Request req[2];
                    MPI_Isend(work_buf_handle.data,
                              local_size*sizeof(cufftComplex),
                              MPI_BYTE,
                              up_rank,
                              0,
                              m_exec_conf->getMPICommunicator(),
                              &req[0]);
                    MPI_Irecv(stage_buf_handle.data,
                              local_size*sizeof(cufftComplex),
                              MPI_BYTE,
                              down_rank,
                              0,
                              m_exec_conf->getMPICommunicator(),
                              &req[1]);

                    MPI_Status stat[2];
                    MPI_Waitall(2, req, stat);
                    } 

                if (k == 0)
                    {
                    // reset combining buffer
                    ArrayHandle<cufftComplex> d_combine_buf(m_combine_buf, access_location::device, access_mode::overwrite);
                    cudaMemset(d_combine_buf.data, 0, sizeof(cufftComplex)*local_size);
                    }

                    {
                    // do partial DFT
                    ArrayHandle<cufftComplex> d_combine_buf(m_combine_buf, access_location::device, access_mode::readwrite);
                    ArrayHandle<cufftComplex> d_stage_buf(m_stage_buf, access_location::device, access_mode::read);

                    int remote = down/div - k;
                    if (remote < 0) remote += L;

                    gpu_partial_dft(grid_idx / div,
                                    remote,
                                   (grid_idx % div)* n_current_dir,
                                    L,
                                    m_nx,
                                    m_ny,
                                    m_nz,
                                    i,
                                    n_current_dir * len,
                                    stride,
                                    d_combine_buf.data,
                                    d_stage_buf.data);

                    if (m_exec_conf->isCUDAErrorCheckingEnabled())
                        CHECK_CUDA_ERROR();
                    } 

                    #if 0
                    {
                    ArrayHandle<cufftComplex> h_stage_buf(m_stage_buf, access_location::host, access_mode::read);
                    ArrayHandle<cufftComplex> h_combine_buf(m_combine_buf, access_location::host, access_mode::read);
                    std::cout << "R " << m_exec_conf->getRank() << " recvd " << h_stage_buf.data[0].x << " " << h_stage_buf.data[0].y << std::endl;
                    std::cout << "R " << m_exec_conf->getRank() << " combine " << h_combine_buf.data[0].x << " " << h_combine_buf.data[0].y << std::endl;
                    }
                    #endif
 
                }
            }

        /*
         * Perform power-of-two FFT
         */

        while (div >= 1)
            {
            if (div == 1)
                {
                /*
                 * do local FFT
                 */
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
                        if (m_exec_conf->isCUDAErrorCheckingEnabled())
                            CHECK_CUDA_ERROR();
                        break;
                        }
                    case 0:
                        {
                        ArrayHandle<cufftComplex> d_combine_buf(m_combine_buf, access_location::device, access_mode::read);
                        ArrayHandle<cufftComplex> d_work_buf(m_work_buf, access_location::device, access_mode::overwrite);

                        // rotate to column-major in x coordinate
                        gpu_rotate_buf_y_x(m_nx, m_ny, m_nz, d_combine_buf.data, d_work_buf.data);
                        if (m_exec_conf->isCUDAErrorCheckingEnabled())
                            CHECK_CUDA_ERROR();
                        break;
                        }
                    }

                    {
                    const GPUArray<cufftComplex>& src_array = (i == 2) ? ( input_read ? m_combine_buf : in): m_work_buf;
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

#if 0
                    {
                    std::cout << "R " << m_exec_conf->getRank() << " i == " << i << std::endl;
                    const GPUArray<cufftComplex>& src_array = (i == 2) ? ( (idx.getD() == 1) ? in : m_combine_buf): m_work_buf;
                    const GPUArray<cufftComplex>& dest_array = (i == 0 && div == 1) ? out : m_combine_buf;
                    ArrayHandle<cufftComplex> h_src_buf(src_array, access_location::host, access_mode::read);
                    ArrayHandle<cufftComplex> h_out_buf(dest_array, access_location::host, access_mode::read);
                    std::cout << "R " << m_exec_conf->getRank() << " src " << h_src_buf.data[0].x << " " << h_src_buf.data[0].y << std::endl;
                    std::cout << "R " << m_exec_conf->getRank() << " dst " << h_out_buf.data[0].x << " " << h_out_buf.data[0].y << std::endl;
                    }
#endif
                }
            else
                {
                // combine data from different processors
                int dir;
                if ((grid_idx % div) / (div/2))
                    dir = -1;
                else
                    dir = 1;

                if (!input_read)
                    {
                    // copy over data
                    ArrayHandle<cufftComplex> d_combine_buf(m_combine_buf, access_location::device, access_mode::overwrite);
                    ArrayHandle<cufftComplex> d_in_buf(in, access_location::device, access_mode::read);
                    cudaMemcpy(d_combine_buf.data, d_in_buf.data, sizeof(cufftComplex)*local_size, cudaMemcpyDeviceToDevice);
                    input_read = true;
                    }

                    {
                    // send all cells to processor N/2 domains away in direction dir
                    #ifdef ENABLE_MPI_CUDA
                    ArrayHandle<cufftComplex> combine_buf_handle(m_combine_buf, access_location::device, access_mode::read);
                    ArrayHandle<cufftComplex> stage_buf_handle(m_stage_buf, access_location::device, access_mode::overwrite);
                    #else
                    ArrayHandle<cufftComplex> combine_buf_handle(m_combine_buf, access_location::host, access_mode::read);
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
                    MPI_Isend(combine_buf_handle.data,
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
                 
#if 0
                    {
                    ArrayHandle<cufftComplex> h_stage_buf(m_stage_buf, access_location::host, access_mode::read);
                    std::cout << "R " << m_exec_conf->getRank() << " recvd " << h_stage_buf.data[0].x << " " << h_stage_buf.data[0].y << std::endl;
                    }
#endif
                    {
                    // combine data sets
                    ArrayHandle<cufftComplex> d_combine_buf(m_combine_buf, access_location::device, access_mode::readwrite);

                    ArrayHandle<cufftComplex> d_stage_buf(m_stage_buf, access_location::device, access_mode::read);

                    gpu_combine_buf(local_size,
                                    d_combine_buf.data,
                                    d_stage_buf.data,
                                    dir == 1,
                                    n_current_dir*div,
                                    (dir == -1) ? n_current_dir* (grid_idx - div/2) : 0,
                                    stride);
                    if (m_exec_conf->isCUDAErrorCheckingEnabled())
                        CHECK_CUDA_ERROR();
                    }

                }
            div /= 2;
            }
        }
    }

