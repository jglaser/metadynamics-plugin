#include "OrderParameterMeshGPU.h"

namespace py = pybind11;

#ifdef ENABLE_CUDA
#include "OrderParameterMeshGPU.cuh"

/*! \param sysdef The system definition
    \param nx Number of cells along first axis
    \param ny Number of cells along second axis
    \param nz Number of cells along third axis
    \param mode Per-type modes to multiply density
    \param zero_modes List of modes that should be zeroed
 */
OrderParameterMeshGPU::OrderParameterMeshGPU(std::shared_ptr<SystemDefinition> sysdef,
                                            const unsigned int nx,
                                            const unsigned int ny,
                                            const unsigned int nz,
                                            std::vector<Scalar> mode,
                                            std::vector<int3> zero_modes)
    : OrderParameterMesh(sysdef, nx, ny, nz, mode,zero_modes),
      m_local_fft(true),
      m_sum(m_exec_conf),
      m_block_size(256),
      m_gpu_q_max(m_exec_conf)
    {
    unsigned int n_blocks = m_mesh_points.x*m_mesh_points.y*m_mesh_points.z/m_block_size+1;
    GlobalArray<Scalar> sum_partial(n_blocks,m_exec_conf);
    m_sum_partial.swap(sum_partial);

    GlobalArray<Scalar> sum_virial_partial(6*n_blocks,m_exec_conf);
    m_sum_virial_partial.swap(sum_virial_partial);

    GlobalArray<Scalar> sum_virial(6,m_exec_conf);
    m_sum_virial.swap(sum_virial);

    GlobalArray<Scalar4> max_partial(n_blocks, m_exec_conf);
    m_max_partial.swap(max_partial);

    // initial value of number of particles per bin
    m_cell_size = 2;
    }

OrderParameterMeshGPU::~OrderParameterMeshGPU()
    {
    if (m_local_fft)
        cufftDestroy(m_cufft_plan);
    #ifdef ENABLE_MPI
    else
        {
        dfft_destroy_plan(m_dfft_plan_forward);
        dfft_destroy_plan(m_dfft_plan_inverse);
        }
    #endif
    }

void OrderParameterMeshGPU::initializeFFT()
    {
    #ifdef ENABLE_MPI
    m_local_fft = !m_pdata->getDomainDecomposition();

    if (! m_local_fft)
        {
        // ghost cell communicator for charge interpolation
        m_gpu_grid_comm_forward = std::unique_ptr<CommunicatorGridGPUComplex>(
            new CommunicatorGridGPUComplex(m_sysdef,
               make_uint3(m_mesh_points.x, m_mesh_points.y, m_mesh_points.z),
               make_uint3(m_grid_dim.x, m_grid_dim.y, m_grid_dim.z),
               m_n_ghost_cells,
               true));
        // ghost cell communicator for force mesh
        m_gpu_grid_comm_reverse = std::unique_ptr<CommunicatorGridGPUComplex >(
            new CommunicatorGridGPUComplex(m_sysdef,
               make_uint3(m_mesh_points.x, m_mesh_points.y, m_mesh_points.z),
               make_uint3(m_grid_dim.x, m_grid_dim.y, m_grid_dim.z),
               m_n_ghost_cells,
               false));

        // set up distributed FFT
        int gdim[3];
        int pdim[3];
        Index3D decomp_idx = m_pdata->getDomainDecomposition()->getDomainIndexer();
        pdim[0] = decomp_idx.getD();
        pdim[1] = decomp_idx.getH();
        pdim[2] = decomp_idx.getW();
        gdim[0] = m_mesh_points.z*pdim[0];
        gdim[1] = m_mesh_points.y*pdim[1];
        gdim[2] = m_mesh_points.x*pdim[2];
        int embed[3];
        embed[0] = m_mesh_points.z+2*m_n_ghost_cells.z;
        embed[1] = m_mesh_points.y+2*m_n_ghost_cells.y;
        embed[2] = m_mesh_points.x+2*m_n_ghost_cells.x;
        m_ghost_offset = (m_n_ghost_cells.z*embed[1]+m_n_ghost_cells.y)*embed[2]+m_n_ghost_cells.x;
        uint3 pcoord = m_pdata->getDomainDecomposition()->getGridPos();
        int pidx[3];
        pidx[0] = pcoord.z;
        pidx[1] = pcoord.y;
        pidx[2] = pcoord.x;
        int row_m = 0; /* both local grid and proc grid are row major, no transposition necessary */
        ArrayHandle<unsigned int> h_cart_ranks(m_pdata->getDomainDecomposition()->getCartRanks(),
            access_location::host, access_mode::read);
        #ifndef USE_HOST_DFFT
        dfft_cuda_create_plan(&m_dfft_plan_forward, 3, gdim, embed, NULL, pdim, pidx,
            row_m, 0, 1, m_exec_conf->getMPICommunicator(), (int *) h_cart_ranks.data);
        dfft_cuda_create_plan(&m_dfft_plan_inverse, 3, gdim, NULL, embed, pdim, pidx,
            row_m, 0, 1, m_exec_conf->getMPICommunicator(), (int *)h_cart_ranks.data);
        #else
        dfft_create_plan(&m_dfft_plan_forward, 3, gdim, embed, NULL, pdim, pidx,
            row_m, 0, 1, m_exec_conf->getMPICommunicator(), (int *) h_cart_ranks.data);
        dfft_create_plan(&m_dfft_plan_inverse, 3, gdim, NULL, embed, pdim, pidx,
            row_m, 0, 1, m_exec_conf->getMPICommunicator(), (int *) h_cart_ranks.data);
        #endif
        }
    #endif // ENABLE_MPI

    if (m_local_fft)
        {
        cufftPlan3d(&m_cufft_plan, m_mesh_points.z, m_mesh_points.y, m_mesh_points.x, CUFFT_C2C);
        }

    unsigned int n_particle_bins = m_grid_dim.x*m_grid_dim.y*m_grid_dim.z;
    m_bin_idx = Index2D(n_particle_bins,m_cell_size);
    m_scratch_idx = Index2D(n_particle_bins,(2*m_radius+1)*(2*m_radius+1)*(2*m_radius+1));

    // allocate mesh and transformed mesh
    GlobalArray<cufftComplex> mesh(m_n_cells,m_exec_conf);
    m_mesh.swap(mesh);

    GlobalArray<cufftComplex> fourier_mesh(m_n_inner_cells, m_exec_conf);
    m_fourier_mesh.swap(fourier_mesh);

    GlobalArray<cufftComplex> fourier_mesh_G(m_n_inner_cells, m_exec_conf);
    m_fourier_mesh_G.swap(fourier_mesh_G);

    GlobalArray<cufftComplex> inv_fourier_mesh(m_n_cells, m_exec_conf);
    m_inv_fourier_mesh.swap(inv_fourier_mesh);

    GlobalArray<Scalar4> particle_bins(m_bin_idx.getNumElements(), m_exec_conf);
    m_particle_bins.swap(particle_bins);

    GlobalArray<unsigned int> n_cell(m_bin_idx.getW(), m_exec_conf);
    m_n_cell.swap(n_cell);

    GPUFlags<unsigned int> cell_overflowed(m_exec_conf);
    m_cell_overflowed.swap(cell_overflowed);

    m_cell_overflowed.resetFlags(0);

    // allocate scratch space for density reduction
    GlobalArray<Scalar> mesh_scratch(m_scratch_idx.getNumElements(), m_exec_conf);
    m_mesh_scratch.swap(mesh_scratch);
    }

//! Assignment of particles to mesh using three-point scheme (triangular shaped cloud)
/*! This is a second order accurate scheme with continuous value and continuous derivative
 */
void OrderParameterMeshGPU::assignParticles()
    {
    if (m_prof) m_prof->push(m_exec_conf, "assign");

    ArrayHandle<Scalar4> d_postype(m_pdata->getPositions(), access_location::device, access_mode::read);
    ArrayHandle<cufftComplex> d_mesh(m_mesh, access_location::device, access_mode::overwrite);
    ArrayHandle<Scalar> d_mode(m_mode, access_location::device, access_mode::read);

    ArrayHandle<unsigned int> d_n_cell(m_n_cell, access_location::device, access_mode::overwrite);

    bool cont = true;
    while (cont)
        {
        cudaMemset(d_n_cell.data,0,sizeof(unsigned int)*m_n_cell.getNumElements());

            {
            ArrayHandle<Scalar4> d_particle_bins(m_particle_bins, access_location::device, access_mode::overwrite);

            gpu_bin_particles(m_pdata->getN(),
                              d_postype.data,
                              d_particle_bins.data,
                              d_n_cell.data,
                              m_cell_overflowed.getDeviceFlags(),
                              m_bin_idx,
                              m_mesh_points,
                              m_n_ghost_cells,
                              d_mode.data,
                              m_pdata->getBox());

            if (m_exec_conf->isCUDAErrorCheckingEnabled())
                CHECK_CUDA_ERROR();
            }

        unsigned int flags = m_cell_overflowed.readFlags();

        if (flags)
            {
            // reallocate particle bins array
            m_cell_size = flags;

            m_bin_idx = Index2D(m_bin_idx.getW(),m_cell_size);
            GlobalArray<Scalar4> particle_bins(m_bin_idx.getNumElements(),m_exec_conf);
            m_particle_bins.swap(particle_bins);
            m_cell_overflowed.resetFlags(0);
            }
        else
            {
            cont = false;
            }

        // assign particles to mesh
        ArrayHandle<Scalar4> d_particle_bins(m_particle_bins, access_location::device, access_mode::read);
        ArrayHandle<Scalar> d_mesh_scratch(m_mesh_scratch, access_location::device, access_mode::overwrite);

        gpu_assign_binned_particles_to_mesh(m_mesh_points,
                                            m_n_ghost_cells,
                                            m_grid_dim,
                                            d_particle_bins.data,
                                            d_mesh_scratch.data,
                                            m_bin_idx,
                                            m_scratch_idx,
                                            d_n_cell.data,
                                            d_mesh.data);

        if (m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();
        }

    // reduce squares of mode amplitude
    m_mode_sq = gpu_compute_mode_sq(m_pdata->getN(),
        d_postype.data,
        d_mode.data,
        m_exec_conf->getCachedAllocator());

    #ifdef ENABLE_MPI
    if (m_pdata->getDomainDecomposition())
        {
        // reduce sum
        MPI_Allreduce(MPI_IN_PLACE,
                      &m_mode_sq,
                      1,
                      MPI_HOOMD_SCALAR,
                      MPI_SUM,
                      m_exec_conf->getMPICommunicator());
        }
    #endif

    if (m_prof) m_prof->pop(m_exec_conf);
    }

void OrderParameterMeshGPU::updateMeshes()
    {
    if (m_local_fft)
        {
        if (m_prof) m_prof->push(m_exec_conf,"FFT");
        // locally transform the particle mesh
        ArrayHandle<cufftComplex> d_mesh(m_mesh, access_location::device, access_mode::read);
        ArrayHandle<cufftComplex> d_fourier_mesh(m_fourier_mesh, access_location::device, access_mode::overwrite);

        cufftExecC2C(m_cufft_plan, d_mesh.data, d_fourier_mesh.data, CUFFT_FORWARD);
        if (m_prof) m_prof->pop(m_exec_conf);
        }
    #ifdef ENABLE_MPI
    else
        {
        // update inner cells of particle mesh
        if (m_prof) m_prof->push(m_exec_conf,"ghost cell update");
        m_exec_conf->msg->notice(8) << "cv.mesh: Ghost cell update" << std::endl;
        m_gpu_grid_comm_forward->communicate(m_mesh);
        if (m_prof) m_prof->pop(m_exec_conf);

        // perform a distributed FFT
        m_exec_conf->msg->notice(8) << "cv.mesh: Distributed FFT mesh" << std::endl;
        if (m_prof) m_prof->push(m_exec_conf,"FFT");
        #ifndef USE_HOST_DFFT
        ArrayHandle<cufftComplex> d_mesh(m_mesh, access_location::device, access_mode::read);
        ArrayHandle<cufftComplex> d_fourier_mesh(m_fourier_mesh, access_location::device, access_mode::overwrite);

        if (m_exec_conf->isCUDAErrorCheckingEnabled())
            dfft_cuda_check_errors(&m_dfft_plan_forward, 1);
        else
            dfft_cuda_check_errors(&m_dfft_plan_forward, 0);

        dfft_cuda_execute(d_mesh.data+m_ghost_offset, d_fourier_mesh.data, 0, &m_dfft_plan_forward);
        #else
        ArrayHandle<cufftComplex> h_mesh(m_mesh, access_location::host, access_mode::read);
        ArrayHandle<cufftComplex> h_fourier_mesh(m_fourier_mesh, access_location::host, access_mode::overwrite);

        dfft_execute((cpx_t *)(h_mesh.data+m_ghost_offset), (cpx_t *)h_fourier_mesh.data, 0,m_dfft_plan_forward);
        #endif
        if (m_prof) m_prof->pop(m_exec_conf);
        }
    #endif

    if (m_prof) m_prof->push(m_exec_conf,"update");

        {
        ArrayHandle<cufftComplex> d_fourier_mesh(m_fourier_mesh, access_location::device, access_mode::readwrite);
        ArrayHandle<cufftComplex> d_fourier_mesh_G(m_fourier_mesh_G, access_location::device, access_mode::overwrite);

        ArrayHandle<Scalar> d_interpolation_f(m_interpolation_f, access_location::device, access_mode::read);
        ArrayHandle<Scalar3> d_k(m_k, access_location::device, access_mode::read);

        gpu_update_meshes(m_n_inner_cells,
                          d_fourier_mesh.data,
                          d_fourier_mesh_G.data,
                          d_interpolation_f.data,
                          m_mode_sq,
                          d_k.data,
                          m_pdata->getNGlobal());

        if (m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();
        }

    if (m_prof) m_prof->pop(m_exec_conf);

    if (m_local_fft)
        {
        if (m_prof) m_prof->push(m_exec_conf, "FFT");

        // do local inverse transform of force mesh
        ArrayHandle<cufftComplex> d_fourier_mesh_G(m_fourier_mesh_G, access_location::device, access_mode::read);
        ArrayHandle<cufftComplex> d_inv_fourier_mesh(m_inv_fourier_mesh, access_location::device, access_mode::overwrite);

        cufftExecC2C(m_cufft_plan,
                     d_fourier_mesh_G.data,
                     d_inv_fourier_mesh.data,
                     CUFFT_INVERSE);
        if (m_prof) m_prof->pop(m_exec_conf);
        }
    #ifdef ENABLE_MPI
    else
        {
        if (m_prof) m_prof->push(m_exec_conf, "FFT");

        // Distributed inverse transform of force mesh
        m_exec_conf->msg->notice(8) << "cv.mesh: Distributed iFFT" << std::endl;
        #ifndef USE_HOST_DFFT
        ArrayHandle<cufftComplex> d_fourier_mesh_G(m_fourier_mesh_G, access_location::device, access_mode::read);
        ArrayHandle<cufftComplex> d_inv_fourier_mesh(m_inv_fourier_mesh, access_location::device, access_mode::overwrite);

        if (m_exec_conf->isCUDAErrorCheckingEnabled())
            dfft_cuda_check_errors(&m_dfft_plan_inverse, 1);
        else
            dfft_cuda_check_errors(&m_dfft_plan_inverse, 0);

        dfft_cuda_execute(d_fourier_mesh_G.data, d_inv_fourier_mesh.data+m_ghost_offset, 1, &m_dfft_plan_inverse);
        #else
        ArrayHandle<cufftComplex> h_fourier_mesh_G(m_fourier_mesh_G, access_location::host, access_mode::read);
        ArrayHandle<cufftComplex> h_inv_fourier_mesh(m_inv_fourier_mesh, access_location::host, access_mode::overwrite);
        dfft_execute((cpx_t *)h_fourier_mesh_G.data, (cpx_t *)h_inv_fourier_mesh.data+m_ghost_offset, 1, m_dfft_plan_inverse);
        #endif
        if (m_prof) m_prof->pop(m_exec_conf);
        }
    #endif

    #ifdef ENABLE_MPI
    if (! m_local_fft)
        {
        // update outer cells of inverse Fourier mesh using ghost cells from neighboring processors
        if (m_prof) m_prof->push("ghost cell update");
        m_exec_conf->msg->notice(8) << "cv.mesh: Ghost cell update" << std::endl;
        m_gpu_grid_comm_reverse->communicate(m_inv_fourier_mesh);
        if (m_prof) m_prof->pop();
        }
    #endif
    }

void OrderParameterMeshGPU::interpolateForces()
    {
    if (m_prof) m_prof->push(m_exec_conf,"forces");

    ArrayHandle<Scalar4> d_postype(m_pdata->getPositions(), access_location::device, access_mode::read);
    ArrayHandle<cufftComplex> d_inv_fourier_mesh(m_inv_fourier_mesh, access_location::device, access_mode::read);
    ArrayHandle<Scalar> d_mode(m_mode, access_location::device, access_mode::read);

    ArrayHandle<Scalar4> d_force(m_force, access_location::device, access_mode::overwrite);

    gpu_compute_forces(m_pdata->getN(),
                       d_postype.data,
                       d_force.data,
                       m_bias,
                       d_inv_fourier_mesh.data,
                       m_grid_dim,
                       m_n_ghost_cells,
                       d_mode.data,
                       m_pdata->getBox(),
                       m_pdata->getGlobalBox(),
                       m_pdata->getNGlobal(),
                       1.0);

    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();

    if (m_prof) m_prof->pop(m_exec_conf);
    }

void OrderParameterMeshGPU::computeVirial()
    {
    if (m_prof) m_prof->push(m_exec_conf,"virial");

    ArrayHandle<cufftComplex> d_fourier_mesh(m_fourier_mesh, access_location::device, access_mode::read);
    ArrayHandle<cufftComplex> d_fourier_mesh_G(m_fourier_mesh_G, access_location::device, access_mode::read);
    ArrayHandle<Scalar3> d_k(m_k, access_location::device, access_mode::read);
    ArrayHandle<Scalar> d_virial_mesh(m_virial_mesh, access_location::device, access_mode::overwrite);

    bool exclude_dc = true;
    #ifdef ENABLE_MPI
    if (m_pdata->getDomainDecomposition())
        {
        uint3 my_pos = m_pdata->getDomainDecomposition()->getGridPos();
        exclude_dc = !my_pos.x && !my_pos.y && !my_pos.z;
        }
    #endif

    ArrayHandle<Scalar> d_table_D(m_table_d, access_location::device, access_mode::read);

    gpu_compute_mesh_virial(m_n_inner_cells,
                            d_fourier_mesh.data,
                            d_fourier_mesh_G.data,
                            d_virial_mesh.data,
                            d_k.data,
                            exclude_dc,
                            m_k_min,
                            m_k_max,
                            m_delta_k,
                            d_table_D.data,
                            m_use_table,
                            m_pdata->getNGlobal(),
                            1.0);

    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();

        {
        ArrayHandle<Scalar> d_sum_virial(m_sum_virial, access_location::device, access_mode::overwrite);
        ArrayHandle<Scalar> d_sum_virial_partial(m_sum_virial_partial, access_location::device, access_mode::overwrite);

        gpu_compute_virial(m_n_inner_cells,
                           d_sum_virial_partial.data,
                           d_sum_virial.data,
                           d_virial_mesh.data,
                           m_block_size);

        if (m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();
        }

    ArrayHandle<Scalar> h_sum_virial(m_sum_virial, access_location::host, access_mode::read);

    for (unsigned int i = 0; i<6; ++i)
        m_external_virial[i] = m_bias*h_sum_virial.data[i];

    if (m_prof) m_prof->pop(m_exec_conf);
    }

Scalar OrderParameterMeshGPU::computeCV()
    {
    if (m_prof) m_prof->push(m_exec_conf,"sum");

    ArrayHandle<cufftComplex> d_fourier_mesh(m_fourier_mesh, access_location::device, access_mode::read);
    ArrayHandle<cufftComplex> d_fourier_mesh_G(m_fourier_mesh_G, access_location::device, access_mode::read);
    ArrayHandle<Scalar> d_interpolation_f(m_interpolation_f, access_location::device, access_mode::read);

    ArrayHandle<Scalar> d_sum_partial(m_sum_partial, access_location::device, access_mode::overwrite);

    bool exclude_dc = true;
    #ifdef ENABLE_MPI
    if (m_pdata->getDomainDecomposition())
        {
        uint3 my_pos = m_pdata->getDomainDecomposition()->getGridPos();
        exclude_dc = !my_pos.x && !my_pos.y && !my_pos.z;
        }
    #endif

    gpu_compute_cv(m_n_inner_cells,
                   d_sum_partial.data,
                   m_sum.getDeviceFlags(),
                   d_fourier_mesh.data,
                   d_fourier_mesh_G.data,
                   m_block_size,
                   m_mesh_points,
                   d_interpolation_f.data,
                   m_mode_sq,
                   m_pdata->getNGlobal(),
                   exclude_dc);

    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();

    Scalar sum = m_sum.readFlags()*Scalar(1.0/2.0);

    #ifdef ENABLE_MPI
    if (m_pdata->getDomainDecomposition())
        {
        // reduce sum
        MPI_Allreduce(MPI_IN_PLACE,
                      &sum,
                      1,
                      MPI_HOOMD_SCALAR,
                      MPI_SUM,
                      m_exec_conf->getMPICommunicator());
        }
    #endif

    if (m_prof) m_prof->pop(m_exec_conf);

    return sum;
    }

void OrderParameterMeshGPU::computeQmax(unsigned int timestep)
    {
    // compute Fourier grid
    getCurrentValue(timestep);

    if (timestep && m_q_max_last_computed == timestep) return;

    if (m_prof) m_prof->push("max q");
    m_q_max_last_computed = timestep;

    ArrayHandle<cufftComplex> d_fourier_mesh(m_fourier_mesh, access_location::device, access_mode::read);
    ArrayHandle<Scalar3> d_k(m_k, access_location::device, access_mode::read);

    ArrayHandle<Scalar4> d_max_partial(m_max_partial, access_location::device, access_mode::overwrite);

    gpu_compute_q_max(m_n_inner_cells,
                      d_max_partial.data,
                      m_gpu_q_max.getDeviceFlags(),
                      d_k.data,
                      d_fourier_mesh.data,
                      m_block_size);

    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();

    Scalar4 q_max = m_gpu_q_max.readFlags();

    #ifdef ENABLE_MPI
    if (m_pdata->getDomainDecomposition())
        {
        // all processes send their results to all other processes
        // and then they determine the maximum wave vector
        Scalar4 *all_q_max = new Scalar4[m_exec_conf->getNRanks()];
        MPI_Allgather(&q_max,
                     sizeof(Scalar4),
                     MPI_BYTE,
                     all_q_max,
                     sizeof(Scalar4),
                     MPI_BYTE,
                     m_exec_conf->getMPICommunicator());

        for (unsigned int i = 0; i < m_exec_conf->getNRanks(); ++i)
            {
            if (all_q_max[i].w > q_max.w)
                {
                q_max = all_q_max[i];
                }
            }

        delete[] all_q_max;
        }
    #endif

    if (m_prof) m_prof->pop();

    m_q_max = make_scalar3(q_max.x,q_max.y,q_max.z);
    m_sq_max = q_max.w;

    // normalize with 1/V
    unsigned int n_global = m_pdata->getNGlobal();
    m_sq_max *= (Scalar)n_global;
    }


void export_OrderParameterMeshGPU(py::module& m)
    {
    py::class_<OrderParameterMeshGPU, std::shared_ptr<OrderParameterMeshGPU> >(m, "OrderParameterMeshGPU", py::base<OrderParameterMesh>())
        .def(py::init< std::shared_ptr<SystemDefinition>,
                         const unsigned int,
                         const unsigned int,
                         const unsigned int,
                         const std::vector<Scalar>,
                         const std::vector<int3>
                        >());
    }

#endif // ENABLE_CUDA
