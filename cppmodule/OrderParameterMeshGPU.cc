#include "OrderParameterMeshGPU.h"

#ifdef ENABLE_CUDA
#include "OrderParameterMeshGPU.cuh"

using namespace boost::python;

/*! \param sysdef The system definition
    \param nx Number of cells along first axis
    \param ny Number of cells along second axis
    \param nz Number of cells along third axis
    \param qstar Short-wave length cutoff
    \param mode Per-type modes to multiply density
    \param zero_modes List of modes that should be zeroed
 */
OrderParameterMeshGPU::OrderParameterMeshGPU(boost::shared_ptr<SystemDefinition> sysdef,
                                            const unsigned int nx,
                                            const unsigned int ny,
                                            const unsigned int nz,
                                            const Scalar qstar,
                                            std::vector<Scalar> mode,
                                            std::vector<int3> zero_modes)
    : OrderParameterMesh(sysdef, nx, ny, nz, qstar, mode,zero_modes),
      m_local_fft(true),
      m_sum(m_exec_conf),
      m_block_size(256),
      m_gpu_q_max(m_exec_conf)
    {
    unsigned int n_blocks = m_mesh_points.x*m_mesh_points.y*m_mesh_points.z/m_block_size+1;
    GPUArray<Scalar> sum_partial(n_blocks,m_exec_conf);
    m_sum_partial.swap(sum_partial);

    GPUArray<Scalar> sum_virial_partial(6*n_blocks,m_exec_conf);
    m_sum_virial_partial.swap(sum_virial_partial);

    GPUArray<Scalar> sum_virial(6,m_exec_conf);
    m_sum_virial.swap(sum_virial);

    GPUArray<Scalar4> max_partial(n_blocks, m_exec_conf);
    m_max_partial.swap(max_partial);

    // initial value of number of particles per bin
    m_cell_size = 2;

    uchar3 periodic = m_pdata->getBox().getPeriodic();
    m_n_ghost_bins = make_uint3(periodic.x ? 0 : 2*m_radius,
                                periodic.y ? 0 : 2*m_radius,
                                periodic.z ? 0 : 2*m_radius);
    unsigned int n_particle_bins = (m_mesh_points.x+m_n_ghost_bins.x)
                                  *(m_mesh_points.y+m_n_ghost_bins.y)
                                  *(m_mesh_points.z+m_n_ghost_bins.z);
    m_bin_idx = Index2D(n_particle_bins,m_cell_size);
    m_scratch_idx = Index2D(m_mesh_index.getNumElements(),(2*m_radius+1)*(2*m_radius+1)*(2*m_radius+1));
    }

OrderParameterMeshGPU::~OrderParameterMeshGPU()
    {
    cufftDestroy(m_cufft_plan);
    }

void OrderParameterMeshGPU::initializeFFT()
    {
    #ifdef ENABLE_MPI
    m_local_fft = !m_pdata->getDomainDecomposition();

    if (! m_local_fft)
        {
        // ghost cell exchanger 
        m_gpu_mesh_comm = boost::shared_ptr<CommunicatorMeshGPUComplex >(
            new CommunicatorMeshGPUComplex(m_sysdef,
                                           m_comm, m_n_ghost_cells,
                                           make_uint3(m_force_mesh_index.getW(), m_force_mesh_index.getH(), m_force_mesh_index.getD()),
                                           true));

        // set up distributed FFT 
        int gdim[3];
        int pdim[3];
        Index3D decomp_idx = m_pdata->getDomainDecomposition()->getDomainIndexer();
        pdim[0] = decomp_idx.getW();
        pdim[1] = decomp_idx.getH();
        pdim[2] = decomp_idx.getD();
        gdim[0] = m_mesh_points.x*pdim[0];
        gdim[1] = m_mesh_points.y*pdim[1];
        gdim[2] = m_mesh_points.z*pdim[2];
        int embed[3];
        embed[0] = m_mesh_points.x+m_n_ghost_cells.x;
        embed[1] = m_mesh_points.y+m_n_ghost_cells.y;
        embed[2] = m_mesh_points.z+m_n_ghost_cells.z;
        m_ghost_offset = ((m_n_ghost_cells.x/2)*embed[1]+m_n_ghost_cells.y/2)*embed[2]+m_n_ghost_cells.z/2;
        uint3 pcoord = m_pdata->getDomainDecomposition()->getDomainIndexer().getTriple(m_exec_conf->getRank());
        int pidx[3];
        pidx[0] = pcoord.x;
        pidx[1] = pcoord.y;
        pidx[2] = pcoord.z;
        int row_m = 1; /* Hoomd uses row-major process-id mapping */
        #ifndef USE_HOST_DFFT
        dfft_cuda_create_plan(&m_dfft_plan_forward, 3, gdim, NULL, NULL, pdim, pidx, row_m, 0, 1, m_exec_conf->getMPICommunicator());
        dfft_cuda_create_plan(&m_dfft_plan_inverse, 3, gdim, NULL, embed, pdim, pidx, row_m, 0, 1, m_exec_conf->getMPICommunicator());
        #else
        dfft_create_plan(&m_dfft_plan_forward, 3, gdim, NULL, NULL, pdim, pidx, row_m, 0, 1, m_exec_conf->getMPICommunicator());
        dfft_create_plan(&m_dfft_plan_inverse, 3, gdim, NULL, embed, pdim, pidx, row_m, 0, 1, m_exec_conf->getMPICommunicator());
        #endif
        }
    #endif // ENABLE_MPI

    if (m_local_fft)
        {
        cufftPlan3d(&m_cufft_plan, m_mesh_points.x, m_mesh_points.y, m_mesh_points.z, CUFFT_C2C);
        }

    // allocate mesh and transformed mesh
    GPUArray<cufftComplex> mesh(m_n_inner_cells,m_exec_conf);
    m_mesh.swap(mesh);

    GPUArray<cufftComplex> fourier_mesh(m_n_inner_cells, m_exec_conf);
    m_fourier_mesh.swap(fourier_mesh);

    GPUArray<cufftComplex> fourier_mesh_G(m_n_inner_cells, m_exec_conf);
    m_fourier_mesh_G.swap(fourier_mesh_G);

    unsigned int num_cells = m_force_mesh_index.getNumElements();

    GPUArray<cufftComplex> inv_fourier_mesh(num_cells, m_exec_conf);
    m_inv_fourier_mesh.swap(inv_fourier_mesh);

    GPUArray<Scalar4> particle_bins(m_bin_idx.getNumElements(), m_exec_conf);
    m_particle_bins.swap(particle_bins);

    GPUArray<unsigned int> n_cell(m_bin_idx.getW(), m_exec_conf);
    m_n_cell.swap(n_cell);

    GPUFlags<unsigned int> cell_overflowed(m_exec_conf);
    m_cell_overflowed.swap(cell_overflowed);

    m_cell_overflowed.resetFlags(0);

    // allocate scratch space for density reduction
    GPUArray<Scalar> mesh_scratch(m_scratch_idx.getNumElements(), m_exec_conf);
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
            gpu_bin_particles(m_pdata->getN()+m_pdata->getNGhosts(),
                              d_postype.data,
                              d_particle_bins.data,
                              d_n_cell.data,
                              m_cell_overflowed.getDeviceFlags(),
                              m_bin_idx,
                              m_mesh_index,
                              m_n_ghost_bins,
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
            GPUArray<Scalar4> particle_bins(m_bin_idx.getNumElements(),m_exec_conf);
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
        
        gpu_assign_binned_particles_to_mesh(m_mesh_index,
                                            m_n_ghost_bins,
                                            d_particle_bins.data,
                                            d_mesh_scratch.data,
                                            m_bin_idx,
                                            m_scratch_idx,
                                            d_n_cell.data,
                                            d_mesh.data,
                                            m_local_fft);

        if (m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();
        }

    if (m_prof) m_prof->pop(m_exec_conf);
    }

void OrderParameterMeshGPU::updateMeshes()
    {

    if (m_prof) m_prof->push(m_exec_conf,"FFT");

    if (m_local_fft)
        {
        // locally transform the particle mesh
        ArrayHandle<cufftComplex> d_mesh(m_mesh, access_location::device, access_mode::read);
        ArrayHandle<cufftComplex> d_fourier_mesh(m_fourier_mesh, access_location::device, access_mode::overwrite);
 
        cufftExecC2C(m_cufft_plan, d_mesh.data, d_fourier_mesh.data, CUFFT_FORWARD);
        }
    #ifdef ENABLE_MPI
    else
        {
        // perform a distributed FFT
        m_exec_conf->msg->notice(8) << "cv.mesh: Distributed FFT mesh" << std::endl;
        #ifndef USE_HOST_DFFT
        ArrayHandle<cufftComplex> d_mesh(m_mesh, access_location::device, access_mode::read);
        ArrayHandle<cufftComplex> d_fourier_mesh(m_fourier_mesh, access_location::device, access_mode::overwrite);

        if (m_exec_conf->isCUDAErrorCheckingEnabled())
            dfft_cuda_check_errors(&m_dfft_plan_forward, 1);
        else
            dfft_cuda_check_errors(&m_dfft_plan_forward, 0);

        dfft_cuda_execute(d_mesh.data, d_fourier_mesh.data, 0, &m_dfft_plan_forward);
        #else
        ArrayHandle<cufftComplex> h_mesh(m_mesh, access_location::host, access_mode::read);
        ArrayHandle<cufftComplex> h_fourier_mesh(m_fourier_mesh, access_location::host, access_mode::overwrite);

        dfft_execute((cpx_t *)h_mesh.data, (cpx_t *)h_fourier_mesh.data, 0,m_dfft_plan_forward);
        #endif
        }
    #endif
    if (m_prof) m_prof->pop(m_exec_conf);

    if (m_prof) m_prof->push(m_exec_conf,"update");

        {
        ArrayHandle<cufftComplex> d_fourier_mesh(m_fourier_mesh, access_location::device, access_mode::readwrite);
        ArrayHandle<cufftComplex> d_fourier_mesh_G(m_fourier_mesh_G, access_location::device, access_mode::overwrite);

        ArrayHandle<Scalar> d_inf_f(m_inf_f, access_location::device, access_mode::read);
        ArrayHandle<Scalar3> d_k(m_k, access_location::device, access_mode::read);

        gpu_update_meshes(m_n_inner_cells,
                          d_fourier_mesh.data,
                          d_fourier_mesh_G.data,
                          d_inf_f.data,
                          d_k.data,
                          m_pdata->getNGlobal());

        if (m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();
        }

    if (m_prof) m_prof->pop(m_exec_conf);

    if (m_prof) m_prof->push(m_exec_conf, "FFT");

    if (m_local_fft)
        {
        // do local inverse transform of force mesh
        ArrayHandle<cufftComplex> d_fourier_mesh_G(m_fourier_mesh_G, access_location::device, access_mode::read);
        ArrayHandle<cufftComplex> d_inv_fourier_mesh(m_inv_fourier_mesh, access_location::device, access_mode::overwrite);

        cufftExecC2C(m_cufft_plan,
                     d_fourier_mesh_G.data,
                     d_inv_fourier_mesh.data,
                     CUFFT_INVERSE);
        }
    #ifdef ENABLE_MPI
    else
        {
        // Distributed inverse transform of force mesh
        m_exec_conf->msg->notice(8) << "cv.mesh: Distributed iFFT force mesh" << std::endl;
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
        dfft_execute((cpx_t *)h_fourier_mesh_G.data, (cpx_t *)(h_inv_fourier_mesh.data+m_ghost_offset), 1,m_dfft_plan_inverse);
        #endif
        }
    #endif

    if (m_prof) m_prof->pop(m_exec_conf);

    #ifdef ENABLE_MPI
    if (! m_local_fft)
        {
        // update outer cells of inverse Fourier mesh using ghost cells from neighboring processors
        if (m_prof) m_prof->push("ghost exchange");
        m_exec_conf->msg->notice(8) << "cv.mesh: Ghost cell update" << std::endl;
        m_gpu_mesh_comm->updateGhostCells(m_inv_fourier_mesh);
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
                       m_force_mesh_index,
                       m_n_ghost_cells,
                       d_mode.data,
                       m_pdata->getBox(),
                       m_pdata->getGlobalBox(),
                       m_local_fft,
                       m_pdata->getNGlobal());

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

    gpu_compute_mesh_virial(m_n_inner_cells,
                            d_fourier_mesh.data,
                            d_fourier_mesh_G.data,
                            d_virial_mesh.data,
                            d_k.data,
                            m_qstarsq,
                            m_exec_conf->getRank() == 0);

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

    ArrayHandle<Scalar> d_sum_partial(m_sum_partial, access_location::device, access_mode::overwrite);

    gpu_compute_cv(m_n_inner_cells,
                   d_sum_partial.data,
                   m_sum.getDeviceFlags(),
                   d_fourier_mesh.data,
                   d_fourier_mesh_G.data,
                   m_block_size,
                   m_mesh_index,
                   m_exec_conf->getRank() == 0);
 
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

//! Compute the optimal influence function
void OrderParameterMeshGPU::computeInfluenceFunction()
    {
    if (m_prof) m_prof->push(m_exec_conf, "influence function");

    ArrayHandle<Scalar> d_inf_f(m_inf_f, access_location::device, access_mode::overwrite);
    ArrayHandle<Scalar3> d_k(m_k, access_location::device, access_mode::overwrite);
    ArrayHandle<int3> d_zero_modes(m_zero_modes, access_location::device, access_mode::read);

    uint3 global_dim = m_mesh_points;
    uint3 pidx = make_uint3(0,0,0);
    uint3 pdim = make_uint3(0,0,0);
    #ifdef ENABLE_MPI
    if (m_pdata->getDomainDecomposition())
        {
        const Index3D &didx = m_pdata->getDomainDecomposition()->getDomainIndexer();
        global_dim.x *= didx.getW();
        global_dim.y *= didx.getH();
        global_dim.z *= didx.getD();
        pidx = didx.getTriple(m_exec_conf->getRank());
        pdim = make_uint3(didx.getW(), didx.getH(), didx.getD());
        }
    #endif

    gpu_compute_influence_function(m_mesh_index,
                                   global_dim,
                                   d_inf_f.data,
                                   d_k.data,
                                   m_pdata->getGlobalBox(),
                                   m_qstarsq,
                                   d_zero_modes.data,
                                   m_zero_modes.getNumElements(),
                                   m_local_fft,
                                   pidx,
                                   pdim);
  
    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();

    if (m_prof) m_prof->pop(m_exec_conf);
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
    }


void export_OrderParameterMeshGPU()
    {
    class_<OrderParameterMeshGPU, boost::shared_ptr<OrderParameterMeshGPU>, bases<OrderParameterMesh>, boost::noncopyable >
        ("OrderParameterMeshGPU", init< boost::shared_ptr<SystemDefinition>,
                                     const unsigned int,
                                     const unsigned int,
                                     const unsigned int,
                                     const Scalar,
                                     const std::vector<Scalar>,
                                     const std::vector<int3>
                                    >());
    }

#endif // ENABLE_CUDA
