#include "OrderParameterMeshGPU.h"
#include "OrderParameterMeshGPU.cuh"

#ifdef ENABLE_CUDA
using namespace boost::python;

/*! \param sysdef The system definition
    \param nx Number of cells along first axis
    \param ny Number of cells along second axis
    \param nz Number of cells along third axis
    \param qstar Short-wave length cutoff
    \param mode Per-type modes to multiply density
 */
OrderParameterMeshGPU::OrderParameterMeshGPU(boost::shared_ptr<SystemDefinition> sysdef,
                                            const unsigned int nx,
                                            const unsigned int ny,
                                            const unsigned int nz,
                                            const Scalar qstar,
                                            std::vector<Scalar> mode)
    : OrderParameterMesh(sysdef, nx, ny, nz, qstar, mode), m_local_fft(true), m_sum(m_exec_conf), m_block_size(256)
    {
    unsigned int n_blocks = m_mesh_points.x*m_mesh_points.y*m_mesh_points.z/m_block_size+1;
    GPUArray<Scalar> sum_partial(n_blocks,m_exec_conf);
    m_sum_partial.swap(sum_partial);

    GPUArray<Scalar> sum_virial_partial(6*n_blocks,m_exec_conf);
    m_sum_virial_partial.swap(sum_virial_partial);

    GPUArray<Scalar> sum_virial(6,m_exec_conf);
    m_sum_virial.swap(sum_virial);

    // initial value of number of particles per bin
    m_cell_size = 2;
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
        // ghost cell exchanger for forward direction
        m_gpu_mesh_comm_forward = boost::shared_ptr<CommunicatorMeshGPUComplex >(
            new CommunicatorMeshGPUComplex(m_sysdef, m_comm, m_n_ghost_cells, m_mesh_index, false));

        // ghost cell exchanger for reverse direction
        m_gpu_mesh_comm_inverse = boost::shared_ptr<CommunicatorMeshGPUScalar4 >(
            new CommunicatorMeshGPUScalar4(m_sysdef, m_comm, m_n_ghost_cells, m_mesh_index, true));

        // set up distributed FFT 
        m_gpu_dfft = boost::shared_ptr<DistributedFFTGPU>(
            new DistributedFFTGPU(m_exec_conf, m_pdata->getDomainDecomposition(), m_mesh_index, m_n_ghost_cells));
        m_gpu_dfft->setProfiler(m_prof);

        // set up inverse distributed FFT (batched)
        m_gpu_idfft = boost::shared_ptr<DistributedFFTGPU>(
            new DistributedFFTGPU(m_exec_conf, m_pdata->getDomainDecomposition(), m_mesh_index, m_n_ghost_cells,3));
        m_gpu_idfft->setProfiler(m_prof);

        }
    #endif // ENABLE_MPI

    if (m_local_fft)
        {
        cufftPlan3d(&m_cufft_plan, m_mesh_points.x, m_mesh_points.y, m_mesh_points.z, CUFFT_C2C);
        }
    unsigned int num_cells = m_mesh_index.getNumElements();

    // allocate mesh and transformed mesh
    GPUArray<cufftComplex> mesh(num_cells,m_exec_conf);
    m_mesh.swap(mesh);

    GPUArray<cufftComplex> fourier_mesh(m_n_inner_cells, m_exec_conf);
    m_fourier_mesh.swap(fourier_mesh);

    GPUArray<cufftComplex> fourier_mesh_G(m_n_inner_cells, m_exec_conf);
    m_fourier_mesh_G.swap(fourier_mesh_G);

    GPUArray<cufftComplex> fourier_mesh_force_xyz(m_n_inner_cells*3, m_exec_conf);
    m_fourier_mesh_force_xyz.swap(fourier_mesh_force_xyz);

    GPUArray<cufftComplex> force_mesh_xyz(num_cells*3, m_exec_conf);
    m_force_mesh_xyz.swap(force_mesh_xyz);

    GPUArray<Scalar4> force_mesh(num_cells, m_exec_conf);
    m_force_mesh.swap(force_mesh);

    if (exec_conf->getComputeCapability() < 300)
        {
        GPUArray<Scalar4> particle_bins(m_n_inner_cells*m_cell_size, m_exec_conf);
        m_particle_bins.swap(particle_bins);

        GPUArray<unsigned int> n_cell(m_n_inner_cells, m_exec_conf);
        m_n_cell.swap(n_cell);

        GPUFlags<unsigned int> cell_overflowed(m_exec_conf);
        m_cell_overflowed.swap(cell_overflowed);

        m_cell_overflowed.resetFlags(0);
        }
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

    // reset particle mesh
    cudaMemset(d_mesh.data, 0, sizeof(cufftComplex)*m_mesh.getNumElements());

    if (exec_conf->getComputeCapability() >= 300)
        {
        // optimized for Kepler
        gpu_assign_particles_30(m_pdata->getN(),
                             d_postype.data,
                             d_mesh.data,
                             m_mesh_index,
                             m_n_ghost_cells,
                             d_mode.data,
                             m_pdata->getBox(),
                             m_local_fft);

        if (m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();
        }
    else
        {
        // optimized for Fermi
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
                                  m_cell_size,
                                  m_mesh_index,
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

                GPUArray<Scalar4> particle_bins(m_n_inner_cells*m_cell_size,m_exec_conf);
                m_particle_bins.swap(particle_bins);
                m_cell_overflowed.resetFlags(0);
                }
            else
                {
                cont = false;
                }
            }

        // assign particles to mesh
        ArrayHandle<Scalar4> d_particle_bins(m_particle_bins, access_location::device, access_mode::read);
        
        gpu_assign_binned_particles_to_mesh(m_mesh_index,
                                            m_n_ghost_cells,
                                            d_particle_bins.data,     
                                            d_n_cell.data,
                                            m_cell_size,
                                            d_mesh.data,
                                            m_pdata->getBox(),
                                            m_local_fft);

        if (m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();
        }

    if (m_prof) m_prof->pop(m_exec_conf);
    }

void OrderParameterMeshGPU::updateMeshes()
    {

    #ifdef ENABLE_MPI
    if (m_pdata->getDomainDecomposition())
        {
        // update inner cells of mesh using ghost cells from neighboring processors
        if (m_prof) m_prof->push("ghost exchange");
        m_gpu_mesh_comm_forward->updateGhostCells(m_mesh);
        if (m_prof) m_prof->pop();
        }
    #endif

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
        m_gpu_dfft->FFT3D(m_mesh, m_fourier_mesh, false);
        }
    #endif


        {
        ArrayHandle<cufftComplex> d_fourier_mesh(m_fourier_mesh, access_location::device, access_mode::readwrite);
        ArrayHandle<cufftComplex> d_fourier_mesh_G(m_fourier_mesh_G, access_location::device, access_mode::overwrite);

        ArrayHandle<cufftComplex> d_fourier_mesh_force_xyz(m_fourier_mesh_force_xyz, access_location::device, access_mode::overwrite);

        ArrayHandle<Scalar> d_inf_f(m_inf_f, access_location::device, access_mode::read);
        ArrayHandle<Scalar3> d_k(m_k, access_location::device, access_mode::read);

        Scalar V_cell = m_pdata->getBox().getVolume()/(Scalar)m_n_inner_cells;

        gpu_update_meshes(m_n_inner_cells,
                          d_fourier_mesh.data,
                          d_fourier_mesh_G.data,
                          d_inf_f.data,
                          d_k.data,
                          V_cell,
                          m_pdata->getNGlobal(),
                          d_fourier_mesh_force_xyz.data);

        if (m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();
        }

    if (m_local_fft)
        {
        // do local inverse transform of force mesh
        ArrayHandle<cufftComplex> d_fourier_mesh_force_xyz(m_fourier_mesh_force_xyz, access_location::device, access_mode::read);
        ArrayHandle<cufftComplex> d_force_mesh_xyz(m_force_mesh_xyz, access_location::device, access_mode::overwrite);

        cufftExecC2C(m_cufft_plan,
                     d_fourier_mesh_force_xyz.data,
                     d_force_mesh_xyz.data,
                     CUFFT_INVERSE);

        cufftExecC2C(m_cufft_plan,
                     d_fourier_mesh_force_xyz.data+m_n_inner_cells,
                     d_force_mesh_xyz.data+m_n_inner_cells,
                     CUFFT_INVERSE);

        cufftExecC2C(m_cufft_plan,
                     d_fourier_mesh_force_xyz.data+m_n_inner_cells*2,
                     d_force_mesh_xyz.data+m_n_inner_cells*2,
                     CUFFT_INVERSE);
        }
    #ifdef ENABLE_MPI
    else
        {
        // Distributed inverse transform of force mesh
        m_gpu_idfft->FFT3D(m_fourier_mesh_force_xyz, m_force_mesh_xyz, true);
        }
    #endif

        {
        // coalesce forces
        ArrayHandle<cufftComplex> d_force_mesh_xyz(m_force_mesh_xyz, access_location::device, access_mode::read);
        ArrayHandle<Scalar4> d_force_mesh(m_force_mesh, access_location::device, access_mode::overwrite);
        
        gpu_coalesce_forces(m_mesh_index.getNumElements(),
                            d_force_mesh_xyz.data,
                            d_force_mesh.data);

        if (m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();
        }

    if (m_prof) m_prof->pop(m_exec_conf);

    #ifdef ENABLE_MPI
    if (! m_local_fft)
        {
        // update outer cells of force mesh using ghost cells from neighboring processors
        if (m_prof) m_prof->push("ghost exchange");
        m_gpu_mesh_comm_inverse->updateGhostCells(m_force_mesh);
        if (m_prof) m_prof->pop();
        }
    #endif

    }

void OrderParameterMeshGPU::interpolateForces()
    {
    if (m_prof) m_prof->push(m_exec_conf,"interpolate");

    ArrayHandle<Scalar4> d_postype(m_pdata->getPositions(), access_location::device, access_mode::read);
    ArrayHandle<Scalar4> d_force_mesh(m_force_mesh, access_location::device, access_mode::read);
    ArrayHandle<Scalar> d_mode(m_mode, access_location::device, access_mode::read);

    ArrayHandle<Scalar4> d_force(m_force, access_location::device, access_mode::overwrite);

    gpu_interpolate_forces(m_pdata->getN(),
                           d_postype.data,
                           d_force.data,
                           m_bias,
                           d_force_mesh.data,
                           m_mesh_index,
                           m_n_ghost_cells,
                           d_mode.data,
                           m_pdata->getBox(),
                           m_pdata->getGlobalBox(),
                           m_local_fft);

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
        m_external_virial[i] = m_bias*Scalar(1.0/2.0)*h_sum_virial.data[i];
      
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

    uint3 global_dim = m_mesh_points;
    #ifdef ENABLE_MPI
    DFFTIndex dffti;
    if (m_pdata->getDomainDecomposition())
        {
        const Index3D &didx = m_pdata->getDomainDecomposition()->getDomainIndexer();
        global_dim.x *= didx.getW();
        global_dim.y *= didx.getH();
        global_dim.z *= didx.getD();

        dffti = m_gpu_dfft->getIndexer();
        }
    #endif

    gpu_compute_influence_function(m_mesh_index,
                                   m_n_ghost_cells,
                                   global_dim,
                                   d_inf_f.data,
                                   d_k.data,
                                   m_pdata->getGlobalBox(),
                                   m_qstarsq,
    #ifdef ENABLE_MPI
                                   dffti,
    #endif
                                   m_local_fft);
  
    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();

    if (m_prof) m_prof->pop(m_exec_conf);
    }

void export_OrderParameterMeshGPU()
    {
    class_<OrderParameterMeshGPU, boost::shared_ptr<OrderParameterMeshGPU>, bases<OrderParameterMesh>, boost::noncopyable >
        ("OrderParameterMeshGPU", init< boost::shared_ptr<SystemDefinition>,
                                     const unsigned int,
                                     const unsigned int,
                                     const unsigned int,
                                     const Scalar,
                                     const std::vector<Scalar>&
                                    >());
    }

#endif // ENABLE_CUDA
