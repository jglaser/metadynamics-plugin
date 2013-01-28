#include "OrderParameterMeshGPU.h"
#include "OrderParameterMeshGPU.cuh"

#include "DAFTGPU.h"

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
    : OrderParameterMesh(sysdef, nx, ny, nz, qstar, mode), m_sum(m_exec_conf), m_block_size(256)
    {
    GPUArray<Scalar> sum_partial(m_mesh_points.x*m_mesh_points.y*(m_mesh_points.z/2+1)/m_block_size+1,m_exec_conf);
    m_sum_partial.swap(sum_partial);

    GPUArray<Scalar> sum_virial_partial(6*m_mesh_points.x*m_mesh_points.y*(m_mesh_points.z/2+1)/m_block_size+1,m_exec_conf);
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
    cufftPlan3d(&m_cufft_plan, m_mesh_points.x, m_mesh_points.y, m_mesh_points.z, CUFFT_R2C);

    unsigned int num_cells = m_mesh_index.getNumElements();
    m_num_fourier_cells = m_mesh_points.x * m_mesh_points.y * (m_mesh_points.z/2+1);

    int dims[3] = {m_mesh_points.x, m_mesh_points.y, m_mesh_points.z};
    int fourier_dims[3] = {m_mesh_points.x, m_mesh_points.y, m_mesh_points.z/2+1};
    cufftPlanMany(&m_cufft_plan_force, 3, dims, fourier_dims, 1, m_num_fourier_cells,
                  dims, 1, num_cells, CUFFT_C2R, 3);

    // allocate mesh and transformed mesh

    GPUArray<cufftReal> mesh(num_cells,m_exec_conf);
    m_mesh.swap(mesh);


    GPUArray<cufftComplex> fourier_mesh(m_num_fourier_cells, m_exec_conf);
    m_fourier_mesh.swap(fourier_mesh);

    GPUArray<cufftComplex> fourier_mesh_G(m_num_fourier_cells, m_exec_conf);
    m_fourier_mesh_G.swap(fourier_mesh_G);

    GPUArray<cufftComplex> fourier_mesh_force(3*m_num_fourier_cells, m_exec_conf);
    m_fourier_mesh_force.swap(fourier_mesh_force);

    GPUArray<cufftReal> ifourier_mesh_force(3*num_cells, m_exec_conf);
    m_ifourier_mesh_force.swap(ifourier_mesh_force);

    GPUArray<Scalar4> force_mesh(num_cells, m_exec_conf);
    m_force_mesh.swap(force_mesh);

    if (exec_conf->getComputeCapability() < 300)
        {
        GPUArray<Scalar4> particle_bins(num_cells*m_cell_size, m_exec_conf);
        m_particle_bins.swap(particle_bins);

        GPUArray<unsigned int> n_cell(num_cells, m_exec_conf);
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
    ArrayHandle<cufftReal> d_mesh(m_mesh, access_location::device, access_mode::overwrite);
    ArrayHandle<Scalar> d_mode(m_mode, access_location::device, access_mode::read);

    if (exec_conf->getComputeCapability() >= 300)
        {
        // optimized for Kepler
        cudaMemset(d_mesh.data, 0, sizeof(cufftReal)*m_mesh.getNumElements());
        gpu_assign_particles_30(m_pdata->getN(),
                             d_postype.data,
                             d_mesh.data,
                             m_mesh_index,
                             d_mode.data,
                             m_pdata->getGlobalBox());

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
            cudaMemset(d_n_cell.data,0,sizeof(unsigned int)*m_mesh_index.getNumElements());

                {
                ArrayHandle<Scalar4> d_particle_bins(m_particle_bins, access_location::device, access_mode::overwrite);
                gpu_bin_particles(m_pdata->getN(),
                                  d_postype.data,
                                  d_particle_bins.data,
                                  d_n_cell.data,
                                  m_cell_overflowed.getDeviceFlags(),
                                  m_cell_size,
                                  m_mesh_index,
                                  d_mode.data,
                                  m_pdata->getGlobalBox());

                if (m_exec_conf->isCUDAErrorCheckingEnabled())
                    CHECK_CUDA_ERROR();
                }

            unsigned int flags = m_cell_overflowed.readFlags();
            
            if (flags)
                {
                // reallocate particle bins array
                m_cell_size = flags;

                GPUArray<Scalar4> particle_bins(m_mesh_index.getNumElements()*m_cell_size,m_exec_conf);
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
                                            d_particle_bins.data,     
                                            d_n_cell.data,
                                            m_cell_size,
                                            d_mesh.data,
                                            m_pdata->getGlobalBox());

        if (m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();
        }

    if (m_prof) m_prof->pop(m_exec_conf);
    }

void OrderParameterMeshGPU::updateMeshes()
    {
    if (m_prof) m_prof->push(m_exec_conf,"FFT");

    ArrayHandle<cufftReal> d_mesh(m_mesh, access_location::device, access_mode::read);
    ArrayHandle<cufftComplex> d_fourier_mesh(m_fourier_mesh, access_location::device, access_mode::overwrite);
    ArrayHandle<cufftComplex> d_fourier_mesh_G(m_fourier_mesh_G, access_location::device, access_mode::overwrite);

    ArrayHandle<cufftComplex> d_fourier_mesh_force(m_fourier_mesh_force, access_location::device, access_mode::overwrite);
    ArrayHandle<cufftReal> d_ifourier_mesh_force(m_ifourier_mesh_force, access_location::device, access_mode::overwrite);

    ArrayHandle<Scalar> d_inf_f(m_inf_f, access_location::device, access_mode::read);
    ArrayHandle<Scalar3> d_k(m_k, access_location::device, access_mode::read);

    // transform the particle mesh
    cufftExecR2C(m_cufft_plan, d_mesh.data, d_fourier_mesh.data);

    Scalar V_cell = m_pdata->getGlobalBox().getVolume()/(Scalar)m_mesh_index.getNumElements();

    gpu_update_meshes(m_num_fourier_cells,
                      d_fourier_mesh.data,
                      d_fourier_mesh_G.data,
                      d_inf_f.data,
                      d_k.data,
                      V_cell,
                      d_fourier_mesh_force.data);

    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();

    cufftExecC2R(m_cufft_plan_force, d_fourier_mesh_force.data, d_ifourier_mesh_force.data);
 
    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();

    if (m_prof) m_prof->pop(m_exec_conf);
    }

void OrderParameterMeshGPU::interpolateForces()
    {
    if (m_prof) m_prof->push(m_exec_conf,"interpolate");

    ArrayHandle<Scalar4> d_postype(m_pdata->getPositions(), access_location::device, access_mode::read);
    ArrayHandle<cufftReal> d_ifourier_mesh_force(m_ifourier_mesh_force, access_location::device, access_mode::read);
    ArrayHandle<Scalar4> d_force_mesh(m_force_mesh, access_location::device, access_mode::overwrite);
    ArrayHandle<Scalar> d_mode(m_mode, access_location::device, access_mode::read);

    ArrayHandle<Scalar4> d_force(m_force, access_location::device, access_mode::overwrite);

    gpu_interpolate_forces(m_pdata->getN(),
                           m_pdata->getNGlobal(),
                           d_postype.data,
                           d_force.data,
                           m_bias,
                           d_ifourier_mesh_force.data,
                           d_force_mesh.data,
                           m_mesh_index,
                           d_mode.data,
                           m_pdata->getGlobalBox());

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

    gpu_compute_mesh_virial(m_num_fourier_cells,
                            d_fourier_mesh.data,
                            d_fourier_mesh_G.data,
                            d_virial_mesh.data,
                            d_k.data,
                            m_qstarsq);

    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();

        {
        ArrayHandle<Scalar> d_sum_virial(m_sum_virial, access_location::device, access_mode::overwrite);
        ArrayHandle<Scalar> d_sum_virial_partial(m_sum_virial_partial, access_location::device, access_mode::overwrite);

        gpu_compute_virial(m_num_fourier_cells,
                           d_sum_virial_partial.data,
                           d_sum_virial.data,
                           d_virial_mesh.data,
                           m_block_size,
                           m_mesh_index);

        if (m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();
        }
     
    ArrayHandle<Scalar> h_sum_virial(m_sum_virial, access_location::host, access_mode::read);

    Scalar Nsq = m_pdata->getNGlobal();
    Nsq *= Nsq;

    Scalar V = m_pdata->getGlobalBox().getVolume();
    for (unsigned int i = 0; i<6; ++i)
        m_external_virial[i] = m_bias*Scalar(1.0/2.0)*h_sum_virial.data[i]/V/Nsq/Nsq;
      
    if (m_prof) m_prof->pop(m_exec_conf);
    }

Scalar OrderParameterMeshGPU::computeCV()
    {
    if (m_prof) m_prof->push(m_exec_conf,"sum");

    ArrayHandle<cufftComplex> d_fourier_mesh(m_fourier_mesh, access_location::device, access_mode::read);
    ArrayHandle<cufftComplex> d_fourier_mesh_G(m_fourier_mesh_G, access_location::device, access_mode::read);

    ArrayHandle<Scalar> d_sum_partial(m_sum_partial, access_location::device, access_mode::overwrite);

    gpu_compute_cv(m_num_fourier_cells,
                   d_sum_partial.data,
                   m_sum.getDeviceFlags(),
                   d_fourier_mesh.data,
                   d_fourier_mesh_G.data,
                   m_block_size,
                   m_mesh_index);
 
    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();

    Scalar sum = m_sum.readFlags()*Scalar(1.0/2.0) /m_pdata->getGlobalBox().getVolume();
    Scalar Nsq = m_pdata->getNGlobal();
    Nsq *= Nsq;
    sum /= Nsq*Nsq;

    if (m_prof) m_prof->pop(m_exec_conf);

    return sum;
    }

//! Compute the optimal influence function
void OrderParameterMeshGPU::computeInfluenceFunction()
    {
    if (m_prof) m_prof->push(m_exec_conf, "influence function");

    ArrayHandle<Scalar> d_inf_f(m_inf_f, access_location::device, access_mode::overwrite);
    ArrayHandle<Scalar3> d_k(m_k, access_location::device, access_mode::overwrite);

    gpu_compute_influence_function(m_mesh_index,
                                   m_pdata->getN(),
                                   d_inf_f.data,
                                   d_k.data,
                                   m_pdata->getGlobalBox(),
                                   m_qstarsq);
  
    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();

    if (m_prof) m_prof->pop(m_exec_conf);
    }

void OrderParameterMeshGPU::testFFT()
    {
    GPUArray<cufftComplex> in(1, m_exec_conf);
    GPUArray<cufftComplex> out(1, m_exec_conf);
        {
        ArrayHandle<cufftComplex> h_in(in, access_location::host, access_mode::overwrite);
        h_in.data[0].x = (Scalar)(m_exec_conf->getRank());
        h_in.data[0].y = 0.0;
        }

    DAFTGPU daft(m_exec_conf,m_pdata->getDomainDecomposition(), 1,1,1);
    daft.FFT3D(in, out, false);

        {
        ArrayHandle<cufftComplex> h_out(out, access_location::host, access_mode::read);
        std::cout << "Rank " << m_exec_conf->getRank() << " " << h_out.data[0].x << " " << h_out.data[0].y << std::endl;
        }

    daft.FFT3D(out,in ,true);
        {
        ArrayHandle<cufftComplex> h_in(in, access_location::host, access_mode::read);
        std::cout << "Rank " << m_exec_conf->getRank() << " " << h_in.data[0].x << " " << h_in.data[0].y << " (inverse)" << std::endl;
        } 
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
                                    >())
        .def("testFFT", &OrderParameterMeshGPU::testFFT);

    }
