#include "OrderParameterMeshGPU.h"
#include "OrderParameterMeshGPU.cuh"

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
    GPUArray<Scalar> sum_partial(m_mesh_points.x*m_mesh_points.y*m_mesh_points.z/m_block_size+1,m_exec_conf);
    m_sum_partial.swap(sum_partial);
    }

OrderParameterMeshGPU::~OrderParameterMeshGPU()
    {
    cufftDestroy(m_cufft_plan);
    }

void OrderParameterMeshGPU::initializeFFT()
    {
    cufftPlan3d(&m_cufft_plan, m_mesh_points.x, m_mesh_points.y, m_mesh_points.z, CUFFT_C2C);

    int dims[3] = {m_mesh_points.x, m_mesh_points.y, m_mesh_points.z};
    cufftPlanMany(&m_cufft_plan_force, 3, dims, NULL, 1, m_mesh_index.getNumElements(),
                  NULL, 1, m_mesh_index.getNumElements(), CUFFT_C2C, 3);

    // allocate mesh and transformed mesh
    unsigned int num_cells = m_mesh_index.getNumElements();

    GPUArray<cufftComplex> mesh(num_cells,m_exec_conf);
    m_mesh.swap(mesh);

    GPUArray<cufftComplex> fourier_mesh(num_cells, m_exec_conf);
    m_fourier_mesh.swap(fourier_mesh);

    GPUArray<cufftComplex> fourier_mesh_G(num_cells, m_exec_conf);
    m_fourier_mesh_G.swap(fourier_mesh_G);

    GPUArray<cufftComplex> fourier_mesh_force(3*num_cells, m_exec_conf);
    m_fourier_mesh_force.swap(fourier_mesh_force);

    GPUArray<cufftComplex> ifourier_mesh_force(3*num_cells, m_exec_conf);
    m_ifourier_mesh_force.swap(ifourier_mesh_force);

    GPUArray<Scalar4> force_mesh(num_cells, m_exec_conf);
    m_force_mesh.swap(force_mesh);
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

    cudaMemset(d_mesh.data, 0, sizeof(cufftComplex)*m_mesh.getNumElements());

    gpu_assign_particles(m_pdata->getN(),
                         d_postype.data,
                         d_mesh.data,
                         m_mesh_index,
                         d_mode.data,
                         m_pdata->getGlobalBox());

    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();

    if (m_prof) m_prof->pop(m_exec_conf);
    }

void OrderParameterMeshGPU::updateMeshes()
    {
    if (m_prof) m_prof->push(m_exec_conf,"FFT");

    ArrayHandle<cufftComplex> d_mesh(m_mesh, access_location::device, access_mode::read);
    ArrayHandle<cufftComplex> d_fourier_mesh(m_fourier_mesh, access_location::device, access_mode::overwrite);
    ArrayHandle<cufftComplex> d_fourier_mesh_G(m_fourier_mesh_G, access_location::device, access_mode::overwrite);

    ArrayHandle<cufftComplex> d_fourier_mesh_force(m_fourier_mesh_force, access_location::device, access_mode::overwrite);
    ArrayHandle<cufftComplex> d_ifourier_mesh_force(m_ifourier_mesh_force, access_location::device, access_mode::overwrite);

    ArrayHandle<Scalar> d_inf_f(m_inf_f, access_location::device, access_mode::read);
    ArrayHandle<Scalar3> d_k(m_k, access_location::device, access_mode::read);

    // transform the particle mesh
    cufftExecC2C(m_cufft_plan, d_mesh.data, d_fourier_mesh.data, CUFFT_FORWARD);

    Scalar V_cell = m_pdata->getGlobalBox().getVolume()/(Scalar)m_mesh_index.getNumElements();

    gpu_update_meshes(m_mesh_index.getNumElements(),
                      d_fourier_mesh.data,
                      d_fourier_mesh_G.data,
                      d_inf_f.data,
                      d_k.data,
                      V_cell,
                      d_fourier_mesh_force.data);

    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();

    cufftExecC2C(m_cufft_plan, d_fourier_mesh_force.data, d_ifourier_mesh_force.data, CUFFT_INVERSE);
 
    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();

    if (m_prof) m_prof->pop(m_exec_conf);
    }

void OrderParameterMeshGPU::interpolateForces()
    {
    if (m_prof) m_prof->push(m_exec_conf,"interpolate");

    ArrayHandle<Scalar4> d_postype(m_pdata->getPositions(), access_location::device, access_mode::read);
    ArrayHandle<cufftComplex> d_ifourier_mesh_force(m_ifourier_mesh_force, access_location::device, access_mode::read);
    ArrayHandle<Scalar4> d_force_mesh(m_force_mesh, access_location::device, access_mode::overwrite);
    ArrayHandle<Scalar> d_mode(m_mode, access_location::device, access_mode::read);

    ArrayHandle<Scalar4> d_force(m_force, access_location::device, access_mode::overwrite);

    gpu_interpolate_forces(m_pdata->getN(),
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

Scalar OrderParameterMeshGPU::computeCV()
    {
    if (m_prof) m_prof->push(m_exec_conf,"sum");

    ArrayHandle<cufftComplex> d_fourier_mesh(m_fourier_mesh, access_location::device, access_mode::read);
    ArrayHandle<cufftComplex> d_fourier_mesh_G(m_fourier_mesh_G, access_location::device, access_mode::read);

    ArrayHandle<Scalar> d_sum_partial(m_sum_partial, access_location::device, access_mode::overwrite);

    gpu_compute_cv(m_mesh_index.getNumElements(),
                   d_sum_partial.data,
                   m_sum.getDeviceFlags(),
                   d_fourier_mesh.data,
                   d_fourier_mesh_G.data,
                   m_block_size);
 
    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();

    Scalar sum = m_sum.readFlags() * Scalar(1.0/2.0)/m_pdata->getGlobalBox().getVolume();

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
