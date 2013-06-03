#include "OrderParameterMesh.h"

using namespace boost::python;
//! Coefficients of f(x) = sin(x)/x = a_0 + a_2 * x^2 + a_4 * x^4 + a_6 * x^6 + a_8 * x^8 + a_10 * x^10
const Scalar coeff[] = {Scalar(1.0), Scalar(-1.0/6.0), Scalar(1.0/120.0), Scalar(-1.0/5040.0),
                        Scalar(1.0/362880.0), Scalar(-1.0/39916800.0)};

/*! \param sysdef The system definition
    \param nx Number of cells along first axis
    \param ny Number of cells along second axis
    \param nz Number of cells along third axis
    \param qstar Short-wave length cutoff
    \param mode Per-type modes to multiply density
 */
OrderParameterMesh::OrderParameterMesh(boost::shared_ptr<SystemDefinition> sysdef,
                                            const unsigned int nx,
                                            const unsigned int ny,
                                            const unsigned int nz,
                                            const Scalar qstar,
                                            std::vector<Scalar> mode,
                                            std::vector<int3> zero_modes)
    : CollectiveVariable(sysdef, "mesh"),
      m_n_ghost_cells(make_uint3(0,0,0)),
      m_radius(1),
      m_n_inner_cells(0),
      m_is_first_step(true),
      m_cv_last_updated(0),
      m_box_changed(false),
      m_cv(Scalar(0.0)),
      m_q_max_last_computed(0),
      m_kiss_fft_initialized(false),
      m_ghost_layer_width(0.0)
    {

    if (mode.size() != m_pdata->getNTypes())
        {
        m_exec_conf->msg->error() << "Number of modes unequal number of particle types."
                                  << std::endl << std::endl;
        throw std::runtime_error("Error setting up cv.mesh");
        }

    GPUArray<Scalar> mode_array(m_pdata->getNTypes(), m_exec_conf);
    m_mode.swap(mode_array);

    ArrayHandle<Scalar> h_mode(m_mode, access_location::host, access_mode::overwrite);
    std::copy(mode.begin(), mode.end(), h_mode.data);

    GPUArray<int3> zero_modes_array(zero_modes.size(), m_exec_conf);
    m_zero_modes.swap(zero_modes_array);

    ArrayHandle<int3> h_zero_modes(m_zero_modes, access_location::host, access_mode::overwrite);
    std::copy(zero_modes.begin(), zero_modes.end(), h_zero_modes.data);

    m_qstarsq = qstar*qstar;
    m_boxchange_connection = m_pdata->connectBoxChange(boost::bind(&OrderParameterMesh::setBoxChange, this));

    m_mesh_points = make_uint3(nx, ny, nz);

    #ifdef ENABLE_MPI
    if (m_pdata->getDomainDecomposition())
        {
        const Index3D& didx = m_pdata->getDomainDecomposition()->getDomainIndexer();

        if (nx % didx.getW())
            {
            m_exec_conf->msg->error()
                << "The number of mesh points along the x-direction ("<< nx <<") is not" << std::endl
                << "a multiple of the width (" << didx.getW() << ") of the processsor grid!" << std::endl
                << std::endl;
            throw std::runtime_error("Error initializing cv.mesh");
            }
        if (ny % didx.getH())
            {
            m_exec_conf->msg->error()
                << "The number of mesh points along the y-direction ("<< ny <<") is not" << std::endl 
                << "a multiple of the height (" << didx.getH() << ") of the processsor grid!" << std::endl
                << std::endl;
            throw std::runtime_error("Error initializing cv.mesh");
            }
        if (nz % didx.getD())
            {
            m_exec_conf->msg->error()
                << "The number of mesh points along the z-direction ("<< nz <<") is not" << std::endl
                << "a multiple of the depth (" << didx.getD() << ") of the processsor grid!" << std::endl
                << std::endl;
            throw std::runtime_error("Error initializing cv.mesh");
            }

        m_mesh_points.x /= didx.getW();
        m_mesh_points.y /= didx.getH();
        m_mesh_points.z /= didx.getD();
        }
    #endif // ENABLE_MPI

    // reset virial
    ArrayHandle<Scalar> h_virial(m_virial, access_location::host, access_mode::overwrite);
    memset(h_virial.data, 0, sizeof(Scalar)*m_virial.getNumElements());

    m_bias = Scalar(1.0);

    m_log_names.push_back("cv_mesh");
    m_log_names.push_back("qx_max");
    m_log_names.push_back("qy_max");
    m_log_names.push_back("qz_max");

    // we need to compute the particle ghost layer before the first force calculation
    computeParticleGhostLayerWidth();
    }

OrderParameterMesh::~OrderParameterMesh()
    {
    if (m_kiss_fft_initialized)
        {
        free(m_kiss_fft);
        free(m_kiss_ifft_x);
        free(m_kiss_ifft_y);
        free(m_kiss_ifft_z);
        kiss_fft_cleanup();
        }
    m_boxchange_connection.disconnect();
    if (m_ghost_layer_connection.connected())
        m_ghost_layer_connection.disconnect();
    }

void OrderParameterMesh::setupMesh()
    {
    m_mesh_index = Index3D(m_mesh_points.x,
                           m_mesh_points.y,
                           m_mesh_points.z);
    m_force_mesh_index = Index3D(m_mesh_points.x+m_n_ghost_cells.x,
                           m_mesh_points.y+m_n_ghost_cells.y,
                           m_mesh_points.z+m_n_ghost_cells.z);
 
    m_n_inner_cells = m_mesh_points.x * m_mesh_points.y * m_mesh_points.z;

    // allocate memory for influence function and k values
    GPUArray<Scalar> inf_f(m_n_inner_cells, m_exec_conf);
    m_inf_f.swap(inf_f);

    GPUArray<Scalar3> k(m_n_inner_cells, m_exec_conf);
    m_k.swap(k);

    GPUArray<Scalar> virial_mesh(6*m_n_inner_cells, m_exec_conf);
    m_virial_mesh.swap(virial_mesh);

    initializeFFT();
    } 

uint3 OrderParameterMesh::computeNumGhostCells()
    {
    uint3 n_ghost_cells = make_uint3(0,0,0);
    #ifdef ENABLE_MPI
    if (m_pdata->getDomainDecomposition())
        {
        // particles can move a max distance of r_buff/2 outside the box between particle migration
        Scalar d_max = m_comm->getRBuff()/Scalar(2.0);

        const BoxDim& box = m_pdata->getBox();

        // maximum fractional distance
        Scalar3 d_max_frac = d_max/box.getNearestPlaneDistance();

        uchar3 periodic = box.getPeriodic();
        n_ghost_cells = make_uint3(periodic.x ? 0 : 2*m_radius,
                                   periodic.y ? 0 : 2*m_radius,
                                   periodic.z ? 0 : 2*m_radius);

        // the ghost layer must have a width of the maximum distance in addition
        // to one radius of ghost cells
        if (!periodic.x) n_ghost_cells.x += 2*(d_max_frac.x*m_mesh_points.x+1);
        if (!periodic.y) n_ghost_cells.y += 2*(d_max_frac.y*m_mesh_points.y+1);
        if (!periodic.z) n_ghost_cells.z += 2*(d_max_frac.z*m_mesh_points.z+1);
        }
    #endif

    return n_ghost_cells;
    }

void OrderParameterMesh::computeParticleGhostLayerWidth()
    {
    const BoxDim& box = m_pdata->getBox();

    // The width of the ghost layer for particles is the width of one cell times radius
    Scalar3 ghost_width = box.getNearestPlaneDistance()*(Scalar)m_radius;
    ghost_width = ghost_width / make_scalar3(m_mesh_points.x,m_mesh_points.y,m_mesh_points.z);

    // ignore periodic directions
    uchar3 periodic = box.getPeriodic();

    if (periodic.x) ghost_width.x = Scalar(0.0);
    if (periodic.y) ghost_width.y = Scalar(0.0);
    if (periodic.z) ghost_width.z = Scalar(0.0);

    // take max of all directions
    m_ghost_layer_width = ghost_width.x;
    m_ghost_layer_width = m_ghost_layer_width > ghost_width.y ? m_ghost_layer_width : ghost_width.y;
    m_ghost_layer_width = m_ghost_layer_width > ghost_width.z ? m_ghost_layer_width : ghost_width.z;
    }
 
void OrderParameterMesh::initializeFFT()
    {
    bool local_fft = true;

    #ifdef ENABLE_MPI
    local_fft = !m_pdata->getDomainDecomposition();

    if (! local_fft)
        {
        // ghost cell exchanger for reverse direction
        m_mesh_comm = boost::shared_ptr<CommunicatorMesh<kiss_fft_cpx> >(
            new CommunicatorMesh<kiss_fft_cpx>(m_sysdef, m_comm, m_n_ghost_cells, m_force_mesh_index, true));

        // set up distributed FFTs
        m_kiss_dfft = boost::shared_ptr<DistributedKISSFFT>(
            new DistributedKISSFFT(m_exec_conf, m_pdata->getDomainDecomposition(), m_mesh_index, make_uint3(0,0,0)));
        m_kiss_dfft->setProfiler(m_prof);

        m_kiss_idfft = boost::shared_ptr<DistributedKISSFFT>(
            new DistributedKISSFFT(m_exec_conf, m_pdata->getDomainDecomposition(), m_force_mesh_index, m_n_ghost_cells));
        m_kiss_idfft->setProfiler(m_prof);
        }
    #endif // ENABLE_MPI

    if (local_fft)
        {
        int dims[3];
        dims[0] = m_mesh_points.z;
        dims[1] = m_mesh_points.y;
        dims[2] = m_mesh_points.x;

        m_kiss_fft = kiss_fftnd_alloc(dims, 3, 0, NULL, NULL);
        m_kiss_ifft_x = kiss_fftnd_alloc(dims, 3, 1, NULL, NULL);
        m_kiss_ifft_y = kiss_fftnd_alloc(dims, 3, 1, NULL, NULL);
        m_kiss_ifft_z = kiss_fftnd_alloc(dims, 3, 1, NULL, NULL);

        m_kiss_fft_initialized = true;
        }

    // allocate mesh and transformed mesh
    GPUArray<kiss_fft_cpx> mesh(m_n_inner_cells,m_exec_conf);
    m_mesh.swap(mesh);

    GPUArray<kiss_fft_cpx> fourier_mesh(m_n_inner_cells, m_exec_conf);
    m_fourier_mesh.swap(fourier_mesh);

    GPUArray<kiss_fft_cpx> fourier_mesh_G(m_n_inner_cells, m_exec_conf);
    m_fourier_mesh_G.swap(fourier_mesh_G);

    unsigned int num_cells = m_force_mesh_index.getNumElements();

    GPUArray<kiss_fft_cpx> inv_fourier_mesh(num_cells, m_exec_conf);
    m_inv_fourier_mesh.swap(inv_fourier_mesh);
    }

void OrderParameterMesh::computeInfluenceFunction()
    {
    if (m_prof) m_prof->push("influence function");

    ArrayHandle<Scalar> h_inf_f(m_inf_f,access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar3> h_k(m_k,access_location::host, access_mode::overwrite);

    // reset arrays
    memset(h_inf_f.data, 0, sizeof(Scalar)*m_inf_f.getNumElements());
    memset(h_k.data, 0, sizeof(Scalar3)*m_k.getNumElements());

    const BoxDim& global_box = m_pdata->getGlobalBox();

    // compute reciprocal lattice vectors
    Scalar3 a1 = global_box.getLatticeVector(0);
    Scalar3 a2 = global_box.getLatticeVector(1);
    Scalar3 a3 = global_box.getLatticeVector(2);

    Scalar V_box = global_box.getVolume();
    Scalar3 b1 = Scalar(2.0*M_PI)*make_scalar3(a2.y*a3.z-a2.z*a3.y, a2.z*a3.x-a2.x*a3.z, a2.x*a3.y-a2.y*a3.x)/V_box;
    Scalar3 b2 = Scalar(2.0*M_PI)*make_scalar3(a3.y*a1.z-a3.z*a1.y, a3.z*a1.x-a3.x*a1.z, a3.x*a1.y-a3.y*a1.x)/V_box;
    Scalar3 b3 = Scalar(2.0*M_PI)*make_scalar3(a1.y*a2.z-a1.z*a2.y, a1.z*a2.x-a1.x*a2.z, a1.x*a2.y-a1.y*a2.x)/V_box;

    #ifdef ENABLE_MPI
    DFFTIndex dfft_idx;
    if (m_pdata->getDomainDecomposition())
        dfft_idx = m_kiss_dfft->getIndexer();
    #endif
    bool local_fft = m_kiss_fft_initialized;

    uint3 global_dim = m_mesh_points;
    #ifdef ENABLE_MPI
    if (m_pdata->getDomainDecomposition())
        {
        const Index3D &didx = m_pdata->getDomainDecomposition()->getDomainIndexer();
        global_dim.x *= didx.getW();
        global_dim.y *= didx.getH();
        global_dim.z *= didx.getD();
        }
    #endif

    for (unsigned int cell_idx = 0; cell_idx < m_n_inner_cells; ++cell_idx)
        {
        uint3 wave_idx;
        #ifdef ENABLE_MPI
        if (! local_fft)
            wave_idx = dfft_idx(cell_idx);
        else
        #endif
            {
            // kiss FFT expects data in row major format
            wave_idx.z = cell_idx / (m_mesh_points.y * m_mesh_points.x);
            wave_idx.y = (cell_idx - wave_idx.z * m_mesh_points.x * m_mesh_points.y)/ m_mesh_points.x;
            wave_idx.x = cell_idx % m_mesh_points.x;
            }

        int3 n = make_int3(wave_idx.x,wave_idx.y,wave_idx.z);

        // compute Miller indices
        if (n.x >= (int)(global_dim.x/2 + global_dim.x%2))
            n.x -= (int) global_dim.x;
        if (n.y >= (int)(global_dim.y/2 + global_dim.y%2))
            n.y -= (int) global_dim.y;
        if (n.z >= (int)(global_dim.z/2 + global_dim.z%2))
            n.z -= (int) global_dim.z;
        
        Scalar3 k = (Scalar)n.x*b1+(Scalar)n.y*b2+(Scalar)n.z*b3;
        Scalar ksq = dot(k,k);

        Scalar val = exp(-ksq/m_qstarsq*Scalar(1.0/2.0));

        h_inf_f.data[cell_idx] = val;

        h_k.data[cell_idx] = k;
        }

    if (m_prof) m_prof->pop();
    }
                             

/*! \param x Distance on mesh in units of the mesh size
 */
Scalar OrderParameterMesh::assignTSC(Scalar x)
    {
    Scalar xsq = x*x;
    Scalar xabs = sqrt(xsq);

    if (xsq <= Scalar(1.0/4.0))
        return Scalar(3.0/4.0) - xsq;
    else if (xsq <= Scalar(9.0/4.0))
        return Scalar(1.0/2.0)*(Scalar(3.0/2.0)-xabs)*(Scalar(3.0/2.0)-xabs);
    else
        return Scalar(0.0);
    }

Scalar OrderParameterMesh::assignTSCderiv(Scalar x)
    {
    Scalar xsq = x*x;
    Scalar xabs = copysignf(x,Scalar(1.0));
    Scalar fac =(Scalar(3.0/2.0)-xabs);

    Scalar ret(0.0);
    if (xsq <= Scalar(1.0/4.0))
        ret = -Scalar(2.0)*x;
    else if (xsq <= Scalar(9.0/4.0))
        ret = -fac*x/xabs;

    return ret;
    }

//! Assignment of particles to mesh using three-point scheme (triangular shaped cloud)
/*! This is a second order accurate scheme with continuous value and continuous derivative
 */
void OrderParameterMesh::assignParticles()
    {
    if (m_prof) m_prof->push("assign");

    ArrayHandle<Scalar4> h_postype(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<kiss_fft_cpx> h_mesh(m_mesh, access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar> h_mode(m_mode, access_location::host, access_mode::read);

    const BoxDim& box = m_pdata->getBox();

    // set mesh to zero
    memset(h_mesh.data, 0, sizeof(kiss_fft_cpx)*m_mesh.getNumElements());
 
    // inverse dimensions
    Scalar3 dim_inv = make_scalar3(Scalar(1.0)/(Scalar)m_mesh_points.x,
                                   Scalar(1.0)/(Scalar)m_mesh_points.y,
                                   Scalar(1.0)/(Scalar)m_mesh_points.z); 

    bool local_fft = m_kiss_fft_initialized;

    unsigned int ntot = m_pdata->getN() + m_pdata->getNGhosts();

    for (unsigned int idx = 0; idx < ntot; ++idx)
        {
        Scalar4 postype = h_postype.data[idx];

        Scalar3 pos = make_scalar3(postype.x, postype.y, postype.z);
        unsigned int type = __scalar_as_int(postype.w);

        // compute coordinates in units of the mesh size
        Scalar3 f = box.makeFraction(pos);
        Scalar3 reduced_pos = make_scalar3(f.x * (Scalar) m_mesh_points.x,
                                           f.y * (Scalar) m_mesh_points.y,
                                           f.z * (Scalar) m_mesh_points.z);

        // find cell the particle is in
        int ix = ((reduced_pos.x >= 0) ? reduced_pos.x : (reduced_pos.x - Scalar(1.0)));
        int iy = ((reduced_pos.y >= 0) ? reduced_pos.y : (reduced_pos.y - Scalar(1.0)));
        int iz = ((reduced_pos.z >= 0) ? reduced_pos.z : (reduced_pos.z - Scalar(1.0)));

        // handle particles on the boundary
        if (ix == (int)m_mesh_points.x && !m_n_ghost_cells.x)
            ix = 0;
        if (iy == (int)m_mesh_points.y && !m_n_ghost_cells.y)
            iy = 0;
        if (iz == (int)m_mesh_points.z && !m_n_ghost_cells.z)
            iz = 0;

        // compute distance between particle and cell center
        // in fractional coordinates 
        Scalar3 cell_center = make_scalar3((Scalar)ix + Scalar(0.5),
                                           (Scalar)iy + Scalar(0.5),
                                           (Scalar)iz + Scalar(0.5));
        Scalar3 shift = reduced_pos - cell_center;

        // assign particle to cell and next neighbors
        for (int i = -1; i <= 1 ; ++i)
            for (int j = -1; j <= 1; ++j)
                for (int k = -1; k <= 1; ++k)
                    {
                    int neighi = (int)ix + i;
                    int neighj = (int)iy + j;
                    int neighk = (int)iz + k;

                    if (! m_n_ghost_cells.x)
                        {
                        if (neighi == (int) m_mesh_points.x)
                            neighi = 0;
                        else if (neighi < 0)
                            neighi += m_mesh_points.x;
                        }
                    else if (neighi < 0 || neighi >= (int) m_mesh_points.x) continue;

                    if (! m_n_ghost_cells.y)
                        {
                        if (neighj == (int) m_mesh_points.y)
                            neighj = 0;
                        else if (neighj < 0)
                            neighj += m_mesh_points.y;
                        }
                    else if (neighj < 0 || neighj >= (int) m_mesh_points.y) continue;


                    if (! m_n_ghost_cells.z)
                        {
                        if (neighk == (int) m_mesh_points.z)
                            neighk = 0;
                        else if (neighk < 0)
                            neighk += m_mesh_points.z;
                        }
                    else if (neighk < 0 || neighk >= (int) m_mesh_points.z) continue;

                    Scalar3 dx_frac = shift - make_scalar3(i,j,k);

                    // compute fraction of particle density assigned to cell
                    Scalar density_fraction = assignTSC(dx_frac.x)*assignTSC(dx_frac.y)*assignTSC(dx_frac.z);
                    unsigned int neigh_idx;
                    if (local_fft)
                        // store in row major order for kiss FFT
                        neigh_idx = neighi + m_mesh_points.x * (neighj + m_mesh_points.y*neighk);
                    else
                        neigh_idx = m_mesh_index(neighi,neighj,neighk);

                    h_mesh.data[neigh_idx].r += h_mode.data[type]*density_fraction;
                    }
                 
        }  // end of loop over particles

    if (m_prof) m_prof->pop();
    }

void OrderParameterMesh::updateMeshes()
    {
    if (m_prof) m_prof->push("FFT");

    ArrayHandle<Scalar> h_inf_f(m_inf_f, access_location::host, access_mode::read);
    ArrayHandle<Scalar3> h_k(m_k, access_location::host, access_mode::read);

    if (m_kiss_fft_initialized)
        {
        // transform the particle mesh locally (forward transform)
        ArrayHandle<kiss_fft_cpx> h_mesh(m_mesh, access_location::host, access_mode::read);
        ArrayHandle<kiss_fft_cpx> h_fourier_mesh(m_fourier_mesh, access_location::host, access_mode::overwrite);

        kiss_fftnd(m_kiss_fft, h_mesh.data, h_fourier_mesh.data);
        }

    #ifdef ENABLE_MPI
    if (m_pdata->getDomainDecomposition())
        {
        // perform a distributed FFT
        m_exec_conf->msg->notice(8) << "cv.mesh: Distributed FFT mesh" << std::endl;
        m_kiss_dfft->FFT3D(m_mesh, m_fourier_mesh, false);
        }
    #endif

    ArrayHandle<kiss_fft_cpx> h_fourier_mesh(m_fourier_mesh, access_location::host, access_mode::readwrite);
    ArrayHandle<kiss_fft_cpx> h_fourier_mesh_G(m_fourier_mesh_G, access_location::host, access_mode::overwrite);
 
    unsigned int N_global = m_pdata->getNGlobal();

        {
        // multiply with influence function
        for (unsigned int k = 0; k < m_n_inner_cells; ++k)
            {
            kiss_fft_cpx f = h_fourier_mesh.data[k];

            // normalization
            f.r /= (Scalar) N_global;
            f.i /= (Scalar) N_global;

            Scalar val = f.r*f.r+f.i*f.i;

            h_fourier_mesh_G.data[k].r = f.r * val * h_inf_f.data[k];
            h_fourier_mesh_G.data[k].i = f.i * val * h_inf_f.data[k];

            h_fourier_mesh.data[k] = f;
            }
        }

    if (m_kiss_fft_initialized)
        {
        // do a local inverse transform of the force mesh
        ArrayHandle<kiss_fft_cpx> h_inv_fourier_mesh(m_inv_fourier_mesh, access_location::host, access_mode::overwrite);
        ArrayHandle<kiss_fft_cpx> h_fourier_mesh_G(m_fourier_mesh_G, access_location::host, access_mode::read);
        kiss_fftnd(m_kiss_ifft_x, h_inv_fourier_mesh.data, h_fourier_mesh_G.data);
        }

    #ifdef ENABLE_MPI
    if (m_pdata->getDomainDecomposition())
        {
        // Distributed inverse transform force on mesh points 
        m_exec_conf->msg->notice(8) << "cv.mesh: Distributed iFFT" << std::endl;
        m_kiss_idfft->FFT3D(m_fourier_mesh_G, m_inv_fourier_mesh, true);
        }
    #endif

    if (m_prof) m_prof->pop();

    #ifdef ENABLE_MPI
    if (m_pdata->getDomainDecomposition())
        {
        // update outer cells of force mesh using ghost cells from neighboring processors
        if (m_prof) m_prof->push("ghost exchange");
        m_exec_conf->msg->notice(8) << "cv.mesh: Ghost cell update" << std::endl;
        m_mesh_comm->updateGhostCells(m_inv_fourier_mesh);
        if (m_prof) m_prof->pop();
        }
    #endif
    }

void OrderParameterMesh::interpolateForces()
    {
    if (m_prof) m_prof->push("interpolate");

    ArrayHandle<Scalar4> h_postype(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<kiss_fft_cpx> h_inv_fourier_mesh(m_inv_fourier_mesh, access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_mode(m_mode, access_location::host, access_mode::read);

    ArrayHandle<Scalar4> h_force(m_force, access_location::host, access_mode::overwrite);

    const BoxDim& box = m_pdata->getBox();

    Scalar3 a1 = box.getLatticeVector(0);
    Scalar3 a2 = box.getLatticeVector(1);
    Scalar3 a3 = box.getLatticeVector(2);

    // reciprocal lattice vectors
    Scalar V_box = box.getVolume();
    Scalar3 b1 = make_scalar3(a2.y*a3.z-a2.z*a3.y, a2.z*a3.x-a2.x*a3.z, a2.x*a3.y-a2.y*a3.x)/V_box;
    Scalar3 b2 = make_scalar3(a3.y*a1.z-a3.z*a1.y, a3.z*a1.x-a3.x*a1.z, a3.x*a1.y-a3.y*a1.x)/V_box;
    Scalar3 b3 = make_scalar3(a1.y*a2.z-a1.z*a2.y, a1.z*a2.x-a1.x*a2.z, a1.x*a2.y-a1.y*a2.x)/V_box;

    // particle number
    bool local_fft = m_kiss_fft_initialized;

    unsigned int n_global = m_pdata->getNGlobal();

    for (unsigned int idx = 0; idx < m_pdata->getN(); ++idx)
        {
        Scalar4 postype = h_postype.data[idx];

        Scalar3 pos = make_scalar3(postype.x, postype.y, postype.z);
        unsigned int type = __scalar_as_int(postype.w);
        Scalar mode = h_mode.data[type];

        // compute coordinates in units of the mesh size
        Scalar3 f = box.makeFraction(pos);
        Scalar3 reduced_pos = make_scalar3(f.x * (Scalar) m_mesh_points.x,
                                           f.y * (Scalar) m_mesh_points.y,
                                           f.z * (Scalar) m_mesh_points.z);

        // find cell of the force mesh the particle is in
        unsigned int ix = (reduced_pos.x + (Scalar)(m_n_ghost_cells.x/2));
        unsigned int iy = (reduced_pos.y + (Scalar)(m_n_ghost_cells.y/2));
        unsigned int iz = (reduced_pos.z + (Scalar)(m_n_ghost_cells.z/2));

        // handle particles on the boundary
        if (ix == m_mesh_points.x && !m_n_ghost_cells.x)
            ix = 0;
        if (iy == m_mesh_points.y && !m_n_ghost_cells.y)
            iy = 0;
        if (iz == m_mesh_points.z && !m_n_ghost_cells.z)
            iz = 0;

        // center of cell (in units of the mesh size)
        Scalar3 cell_center = make_scalar3((Scalar)ix - (Scalar)(m_n_ghost_cells.x/2) + Scalar(0.5),
                                           (Scalar)iy - (Scalar)(m_n_ghost_cells.y/2) + Scalar(0.5),
                                           (Scalar)iz - (Scalar)(m_n_ghost_cells.z/2) + Scalar(0.5));
        Scalar3 shift = reduced_pos - cell_center;

        Scalar3 force = make_scalar3(0.0,0.0,0.0);

        for (int i = -1; i <= 1 ; ++i)
            for (int j = -1; j <= 1; ++j)
                for (int k = -1; k <= 1; ++k)
                    {
                    int neighi = (int)ix + i;
                    int neighj = (int)iy + j;
                    int neighk = (int)iz + k;

                    if (! m_n_ghost_cells.x)
                        {
                        if (neighi == (int)m_mesh_points.x)
                            neighi = 0;
                        else if (neighi < 0)
                            neighi += m_mesh_points.x;
                        }

                    if (! m_n_ghost_cells.y)
                        {
                        if (neighj == (int)m_mesh_points.y)
                            neighj = 0;
                        else if (neighj < 0)
                            neighj += m_mesh_points.y;
                        }


                    if (! m_n_ghost_cells.z)
                        {
                        if (neighk == (int)m_mesh_points.z)
                            neighk = 0;
                        else if (neighk < 0)
                            neighk += m_mesh_points.z;
                        }

                    Scalar3 dx_frac = shift - make_scalar3(i,j,k);

                    unsigned int neigh_idx;
                    if (local_fft)
                        // use row major order for kiss FFT
                        neigh_idx = neighi + m_mesh_points.x * (neighj + m_mesh_points.y*neighk);
                    else
                        neigh_idx = m_force_mesh_index(neighi,neighj,neighk);

                    kiss_fft_cpx inv_mesh = h_inv_fourier_mesh.data[neigh_idx];
                    force += -(Scalar)m_mesh_points.x*b1*mode*assignTSCderiv(dx_frac.x)*assignTSC(dx_frac.y)*assignTSC(dx_frac.z)*inv_mesh.r;
                    force += -(Scalar)m_mesh_points.y*b2*mode*assignTSC(dx_frac.x)*assignTSCderiv(dx_frac.y)*assignTSC(dx_frac.z)*inv_mesh.r;
                    force += -(Scalar)m_mesh_points.z*b3*mode*assignTSC(dx_frac.x)*assignTSC(dx_frac.y)*assignTSCderiv(dx_frac.z)*inv_mesh.r;

                    }  

        // Multiply with bias potential derivative
        force *= Scalar(2.0)/(Scalar)n_global*m_bias;

        h_force.data[idx] = make_scalar4(force.x,force.y,force.z,0.0);
         
        }  // end of loop over particles

    if (m_prof) m_prof->pop();
    }

Scalar OrderParameterMesh::computeCV()
    {
    if (m_prof) m_prof->push("sum");

    ArrayHandle<kiss_fft_cpx> h_fourier_mesh(m_fourier_mesh, access_location::host, access_mode::read);
    ArrayHandle<kiss_fft_cpx> h_fourier_mesh_G(m_fourier_mesh_G, access_location::host, access_mode::read);

    Scalar sum(0.0);

    bool local_fft = m_kiss_fft_initialized;
    #ifdef ENABLE_MPI
    DFFTIndex dffti;
    if (!local_fft) dffti = m_kiss_dfft->getIndexer();
    #endif

    for (unsigned int k = 0; k < m_n_inner_cells; ++k)
        {
        bool exclude;
        if (local_fft)
            // exclude DC bin
            exclude = (k == 0);
        #ifdef ENABLE_MPI
        else
            {
            uint3 n = dffti(k);
            exclude = (n.x == 0 && n.y == 0 && n.z == 0);
            }
        #endif

        if (! exclude)
            sum += h_fourier_mesh_G.data[k].r * h_fourier_mesh.data[k].r
                + h_fourier_mesh_G.data[k].i * h_fourier_mesh.data[k].i;
        }

    sum *= Scalar(1.0/2.0);

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

    if (m_prof) m_prof->pop();

    return sum;
    }

Scalar OrderParameterMesh::getCurrentValue(unsigned int timestep)
    {
    if (m_cv_last_updated == timestep && !m_is_first_step)
        return m_cv;

    if (m_prof) m_prof->push("Mesh");

    if (m_is_first_step)
        {
        // allocate memory and initialize arrays
        m_n_ghost_cells = computeNumGhostCells();
        m_exec_conf->msg->notice(3) << "cv.mesh: Ghost layer " << m_n_ghost_cells.x << "x"
                                    << m_n_ghost_cells.y << "x"
                                    << m_n_ghost_cells.z << std::endl;
 
        setupMesh();

        computeInfluenceFunction();
        m_is_first_step = false;
        }

    if (m_box_changed)
        {
        computeParticleGhostLayerWidth();

        uint3 n_ghost_cells = computeNumGhostCells();

        // do we need to reallocate?
        if (m_n_ghost_cells.x != n_ghost_cells.x ||
            m_n_ghost_cells.y != n_ghost_cells.y ||
            m_n_ghost_cells.z != n_ghost_cells.z)
            {
            m_n_ghost_cells = n_ghost_cells;
            m_exec_conf->msg->notice(3) << "cv.mesh: Reallocating ghost layer "
                                         << m_n_ghost_cells.x << "x"
                                         << m_n_ghost_cells.y << "x"
                                         << m_n_ghost_cells.z << std::endl;
            setupMesh();
            }

        computeInfluenceFunction();
        m_box_changed = false;
        }

    assignParticles();

    updateMeshes();

    m_cv = computeCV();

    m_cv_last_updated = timestep;

    if (m_prof) m_prof->pop();

    return m_cv;
    }

void OrderParameterMesh::computeVirial()
    {
    if (m_prof) m_prof->push("virial");

    ArrayHandle<kiss_fft_cpx> h_fourier_mesh(m_fourier_mesh, access_location::host, access_mode::overwrite);
    ArrayHandle<kiss_fft_cpx> h_fourier_mesh_G(m_fourier_mesh_G, access_location::host, access_mode::overwrite);

    ArrayHandle<Scalar> h_inf_f(m_inf_f, access_location::host, access_mode::read);
    ArrayHandle<Scalar3> h_k(m_k, access_location::host, access_mode::read);

    Scalar virial[6];
    for (unsigned int i = 0; i < 6; ++i)
        virial[i] = Scalar(0.0);

    bool local_fft = m_kiss_fft_initialized;
    #ifdef ENABLE_MPI
    DFFTIndex dffti;
    if (!local_fft) dffti = m_kiss_dfft->getIndexer();
    #endif

    for (unsigned int kidx = 0; kidx < m_n_inner_cells; ++kidx)
        {
        bool exclude;
        if (local_fft)
            // exclude DC bin
            exclude = (kidx == 0);
        #ifdef ENABLE_MPI
        else
            {
            uint3 n = dffti(kidx);
            exclude = (n.x == 0 && n.y == 0 && n.z == 0);
            }
        #endif

        if (! exclude)
            {
            // non-zero wave vector
            kiss_fft_cpx f_g = h_fourier_mesh_G.data[kidx];
            kiss_fft_cpx f = h_fourier_mesh.data[kidx];

            Scalar rhog = f_g.r * f.r + f_g.i * f.i;
            Scalar3 k = h_k.data[kidx];
            Scalar k_cut = sqrt(m_qstarsq);
            Scalar ksq = dot(k,k);
            Scalar knorm = sqrt(ksq);
            Scalar fac = expf(-Scalar(12.0)*(knorm/k_cut-Scalar(1.0)));
            Scalar kfac = -Scalar(6.0)/(Scalar(1.0)+fac)/knorm/k_cut;

            virial[0] += rhog*kfac*k.x*k.x; // xx
            virial[1] += rhog*kfac*k.x*k.y; // xy
            virial[2] += rhog*kfac*k.x*k.z; // xz
            virial[3] += rhog*kfac*k.y*k.y; // yy
            virial[4] += rhog*kfac*k.y*k.z; // yz
            virial[5] += rhog*kfac*k.z*k.z; // zz
            }
        } 

    for (unsigned int i = 0; i < 6; ++i)
        m_external_virial[i] = m_bias*virial[i];

    if (m_prof) m_prof->pop();
    }
 

void OrderParameterMesh::computeBiasForces(unsigned int timestep)
    {

    if (m_is_first_step || m_cv_last_updated != timestep)
        getCurrentValue(timestep);

    if (m_prof) m_prof->push("Mesh");

    interpolateForces();

    PDataFlags flags = m_pdata->getFlags();

    if (flags[pdata_flag::pressure_tensor] || flags[pdata_flag::isotropic_virial])
        {
        computeVirial();
        }
    else
        {
        for (unsigned int i = 0; i < 6; ++i)
            m_external_virial[i] = Scalar(0.0);
        }

    if (m_prof) m_prof->pop();
    }

Scalar OrderParameterMesh::getLogValue(const std::string& quantity, unsigned int timestep)
    {
    if (quantity == m_log_names[0])
        {
        return getCurrentValue(timestep);
        }
    else if (quantity == m_log_names[1])
        {
        computeQmax(timestep);
        return m_q_max.x;
        }
    else if (quantity == m_log_names[2])
        {
        computeQmax(timestep);
        return m_q_max.y;
        }
    else if (quantity == m_log_names[3])
        {
        computeQmax(timestep);
        return m_q_max.z;
        }
    else
        {
        m_exec_conf->msg->error() << "cv.mesh: " << quantity << " is not a valid log quantity"
                  << std::endl;
        throw std::runtime_error("Error getting log value");
        }
    }

void OrderParameterMesh::computeQmax(unsigned int timestep)
    {
    // compute Fourier grid
    getCurrentValue(timestep);

    if (timestep && m_q_max_last_computed == timestep) return;
    m_q_max_last_computed = timestep;

    if (m_prof) m_prof->push("max q");

    ArrayHandle<kiss_fft_cpx> h_fourier_mesh(m_fourier_mesh, access_location::host, access_mode::read);
    ArrayHandle<Scalar3> h_k(m_k, access_location::host, access_mode::read);

    Scalar max_amplitude(0.0);
    Scalar3 q_max(make_scalar3(0.0,0.0,0.0));
    for (unsigned int kidx = 0; kidx < m_n_inner_cells; ++kidx)
        {
        Scalar a = h_fourier_mesh.data[kidx].r*h_fourier_mesh.data[kidx].r
                   + h_fourier_mesh.data[kidx].i*h_fourier_mesh.data[kidx].i;

        if (a > max_amplitude)
            {
            q_max = h_k.data[kidx];
            max_amplitude = a;
            }
        }
    
    #ifdef ENABLE_MPI
    if (m_pdata->getDomainDecomposition())
        {
        // all processes send their results to all other processes
        // and then they determine the maximum wave vector
        Scalar *all_amplitudes = new Scalar[m_exec_conf->getNRanks()];
        Scalar3 *all_q_max = new Scalar3[m_exec_conf->getNRanks()];
        MPI_Alltoall(&max_amplitude,
                       1,
                       MPI_HOOMD_SCALAR,
                       all_amplitudes,
                       1,
                       MPI_HOOMD_SCALAR,
                       m_exec_conf->getMPICommunicator());
        MPI_Alltoall(&q_max,
                       sizeof(Scalar3),
                       MPI_BYTE,
                       all_q_max, 
                       sizeof(Scalar3),
                       MPI_BYTE,
                       m_exec_conf->getMPICommunicator());

        for (unsigned int i = 0; i < m_exec_conf->getNRanks();++i)
            {
            if (all_amplitudes[i] > max_amplitude)
                {
                max_amplitude = all_amplitudes[i];
                q_max = all_q_max[i];
                }
            }
        
        delete [] all_amplitudes;
        delete [] all_q_max;
        }
    #endif

    if (m_prof) m_prof->pop();

    m_q_max = q_max;
    }

void export_OrderParameterMesh()
    {
    class_<OrderParameterMesh, boost::shared_ptr<OrderParameterMesh>, bases<CollectiveVariable>, boost::noncopyable >
        ("OrderParameterMesh", init< boost::shared_ptr<SystemDefinition>,
                                     const unsigned int,
                                     const unsigned int,
                                     const unsigned int,
                                     const Scalar,
                                     const std::vector<Scalar>,
                                     const std::vector<int3>
                                    >());

    }
