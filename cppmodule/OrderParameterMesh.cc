#include "OrderParameterMesh.h"

using namespace boost::python;

bool is_pow2(unsigned int n)
    {
    while (n && n%2 == 0) { n/=2; }

    return (n == 1);
    };

/*! \param sysdef The system definition
    \param nx Number of cells along first axis
    \param ny Number of cells along second axis
    \param nz Number of cells along third axis
    \param mode Per-type modes to multiply density
 */
OrderParameterMesh::OrderParameterMesh(boost::shared_ptr<SystemDefinition> sysdef,
                                            const unsigned int nx,
                                            const unsigned int ny,
                                            const unsigned int nz,
                                            std::vector<Scalar> mode,
                                            std::vector<int3> zero_modes)
    : CollectiveVariable(sysdef, "mesh"),
      m_n_ghost_cells(make_uint3(0,0,0)),
      m_grid_dim(make_uint3(0,0,0)),
      m_ghost_width(make_scalar3(0,0,0)),
      m_n_cells(0),
      m_radius(1),
      m_n_inner_cells(0),
      m_is_first_step(true),
      m_cv_last_updated(0),
      m_box_changed(false),
      m_cv(Scalar(0.0)),
      m_q_max_last_computed(0),
      m_sq_pow(0.0),
      m_k_min(0.0),
      m_k_max(0.0),
      m_delta_k(0.0),
      m_use_table(false),
      m_kiss_fft_initialized(false)
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

    m_boxchange_connection = m_pdata->connectBoxChange(boost::bind(&OrderParameterMesh::setBoxChange, this));

    m_mesh_points = make_uint3(nx, ny, nz);

    #ifdef ENABLE_MPI
    if (m_pdata->getDomainDecomposition())
        {
        const Index3D& didx = m_pdata->getDomainDecomposition()->getDomainIndexer();

        if (!is_pow2(m_mesh_points.x) || !is_pow2(m_mesh_points.y) || !is_pow2(m_mesh_points.z))
            {
            m_exec_conf->msg->error()
                << "The number of mesh points along the every direction must be a power of two!" << std::endl;
            throw std::runtime_error("Error initializing cv.mesh");
            }

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

    m_ghost_offset = 0;
    #endif // ENABLE_MPI

    // reset virial
    ArrayHandle<Scalar> h_virial(m_virial, access_location::host, access_mode::overwrite);
    memset(h_virial.data, 0, sizeof(Scalar)*m_virial.getNumElements());

    m_log_names.push_back("cv_mesh");
    m_log_names.push_back("qx_max");
    m_log_names.push_back("qy_max");
    m_log_names.push_back("qz_max");
    m_log_names.push_back("sq_max");
    }

OrderParameterMesh::~OrderParameterMesh()
    {
    if (m_kiss_fft_initialized)
        {
        free(m_kiss_fft);
        free(m_kiss_ifft);
        kiss_fft_cleanup();
        }
    #ifdef ENABLE_MPI
    else
        {
        dfft_destroy_plan(m_dfft_plan_forward);
        dfft_destroy_plan(m_dfft_plan_inverse);
        }
    #endif
    m_boxchange_connection.disconnect();
    }

/*! \param K Table for the convolution kernel
    \param kmin Minimum k in the potential
    \param kmax Maximum k in the potential
*/
void OrderParameterMesh::setTable(const std::vector<Scalar> &K,
                              const std::vector<Scalar> &d_K,
                              Scalar kmin,
                              Scalar kmax)
    {
    // range check on the parameters
    if (kmin < 0 || kmax < 0 || kmax <= kmin)
        {
        m_exec_conf->msg->error() << "cv.mesh kmin, kmax (" << kmin << "," << kmax
             << ") is invalid" << endl;
        throw runtime_error("Error setting up OrderParameterMesh");
        }

    if (K.size() != d_K.size())
        {
        m_exec_conf->msg->error() << "Convolution kernel and derivative have tables of unequal length "
            << K.size() << " != " << d_K.size() << std::endl;
        throw runtime_error("Error setting up OrderParameterMesh");
        }

    m_k_min = kmin;
    m_k_max = kmax;
    m_delta_k = (kmax - kmin) / Scalar(K.size() - 1);

    // allocate the arrays
    GPUArray<Scalar> table(K.size(), m_exec_conf);
    m_table.swap(table);

    GPUArray<Scalar> table_d(d_K.size(), m_exec_conf);
    m_table_d.swap(table_d);

    // access the array
    ArrayHandle<Scalar> h_table(m_table, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> h_table_d(m_table_d, access_location::host, access_mode::readwrite);

    // fill out the table
    for (unsigned int i = 0; i < m_table.getNumElements(); i++)
        {
        h_table.data[i] = K[i];
        h_table_d.data[i] = d_K[i];
        }
    }

void OrderParameterMesh::setupMesh()
    {
    // update number of ghost cells
    m_n_ghost_cells = computeGhostCellNum();

    // extra ghost cells are as wide as the inner cells
    const BoxDim& box = m_pdata->getBox();
    Scalar3 cell_width = box.getNearestPlaneDistance() /
        make_scalar3(m_mesh_points.x, m_mesh_points.y, m_mesh_points.z);
    m_ghost_width = cell_width*make_scalar3( m_n_ghost_cells.x, m_n_ghost_cells.y, m_n_ghost_cells.z);

    m_exec_conf->msg->notice(6) << "cv.mesh: (Re-)allocating ghost layer ("
                                 << m_n_ghost_cells.x << ","
                                 << m_n_ghost_cells.y << ","
                                 << m_n_ghost_cells.z << ")" << std::endl;

    m_grid_dim = make_uint3(m_mesh_points.x+2*m_n_ghost_cells.x,
                           m_mesh_points.y+2*m_n_ghost_cells.y,
                           m_mesh_points.z+2*m_n_ghost_cells.z);

    m_n_cells = m_grid_dim.x*m_grid_dim.y*m_grid_dim.z;
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

uint3 OrderParameterMesh::computeGhostCellNum()
    {
    // ghost cells
    uint3 n_ghost_cells = make_uint3(0,0,0);
    #ifdef ENABLE_MPI
    if (m_pdata->getDomainDecomposition())
        {
        Index3D di = m_pdata->getDomainDecomposition()->getDomainIndexer();
        n_ghost_cells.x = (di.getW() > 1) ? m_radius : 0;
        n_ghost_cells.y = (di.getH() > 1) ? m_radius : 0;
        n_ghost_cells.z = (di.getD() > 1) ? m_radius : 0;
        }
    #endif

    // extra ghost cells to accomodate skin layer (max 1/2 ghost layer width)
    #ifdef ENABLE_MPI
    if (m_comm)
        {
        Scalar r_buff = m_comm->getGhostLayerWidth();

        const BoxDim& box = m_pdata->getBox();
        Scalar3 cell_width = box.getNearestPlaneDistance() /
            make_scalar3(m_mesh_points.x, m_mesh_points.y, m_mesh_points.z);

        if (n_ghost_cells.x) n_ghost_cells.x += r_buff/cell_width.x + 1;
        if (n_ghost_cells.y) n_ghost_cells.y += r_buff/cell_width.y + 1;
        if (n_ghost_cells.z) n_ghost_cells.z += r_buff/cell_width.z + 1;
        }
    #endif
    return n_ghost_cells;
    }

void OrderParameterMesh::initializeFFT()
    {
    bool local_fft = true;

    #ifdef ENABLE_MPI
    local_fft = !m_pdata->getDomainDecomposition();

    if (! local_fft)
        {
        // ghost cell communicator for charge interpolation
        m_grid_comm_forward = std::auto_ptr<CommunicatorGrid<kiss_fft_cpx> >(
            new CommunicatorGrid<kiss_fft_cpx>(m_sysdef,
               make_uint3(m_mesh_points.x, m_mesh_points.y, m_mesh_points.z),
               make_uint3(m_grid_dim.x, m_grid_dim.y, m_grid_dim.z),
               m_n_ghost_cells,
               true));
        // ghost cell communicator for force mesh
        m_grid_comm_reverse = std::auto_ptr<CommunicatorGrid<kiss_fft_cpx> >(
            new CommunicatorGrid<kiss_fft_cpx>(m_sysdef,
               make_uint3(m_mesh_points.x, m_mesh_points.y, m_mesh_points.z),
               make_uint3(m_grid_dim.x, m_grid_dim.y, m_grid_dim.z),
               m_n_ghost_cells,
               false));
        // set up distributed FFTs
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
        dfft_create_plan(&m_dfft_plan_forward, 3, gdim, embed, NULL, pdim, pidx,
            row_m, 0, 1, m_exec_conf->getMPICommunicator(), (int *)h_cart_ranks.data);
        dfft_create_plan(&m_dfft_plan_inverse, 3, gdim, NULL, embed, pdim, pidx,
            row_m, 0, 1, m_exec_conf->getMPICommunicator(), (int *)h_cart_ranks.data);
        }
    #endif // ENABLE_MPI

    if (local_fft)
        {
        int dims[3];
        dims[0] = m_mesh_points.z;
        dims[1] = m_mesh_points.y;
        dims[2] = m_mesh_points.x;

        m_kiss_fft = kiss_fftnd_alloc(dims, 3, 0, NULL, NULL);
        m_kiss_ifft = kiss_fftnd_alloc(dims, 3, 1, NULL, NULL);

        m_kiss_fft_initialized = true;
        }

    // allocate mesh and transformed mesh
    GPUArray<kiss_fft_cpx> mesh(m_n_cells,m_exec_conf);
    m_mesh.swap(mesh);

    GPUArray<kiss_fft_cpx> fourier_mesh(m_n_inner_cells, m_exec_conf);
    m_fourier_mesh.swap(fourier_mesh);

    GPUArray<kiss_fft_cpx> fourier_mesh_G(m_n_inner_cells, m_exec_conf);
    m_fourier_mesh_G.swap(fourier_mesh_G);

    GPUArray<kiss_fft_cpx> inv_fourier_mesh(m_n_cells, m_exec_conf);
    m_inv_fourier_mesh.swap(inv_fourier_mesh);
    }

void OrderParameterMesh::computeInfluenceFunction()
    {
    if (m_prof) m_prof->push("influence function");

    ArrayHandle<Scalar> h_inf_f(m_inf_f,access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar3> h_k(m_k,access_location::host, access_mode::overwrite);

    ArrayHandle<Scalar> h_table(m_table, access_location::host, access_mode::read);

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

    bool local_fft = m_kiss_fft_initialized;

    uint3 global_dim = m_mesh_points;
    #ifdef ENABLE_MPI
    uint3 pdim=make_uint3(0,0,0);
    uint3 pidx=make_uint3(0,0,0);
    if (m_pdata->getDomainDecomposition())
        {
        const Index3D &didx = m_pdata->getDomainDecomposition()->getDomainIndexer();
        global_dim.x *= didx.getW();
        global_dim.y *= didx.getH();
        global_dim.z *= didx.getD();
        pidx = m_pdata->getDomainDecomposition()->getGridPos();
        pdim = make_uint3(didx.getW(), didx.getH(), didx.getD());
        }
    #endif

    for (unsigned int cell_idx = 0; cell_idx < m_n_inner_cells; ++cell_idx)
        {
        uint3 wave_idx;
        #ifdef ENABLE_MPI
        if (! local_fft)
           {
           // local layout: row major
           int ny = m_mesh_points.y;
           int nx = m_mesh_points.x;
           int n_local = cell_idx/ny/nx;
           int m_local = (cell_idx-n_local*ny*nx)/nx;
           int l_local = cell_idx % nx;
           // cyclic distribution
           wave_idx.x = l_local*pdim.x + pidx.x;
           wave_idx.y = m_local*pdim.y + pidx.y;
           wave_idx.z = n_local*pdim.z + pidx.z;
           }
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

        Scalar knorm = fast::sqrt(ksq);

        Scalar val(1.0);

        if (m_use_table && knorm >= m_k_min && knorm < m_k_max)
            {
            Scalar value_f = (knorm - m_k_min) / m_delta_k;

            unsigned int value_i = (unsigned int) value_f;
            Scalar K0 = h_table.data[value_i];
            Scalar K1 = h_table.data[value_i+1];

            // interpolate
            Scalar f = value_f - Scalar(value_i);

            val = K0 + f * (K1-K0);
            }

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

    unsigned int nparticles = m_pdata->getN();

    // loop over local particles
    for (unsigned int idx = 0; idx < nparticles; ++idx)
        {
        Scalar4 postype = h_postype.data[idx];

        Scalar3 pos = make_scalar3(postype.x, postype.y, postype.z);
        unsigned int type = __scalar_as_int(postype.w);

        // compute coordinates in units of the mesh size
        Scalar3 f = box.makeFraction(pos);
        Scalar3 reduced_pos = make_scalar3(f.x * (Scalar) m_mesh_points.x,
                                           f.y * (Scalar) m_mesh_points.y,
                                           f.z * (Scalar) m_mesh_points.z);

        reduced_pos += make_scalar3(m_n_ghost_cells.x, m_n_ghost_cells.y, m_n_ghost_cells.z);

        // find cell the particle is in (rounding downwards)
        int ix = reduced_pos.x;
        int iy = reduced_pos.y;
        int iz = reduced_pos.z;

        // handle particles on the boundary
        if (ix == (int)m_grid_dim.x && !m_n_ghost_cells.x)
            ix = 0;
        if (iy == (int)m_grid_dim.y && !m_n_ghost_cells.y)
            iy = 0;
        if (iz == (int)m_grid_dim.z && !m_n_ghost_cells.z)
            iz = 0;

        // compute distance between particle and cell center
        // in fractional coordinates
        Scalar3 cell_center = make_scalar3((Scalar)(ix-(int)m_n_ghost_cells.x)+Scalar(0.5),
                             (Scalar)(iy-(int)m_n_ghost_cells.y)+Scalar(0.5),
                             (Scalar)(iz-(int)m_n_ghost_cells.z)+Scalar(0.5));

        // compute minimum image separation to center
        Scalar3 c_cart = box.makeCoordinates(cell_center/make_scalar3(m_mesh_points.x,m_mesh_points.y,m_mesh_points.z));
        Scalar3 shift_cart = box.minImage(pos-c_cart);
        Scalar3 shift_f = box.makeFraction(shift_cart+box.getLo());
        Scalar3 shift = shift_f*make_scalar3(m_mesh_points.x,m_mesh_points.y,m_mesh_points.z);

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
                        if (neighi == (int) m_grid_dim.x)
                            neighi = 0;
                        else if (neighi < 0)
                            neighi += m_grid_dim.x;
                        }
                    assert(neighi >= 0 && neighi < (int) m_grid_dim.x);

                    if (! m_n_ghost_cells.y)
                        {
                        if (neighj == (int) m_grid_dim.y)
                            neighj = 0;
                        else if (neighj < 0)
                            neighj += m_grid_dim.y;
                        }
                    assert(neighj >= 0 && neighj < (int) m_grid_dim.y);

                    if (! m_n_ghost_cells.z)
                        {
                        if (neighk == (int) m_grid_dim.z)
                            neighk = 0;
                        else if (neighk < 0)
                            neighk += m_grid_dim.z;
                        }
                    assert(neighk >= 0 && neighk < (int) m_grid_dim.z);

                    Scalar3 dx_frac = shift - make_scalar3(i,j,k);

                    // compute fraction of particle density assigned to cell
                    Scalar density_fraction = assignTSC(dx_frac.x)*assignTSC(dx_frac.y)*assignTSC(dx_frac.z);
                    unsigned int neigh_idx;

                    // store in row major order
                    neigh_idx = neighi + m_grid_dim.x * (neighj + m_grid_dim.y*neighk);

                    h_mesh.data[neigh_idx].r += h_mode.data[type]*density_fraction;
                    }

        }  // end of loop over particles

    if (m_prof) m_prof->pop();
    }

void OrderParameterMesh::updateMeshes()
    {

    ArrayHandle<Scalar> h_inf_f(m_inf_f, access_location::host, access_mode::read);
    ArrayHandle<Scalar3> h_k(m_k, access_location::host, access_mode::read);

    if (m_kiss_fft_initialized)
        {
        if (m_prof) m_prof->push("FFT");
        // transform the particle mesh locally (forward transform)
        ArrayHandle<kiss_fft_cpx> h_mesh(m_mesh, access_location::host, access_mode::read);
        ArrayHandle<kiss_fft_cpx> h_fourier_mesh(m_fourier_mesh, access_location::host, access_mode::overwrite);

        kiss_fftnd(m_kiss_fft, h_mesh.data, h_fourier_mesh.data);
        if (m_prof) m_prof->pop();
        }

    #ifdef ENABLE_MPI
    if (m_pdata->getDomainDecomposition())
        {
        // update inner cells of particle mesh
        if (m_prof) m_prof->push("ghost cell update");
        m_exec_conf->msg->notice(8) << "cv.mesh: Ghost cell update" << std::endl;
        m_grid_comm_forward->communicate(m_mesh);
        if (m_prof) m_prof->pop();

        // perform a distributed FFT
        m_exec_conf->msg->notice(8) << "cv.mesh: Distributed FFT mesh" << std::endl;

        if (m_prof) m_prof->push("FFT");
        ArrayHandle<kiss_fft_cpx> h_mesh(m_mesh, access_location::host, access_mode::read);
        ArrayHandle<kiss_fft_cpx> h_fourier_mesh(m_fourier_mesh, access_location::host, access_mode::overwrite);

        dfft_execute((cpx_t *)(h_mesh.data+m_ghost_offset), (cpx_t *)h_fourier_mesh.data, 0,m_dfft_plan_forward);
        if (m_prof) m_prof->pop();
        }
    #endif

    if (m_prof) m_prof->push("update");

    ArrayHandle<kiss_fft_cpx> h_fourier_mesh(m_fourier_mesh, access_location::host, access_mode::readwrite);

    unsigned int N_global = m_pdata->getNGlobal();

        {
        ArrayHandle<kiss_fft_cpx> h_fourier_mesh_G(m_fourier_mesh_G, access_location::host, access_mode::overwrite);
        // multiply with influence function
        for (unsigned int k = 0; k < m_n_inner_cells; ++k)
            {
            kiss_fft_cpx f = h_fourier_mesh.data[k];

            // normalization
            f.r /= (Scalar) N_global;
            f.i /= (Scalar) N_global;

            Scalar val(1.0);
            if (m_sq_pow > Scalar(0.0))
                {
                val = fast::pow(f.r*f.r+f.i*f.i, m_sq_pow);
                }

            h_fourier_mesh_G.data[k].r = f.r * val * h_inf_f.data[k];
            h_fourier_mesh_G.data[k].i = f.i * val * h_inf_f.data[k];

            h_fourier_mesh.data[k] = f;
            }
        }

    if (m_prof) m_prof->pop();

    if (m_kiss_fft_initialized)
        {
        if (m_prof) m_prof->push("FFT");
        // do a local inverse transform of the force mesh
        ArrayHandle<kiss_fft_cpx> h_inv_fourier_mesh(m_inv_fourier_mesh, access_location::host, access_mode::overwrite);
        ArrayHandle<kiss_fft_cpx> h_fourier_mesh_G(m_fourier_mesh_G, access_location::host, access_mode::read);
        kiss_fftnd(m_kiss_ifft, h_fourier_mesh_G.data, h_inv_fourier_mesh.data);
        if (m_prof) m_prof->pop();
        }

    #ifdef ENABLE_MPI
    if (m_pdata->getDomainDecomposition())
        {
        if (m_prof) m_prof->push("FFT");
        // Distributed inverse transform force on mesh points
        m_exec_conf->msg->notice(8) << "cv.mesh: Distributed iFFT" << std::endl;

        ArrayHandle<kiss_fft_cpx> h_fourier_mesh_G(m_fourier_mesh_G, access_location::host, access_mode::read);
        ArrayHandle<kiss_fft_cpx> h_inv_fourier_mesh(m_inv_fourier_mesh, access_location::host, access_mode::overwrite);
        dfft_execute((cpx_t *)h_fourier_mesh_G.data, (cpx_t *)(h_inv_fourier_mesh.data+m_ghost_offset), 1,m_dfft_plan_inverse);
        if (m_prof) m_prof->pop();
        }
    #endif

    #ifdef ENABLE_MPI
    if (m_pdata->getDomainDecomposition())
        {
        // update outer cells of force mesh using ghost cells from neighboring processors
        if (m_prof) m_prof->push("ghost cell update");
        m_exec_conf->msg->notice(8) << "cv.mesh: Ghost cell update" << std::endl;
        m_grid_comm_reverse->communicate(m_inv_fourier_mesh);
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
        unsigned int ix = (reduced_pos.x + (Scalar)m_n_ghost_cells.x);
        unsigned int iy = (reduced_pos.y + (Scalar)m_n_ghost_cells.y);
        unsigned int iz = (reduced_pos.z + (Scalar)m_n_ghost_cells.z);

        // handle particles on the boundary
        if (ix == m_grid_dim.x && !m_n_ghost_cells.x)
            ix = 0;
        if (iy == m_grid_dim.y && !m_n_ghost_cells.y)
            iy = 0;
        if (iz == m_grid_dim.z && !m_n_ghost_cells.z)
            iz = 0;

        // center of cell (in units of the mesh size)
        Scalar3 cell_center = make_scalar3((Scalar)ix - (Scalar)(m_n_ghost_cells.x) + Scalar(0.5),
                                           (Scalar)iy - (Scalar)(m_n_ghost_cells.y) + Scalar(0.5),
                                           (Scalar)iz - (Scalar)(m_n_ghost_cells.z) + Scalar(0.5));

        // compute minimum image separation to center
        Scalar3 c_cart = box.makeCoordinates(cell_center/make_scalar3(m_mesh_points.x,m_mesh_points.y,m_mesh_points.z));
        Scalar3 shift_cart = box.minImage(pos-c_cart);
        Scalar3 shift_f = box.makeFraction(shift_cart+box.getLo());
        Scalar3 shift = shift_f*make_scalar3(m_mesh_points.x,m_mesh_points.y,m_mesh_points.z);

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
                        if (neighi == (int)m_grid_dim.x)
                            neighi = 0;
                        else if (neighi < 0)
                            neighi += m_grid_dim.x;
                        }

                    if (! m_n_ghost_cells.y)
                        {
                        if (neighj == (int)m_grid_dim.y)
                            neighj = 0;
                        else if (neighj < 0)
                            neighj += m_grid_dim.y;
                        }

                    if (! m_n_ghost_cells.z)
                        {
                        if (neighk == (int)m_grid_dim.z)
                            neighk = 0;
                        else if (neighk < 0)
                            neighk += m_grid_dim.z;
                        }

                    Scalar3 dx_frac = shift - make_scalar3(i,j,k);

                    unsigned int neigh_idx;
                    neigh_idx = neighi + m_grid_dim.x * (neighj + m_grid_dim.y*neighk);

                    kiss_fft_cpx inv_mesh = h_inv_fourier_mesh.data[neigh_idx];
                    force += -(Scalar)m_mesh_points.x*b1*mode*assignTSCderiv(dx_frac.x)*assignTSC(dx_frac.y)*assignTSC(dx_frac.z)*inv_mesh.r;
                    force += -(Scalar)m_mesh_points.y*b2*mode*assignTSC(dx_frac.x)*assignTSCderiv(dx_frac.y)*assignTSC(dx_frac.z)*inv_mesh.r;
                    force += -(Scalar)m_mesh_points.z*b3*mode*assignTSC(dx_frac.x)*assignTSC(dx_frac.y)*assignTSCderiv(dx_frac.z)*inv_mesh.r;
                   }

        // Multiply with bias potential derivative
        force *= (Scalar(2.0)*(m_sq_pow+Scalar(1.0)))/(Scalar)n_global*m_bias/Scalar(2.0);

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

    bool exclude_dc = true;
    #ifdef ENABLE_MPI
    if (m_pdata->getDomainDecomposition())
        {
        uint3 my_pos = m_pdata->getDomainDecomposition()->getGridPos();
        exclude_dc = !my_pos.x && !my_pos.y && !my_pos.z;
        }
    #endif

    for (unsigned int k = 0; k < m_n_inner_cells; ++k)
        {
        bool exclude = false;
        if (exclude_dc)
            // exclude DC bin
            exclude = (k == 0);

        if (! exclude)
            {
            sum += h_fourier_mesh_G.data[k].r * h_fourier_mesh.data[k].r
                + h_fourier_mesh_G.data[k].i * h_fourier_mesh.data[k].i;
            }
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
        setupMesh();

        computeInfluenceFunction();
        m_is_first_step = false;
        }

    bool ghost_cell_num_changed = false;
    uint3 n_ghost_cells = computeGhostCellNum();

    // do we need to reallocate?
    if (m_n_ghost_cells.x != n_ghost_cells.x ||
        m_n_ghost_cells.y != n_ghost_cells.y ||
        m_n_ghost_cells.z != n_ghost_cells.z)
        ghost_cell_num_changed = true;

    if (m_box_changed || ghost_cell_num_changed)
        {
        if (ghost_cell_num_changed) setupMesh();
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

    ArrayHandle<Scalar> h_table_d(m_table_d, access_location::host, access_mode::read);

    Scalar virial[6];
    for (unsigned int i = 0; i < 6; ++i)
        virial[i] = Scalar(0.0);

    bool exclude_dc = true;
    #ifdef ENABLE_MPI
    if (m_pdata->getDomainDecomposition())
        {
        uint3 my_pos = m_pdata->getDomainDecomposition()->getGridPos();
        exclude_dc = !my_pos.x && !my_pos.y && !my_pos.z;
        }
    #endif

    unsigned int Nglobal = m_pdata->getNGlobal();

    for (unsigned int kidx = 0; kidx < m_n_inner_cells; ++kidx)
        {
        bool exclude = false;
        if (exclude_dc)
            // exclude DC bin
            exclude = (kidx == 0);

        if (! exclude)
            {
            // non-zero wave vector
            kiss_fft_cpx fourier = h_fourier_mesh.data[kidx];

            Scalar3 k = h_k.data[kidx];
            Scalar ksq = dot(k,k);
            Scalar knorm = sqrt(ksq);

            Scalar kfac = Scalar(1.0)/Scalar(2.0)/knorm;

            // derivative of convolution kernel
            Scalar val_D(0.0);

            if (m_use_table && knorm >= m_k_min && knorm < m_k_max)
                {
                Scalar value_f = (knorm - m_k_min) / m_delta_k;

                unsigned int value_i = (unsigned int) value_f;
                Scalar dK0 = h_table_d.data[value_i];
                Scalar dK1 = h_table_d.data[value_i+1];

                // interpolate
                Scalar f = value_f - Scalar(value_i);

                val_D = dK0 + f * (dK1-dK0);
                }

            kfac *= val_D;

            Scalar val(1.0);
            if (m_sq_pow > Scalar(0.0))
                {
                val = fast::pow((fourier.r*fourier.r+fourier.i*fourier.i)/(Scalar)Nglobal, m_sq_pow);
                }

            Scalar rhog = (fourier.r * fourier.r + fourier.i * fourier.i)*val/(Scalar)Nglobal;

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
    else if (quantity == m_log_names[4])
        {
        computeQmax(timestep);
        return m_sq_max;
        }

    // nothing found? return base class value
    return CollectiveVariable::getLogValue(quantity, timestep);
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
        MPI_Allgather(&max_amplitude,
                       1,
                       MPI_HOOMD_SCALAR,
                       all_amplitudes,
                       1,
                       MPI_HOOMD_SCALAR,
                       m_exec_conf->getMPICommunicator());
        MPI_Allgather(&q_max,
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
    m_sq_max = max_amplitude;

    // normalize with 1/V
    unsigned int n_global = m_pdata->getNGlobal();
    m_sq_max *= (Scalar)n_global*(Scalar)n_global/m_pdata->getGlobalBox().getVolume();
    }

void export_OrderParameterMesh()
    {
    class_<OrderParameterMesh, boost::shared_ptr<OrderParameterMesh>, bases<CollectiveVariable>, boost::noncopyable >
        ("OrderParameterMesh", init< boost::shared_ptr<SystemDefinition>,
                                     const unsigned int,
                                     const unsigned int,
                                     const unsigned int,
                                     const std::vector<Scalar>,
                                     const std::vector<int3>
                                    >())
        .def("setSqPower", &OrderParameterMesh::setSqPower)
        .def("setTable", &OrderParameterMesh::setTable)
        .def("setUseTable", &OrderParameterMesh::setUseTable);
    }
