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
                                            std::vector<Scalar> mode)
    : CollectiveVariable(sysdef, "mesh"),
      m_n_ghost_cells(make_uint3(0,0,0)),
      m_radius(1),
      m_n_inner_cells(0),
      m_is_first_step(true),
      m_cv_last_updated(0),
      m_box_changed(false),
      m_cv(Scalar(0.0)),
      m_kiss_fft_initialized(false),
      m_log_name("mesh_energy")
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
    }

void OrderParameterMesh::setupMesh()
    {
    // set up ghost layer
    const BoxDim& box = m_pdata->getBox();
    uchar3 periodic = box.getPeriodic();

    m_n_ghost_cells = make_uint3(periodic.x ? 0 : 2*m_radius,
                                  periodic.y ? 0 : 2*m_radius,
                                  periodic.z ? 0 : 2*m_radius);

    m_mesh_index = Index3D(m_mesh_points.x+m_n_ghost_cells.x,
                           m_mesh_points.y+m_n_ghost_cells.y,
                           m_mesh_points.z+m_n_ghost_cells.z);
    m_n_inner_cells = m_mesh_points.x * m_mesh_points.y * m_mesh_points.z;

    m_cell_adj_indexer = Index2D((m_radius*2+1)*(m_radius*2+1)*(m_radius*2+1), m_mesh_index.getNumElements());

    // allocate adjacency matrix
    GPUArray<unsigned int> cell_adj(m_cell_adj_indexer.getNumElements(), m_exec_conf);
    m_cell_adj.swap(cell_adj);

    // setup adjacency matrix
    initializeCellAdj();

    // allocate memory for influence function and k values
    GPUArray<Scalar> inf_f(m_n_inner_cells, m_exec_conf);
    m_inf_f.swap(inf_f);

    GPUArray<Scalar3> k(m_n_inner_cells, m_exec_conf);
    m_k.swap(k);

    GPUArray<Scalar> virial_mesh(6*m_n_inner_cells, m_exec_conf);
    m_virial_mesh.swap(virial_mesh);

    initializeFFT();
    } 

void OrderParameterMesh::initializeFFT()
    {
    bool local_fft = true;

    #ifdef ENABLE_MPI
    local_fft = !m_pdata->getDomainDecomposition();

    if (! local_fft)
        {
        // ghost cell exchanger for forward direction
        m_mesh_comm_forward = boost::shared_ptr<CommunicatorMesh<kiss_fft_cpx> >(
            new CommunicatorMesh<kiss_fft_cpx>(m_sysdef, m_comm, m_n_ghost_cells, m_mesh_index, false));

        // ghost cell exchanger for reverse direction
        m_mesh_comm_inverse = boost::shared_ptr<CommunicatorMesh<kiss_fft_cpx> >(
            new CommunicatorMesh<kiss_fft_cpx>(m_sysdef, m_comm, m_n_ghost_cells, m_mesh_index, true));


        // set up distributed FFT 
        m_kiss_dfft = boost::shared_ptr<DistributedKISSFFT>(
            new DistributedKISSFFT(m_exec_conf, m_pdata->getDomainDecomposition(), m_mesh_index, m_n_ghost_cells));
        m_kiss_dfft->setProfiler(m_prof);
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
    unsigned int num_cells = m_mesh_index.getNumElements();
    GPUArray<kiss_fft_cpx> mesh(num_cells,m_exec_conf);
    m_mesh.swap(mesh);

    GPUArray<kiss_fft_cpx> fourier_mesh(m_n_inner_cells, m_exec_conf);
    m_fourier_mesh.swap(fourier_mesh);

    GPUArray<kiss_fft_cpx> fourier_mesh_G(m_n_inner_cells, m_exec_conf);
    m_fourier_mesh_G.swap(fourier_mesh_G);

    GPUArray<kiss_fft_cpx> fourier_mesh_x(m_n_inner_cells, m_exec_conf);
    m_fourier_mesh_x.swap(fourier_mesh_x);

    GPUArray<kiss_fft_cpx> fourier_mesh_y(m_n_inner_cells, m_exec_conf);
    m_fourier_mesh_y.swap(fourier_mesh_y);

    GPUArray<kiss_fft_cpx> fourier_mesh_z(m_n_inner_cells, m_exec_conf);
    m_fourier_mesh_z.swap(fourier_mesh_z);
 
    GPUArray<kiss_fft_cpx> force_mesh_x(num_cells, m_exec_conf);
    m_force_mesh_x.swap(force_mesh_x);

    GPUArray<kiss_fft_cpx> force_mesh_y(num_cells, m_exec_conf);
    m_force_mesh_y.swap(force_mesh_y);

    GPUArray<kiss_fft_cpx> force_mesh_z(num_cells, m_exec_conf);
    m_force_mesh_z.swap(force_mesh_z);
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
                             

void OrderParameterMesh::initializeCellAdj()
    {
    ArrayHandle<unsigned int> h_cell_adj(m_cell_adj, access_location::host, access_mode::overwrite);

    // loop over all inner cells
    for (int k = 0; k < int(m_mesh_points.z); k++)
        for (int j = 0; j < int(m_mesh_points.y); j++)
            for (int i = 0; i < int(m_mesh_points.x); i++)
                {
                unsigned int cur_cell = m_mesh_index(i,j,k);
                unsigned int offset = 0;

                // loop over neighboring cells
                // need signed integer values for performing index calculations with negative values
                int r = m_radius;
                int mx = int(m_mesh_points.x);
                int my = int(m_mesh_points.y);
                int mz = int(m_mesh_points.z);
                for (int nk = k-r; nk <= k+r; nk++)
                    for (int nj = j-r; nj <= j+r; nj++)
                        for (int ni = i-r; ni <= i+r; ni++)
                            {
                            int wrapi;
                            if (m_n_ghost_cells.x)
                                wrapi = ni;
                            else
                                {
                                wrapi = ni % mx;
                                if (wrapi < 0)
                                    wrapi += mx;
                                }

                            int wrapj;
                            if (m_n_ghost_cells.y) 
                                wrapj = nj;
                            else
                                {
                                wrapj = nj % my;
                                if (wrapj < 0)
                                    wrapj += my;
                                }

                            int wrapk;
                            if (m_n_ghost_cells.z)
                                wrapk = nk;
                            else
                                {
                                wrapk = nk % mz;
                                if (wrapk < 0)
                                    wrapk += mz;
                                }

                            unsigned int neigh_cell = m_mesh_index(wrapi, wrapj, wrapk);
                            h_cell_adj.data[m_cell_adj_indexer(offset, cur_cell)] = neigh_cell;
                            offset++;
                            }

                // sort the adj list for each cell
                sort(&h_cell_adj.data[m_cell_adj_indexer(0, cur_cell)],
                     &h_cell_adj.data[m_cell_adj_indexer(offset, cur_cell)]);
                }
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

//! Assignment of particles to mesh using three-point scheme (triangular shaped cloud)
/*! This is a second order accurate scheme with continuous value and continuous derivative
 */
void OrderParameterMesh::assignParticles()
    {
    if (m_prof) m_prof->push("assign");

    ArrayHandle<Scalar4> h_postype(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<kiss_fft_cpx> h_mesh(m_mesh, access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar> h_mode(m_mode, access_location::host, access_mode::read);

    ArrayHandle<unsigned int> h_cell_adj(m_cell_adj, access_location::host, access_mode::read);

    const BoxDim& global_box = m_pdata->getGlobalBox();
    const BoxDim& box = m_pdata->getBox();

    #ifdef ENABLE_MPI
    unsigned int nproc = m_exec_conf->getNRanks();
    unsigned int n_tot_mesh_points = nproc*m_n_inner_cells;
    #else
    unsigned int n_tot_mesh_points = m_n_inner_cells;
    #endif

    Scalar V_cell = global_box.getVolume()/((Scalar)n_tot_mesh_points);
 
    // set mesh to zero
    memset(h_mesh.data, 0, sizeof(kiss_fft_cpx)*m_mesh.getNumElements());
 
    // inverse dimensions
    Scalar3 dim_inv = make_scalar3(Scalar(1.0)/(Scalar)m_mesh_points.x,
                                   Scalar(1.0)/(Scalar)m_mesh_points.y,
                                   Scalar(1.0)/(Scalar)m_mesh_points.z); 

    bool local_fft = m_kiss_fft_initialized;

    for (unsigned int idx = 0; idx < m_pdata->getN(); ++idx)
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
        unsigned int ix = (((int)reduced_pos.x) + (int)m_n_ghost_cells.x/2);
        unsigned int iy = (((int)reduced_pos.y) + (int)m_n_ghost_cells.y/2);
        unsigned int iz = (((int)reduced_pos.z) + (int)m_n_ghost_cells.z/2);

        // handle particles on the boundary
        if (ix == m_mesh_points.x && !m_n_ghost_cells.x)
            ix = 0;
        if (iy == m_mesh_points.y && !m_n_ghost_cells.y)
            iy = 0;
        if (iz == m_mesh_points.z && !m_n_ghost_cells.z)
            iz = 0;

        // center of cell (in units of the mesh size)
        unsigned int my_cell = m_mesh_index(ix,iy,iz);

        // assign particle to cell and next neighbors
        for (unsigned int k = 0; k < m_cell_adj_indexer.getW(); ++k)
            {
            unsigned int neigh_cell = h_cell_adj.data[m_cell_adj_indexer(k, my_cell)];

            uint3 cell_coord = m_mesh_index.getTriple(neigh_cell); 

            Scalar3 neigh_frac = make_scalar3((Scalar) (cell_coord.x) - (Scalar)(m_n_ghost_cells.x/2) + Scalar(0.5),
                                              (Scalar) (cell_coord.y) - (Scalar)(m_n_ghost_cells.y/2) + Scalar(0.5),
                                              (Scalar) (cell_coord.z) - (Scalar)(m_n_ghost_cells.z/2) + Scalar(0.5));
 
            // coordinates of the neighboring cell between 0..1 
            Scalar3 neigh_frac_box = neigh_frac * dim_inv;
            Scalar3 neigh_pos = box.makeCoordinates(neigh_frac_box);

            // compute distance between particle and cell center in fractional coordinates using minimum image
            Scalar3 dx = box.minImage(neigh_pos - pos);
            Scalar3 center = box.makeCoordinates(make_scalar3(0.5,0.5,0.5));
            Scalar3 dx_frac_box = box.makeFraction(dx+center) - make_scalar3(0.5,0.5,0.5);
            Scalar3 dx_frac = dx_frac_box*make_scalar3(m_mesh_points.x, m_mesh_points.y, m_mesh_points.z);

            // compute fraction of particle density assigned to cell
            Scalar density_fraction = assignTSC(dx_frac.x)*assignTSC(dx_frac.y)*assignTSC(dx_frac.z)/V_cell;
            unsigned int neigh_idx;
            if (local_fft)
                // store in row major order for kiss FFT
                neigh_idx = cell_coord.x + m_mesh_points.x * (cell_coord.y + m_mesh_points.y*cell_coord.z);
            else
                neigh_idx = neigh_cell;

            h_mesh.data[neigh_idx].r += h_mode.data[type]*density_fraction;
            }
             
        }  // end of loop over particles

    if (m_prof) m_prof->pop();
    }

void OrderParameterMesh::updateMeshes()
    {
    #ifdef ENABLE_MPI
    if (m_pdata->getDomainDecomposition())
        {
        // update inner cells of mesh using ghost cells from neighboring processors
        if (m_prof) m_prof->push("ghost exchange");
        m_mesh_comm_forward->updateGhostCells(m_mesh);
        if (m_prof) m_prof->pop();
        }
    #endif

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
        m_kiss_dfft->FFT3D(m_mesh, m_fourier_mesh, false);
        }
    #endif

    ArrayHandle<kiss_fft_cpx> h_fourier_mesh(m_fourier_mesh, access_location::host, access_mode::readwrite);
    ArrayHandle<kiss_fft_cpx> h_fourier_mesh_G(m_fourier_mesh_G, access_location::host, access_mode::overwrite);
 
    Scalar V = m_pdata->getGlobalBox().getVolume();
    #ifdef ENABLE_MPI
    unsigned int nproc = m_exec_conf->getNRanks();
    unsigned int n_tot_mesh_points = nproc*m_n_inner_cells;
    #else
    unsigned int n_tot_mesh_points = m_n_inner_cells;
    #endif
    Scalar V_cell = V/((Scalar)n_tot_mesh_points);

    unsigned int N_global = m_pdata->getNGlobal();

        {
        ArrayHandle<kiss_fft_cpx> h_fourier_mesh_x(m_fourier_mesh_x, access_location::host, access_mode::overwrite);
        ArrayHandle<kiss_fft_cpx> h_fourier_mesh_y(m_fourier_mesh_y, access_location::host, access_mode::overwrite);
        ArrayHandle<kiss_fft_cpx> h_fourier_mesh_z(m_fourier_mesh_z, access_location::host, access_mode::overwrite);

        // multiply with influence function
        for (unsigned int k = 0; k < m_n_inner_cells; ++k)
            {
            kiss_fft_cpx f = h_fourier_mesh.data[k];

            // normalization
            f.r *= V_cell/ (Scalar) N_global;
            f.i *= V_cell/ (Scalar) N_global;

            Scalar val = f.r*f.r+f.i*f.i;

            h_fourier_mesh_G.data[k].r = f.r * val * h_inf_f.data[k];
            h_fourier_mesh_G.data[k].i = f.i * val * h_inf_f.data[k];

            h_fourier_mesh.data[k] = f;

            // factor of two to account for derivative of fourth power of a mode
            Scalar3 kval = Scalar(2.0)*h_k.data[k]/(Scalar)N_global;
            h_fourier_mesh_x.data[k].r = -h_fourier_mesh_G.data[k].i*kval.x;
            h_fourier_mesh_x.data[k].i = h_fourier_mesh_G.data[k].r*kval.x;

            h_fourier_mesh_y.data[k].r = -h_fourier_mesh_G.data[k].i*kval.y;
            h_fourier_mesh_y.data[k].i = h_fourier_mesh_G.data[k].r*kval.y;

            h_fourier_mesh_z.data[k].r = -h_fourier_mesh_G.data[k].i*kval.z;
            h_fourier_mesh_z.data[k].i = h_fourier_mesh_G.data[k].r*kval.z;
            }
        }

    if (m_kiss_fft_initialized)
        {
        // do a local inverse transform of the force mesh
        ArrayHandle<kiss_fft_cpx> h_force_mesh_x(m_force_mesh_x, access_location::host, access_mode::overwrite);
        ArrayHandle<kiss_fft_cpx> h_force_mesh_y(m_force_mesh_y, access_location::host, access_mode::overwrite);
        ArrayHandle<kiss_fft_cpx> h_force_mesh_z(m_force_mesh_z, access_location::host, access_mode::overwrite);
        ArrayHandle<kiss_fft_cpx> h_fourier_mesh_x(m_fourier_mesh_x, access_location::host, access_mode::read);
        ArrayHandle<kiss_fft_cpx> h_fourier_mesh_y(m_fourier_mesh_y, access_location::host, access_mode::read);
        ArrayHandle<kiss_fft_cpx> h_fourier_mesh_z(m_fourier_mesh_z, access_location::host, access_mode::read);

        kiss_fftnd(m_kiss_ifft_x, h_fourier_mesh_x.data, h_force_mesh_x.data);
        kiss_fftnd(m_kiss_ifft_y, h_fourier_mesh_y.data, h_force_mesh_y.data);
        kiss_fftnd(m_kiss_ifft_z, h_fourier_mesh_z.data, h_force_mesh_z.data);
        }

    #ifdef ENABLE_MPI
    if (m_pdata->getDomainDecomposition())
        {
        // Distributed inverse transform force on mesh points 
        m_kiss_dfft->FFT3D(m_fourier_mesh_x, m_force_mesh_x, true);
        m_kiss_dfft->FFT3D(m_fourier_mesh_y, m_force_mesh_y, true);
        m_kiss_dfft->FFT3D(m_fourier_mesh_z, m_force_mesh_z, true);
        }
    #endif

    if (m_prof) m_prof->pop();

    #ifdef ENABLE_MPI
    if (m_pdata->getDomainDecomposition())
        {
        // update outer cells of force mesh using ghost cells from neighboring processors
        if (m_prof) m_prof->push("ghost exchange");
        m_mesh_comm_inverse->updateGhostCells(m_force_mesh_x);
        m_mesh_comm_inverse->updateGhostCells(m_force_mesh_y);
        m_mesh_comm_inverse->updateGhostCells(m_force_mesh_z);
        if (m_prof) m_prof->pop();
        }
    #endif
    }

void OrderParameterMesh::interpolateForces()
    {
    if (m_prof) m_prof->push("interpolate");

    ArrayHandle<Scalar4> h_postype(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<kiss_fft_cpx> h_force_mesh_x(m_force_mesh_x, access_location::host, access_mode::read);
    ArrayHandle<kiss_fft_cpx> h_force_mesh_y(m_force_mesh_y, access_location::host, access_mode::read);
    ArrayHandle<kiss_fft_cpx> h_force_mesh_z(m_force_mesh_z, access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_mode(m_mode, access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_cell_adj(m_cell_adj, access_location::host, access_mode::read);

    ArrayHandle<Scalar4> h_force(m_force, access_location::host, access_mode::overwrite);

    const BoxDim& box = m_pdata->getBox();

    // inverse dimensions
    Scalar3 dim_inv = make_scalar3(Scalar(1.0)/(Scalar)m_mesh_points.x,
                                   Scalar(1.0)/(Scalar)m_mesh_points.y,
                                   Scalar(1.0)/(Scalar)m_mesh_points.z); 

    // particle number
    bool local_fft = m_kiss_fft_initialized;

    for (unsigned int idx = 0; idx < m_pdata->getN(); ++idx)
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
        unsigned int ix = (((int)reduced_pos.x) + (int)m_n_ghost_cells.x/2);
        unsigned int iy = (((int)reduced_pos.y) + (int)m_n_ghost_cells.y/2);
        unsigned int iz = (((int)reduced_pos.z) + (int)m_n_ghost_cells.z/2);

        // handle particles on the boundary
        if (ix == m_mesh_points.x && !m_n_ghost_cells.x)
            ix = 0;
        if (iy == m_mesh_points.y && !m_n_ghost_cells.y)
            iy = 0;
        if (iz == m_mesh_points.z && !m_n_ghost_cells.z)
            iz = 0;

        // center of cell (in units of the mesh size)
        unsigned int my_cell = m_mesh_index(ix,iy,iz);
        
        Scalar3 force = make_scalar3(0.0,0.0,0.0);

        // interpolate mesh forces from cell and next neighbors
        for (unsigned int k = 0; k < m_cell_adj_indexer.getW(); ++k)
            {
            unsigned int neigh_cell = h_cell_adj.data[m_cell_adj_indexer(k, my_cell)];

            uint3 cell_coord = m_mesh_index.getTriple(neigh_cell); 

            Scalar3 neigh_frac = make_scalar3((Scalar) (cell_coord.x) - (Scalar)(m_n_ghost_cells.x/2) + Scalar(0.5),
                                              (Scalar) (cell_coord.y) - (Scalar)(m_n_ghost_cells.y/2) + Scalar(0.5),
                                              (Scalar) (cell_coord.z) - (Scalar)(m_n_ghost_cells.z/2) + Scalar(0.5));

            // coordinates of the neighboring cell between 0..1 
            Scalar3 neigh_frac_box = neigh_frac * dim_inv;
            Scalar3 neigh_pos = box.makeCoordinates(neigh_frac_box);

            // compute distance between particle and cell center in fractional coordinates using minimum image
            Scalar3 dx = box.minImage(neigh_pos - pos);
            Scalar3 center = box.makeCoordinates(make_scalar3(0.5,0.5,0.5));
            Scalar3 dx_frac_box = box.makeFraction(dx+center) - make_scalar3(0.5,0.5,0.5);
            Scalar3 dx_frac = dx_frac_box*make_scalar3(m_mesh_points.x, m_mesh_points.y, m_mesh_points.z);

            unsigned int neigh_idx;
            if (local_fft)
                // use row major order for kiss FFT
                neigh_idx = cell_coord.x + m_mesh_points.x * (cell_coord.y + m_mesh_points.y*cell_coord.z);
            else
                neigh_idx = neigh_cell;

            force += -assignTSC(dx_frac.x)*assignTSC(dx_frac.y)*assignTSC(dx_frac.z)*h_mode.data[type]
                            *make_scalar3(h_force_mesh_x.data[neigh_idx].r,
                                          h_force_mesh_y.data[neigh_idx].r,
                                          h_force_mesh_z.data[neigh_idx].r);
            }  

        // Multiply with bias potential derivative
        force *= m_bias;

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
        setupMesh();

        computeInfluenceFunction();
        m_is_first_step = false;
        }

    if (m_box_changed)
        {
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
            Scalar kfac = Scalar(1.0)/m_qstarsq;
            virial[0] += rhog*(Scalar(1.0) - kfac*k.x*k.x); // xx
            virial[1] += rhog*(            - kfac*k.x*k.y); // xy
            virial[2] += rhog*(            - kfac*k.x*k.z); // xz
            virial[3] += rhog*(Scalar(1.0) - kfac*k.y*k.y); // yy
            virial[4] += rhog*(            - kfac*k.y*k.z); // yz
            virial[5] += rhog*(Scalar(1.0) - kfac*k.z*k.z); // zz
            }
        } 

    for (unsigned int i = 0; i < 6; ++i)
        m_external_virial[i] = m_bias*Scalar(1.0/2.0)*virial[i];

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
    if (quantity == m_log_name)
        {
        return getCurrentValue(timestep);
        }
    else
        {
        m_exec_conf->msg->error() << "cv.mesh: " << quantity << " is not a valid log quantity"
                  << std::endl;
        throw std::runtime_error("Error getting log value");
        }
    }

void export_OrderParameterMesh()
    {
    class_<OrderParameterMesh, boost::shared_ptr<OrderParameterMesh>, bases<CollectiveVariable>, boost::noncopyable >
        ("OrderParameterMesh", init< boost::shared_ptr<SystemDefinition>,
                                     const unsigned int,
                                     const unsigned int,
                                     const unsigned int,
                                     const Scalar,
                                     const std::vector<Scalar>&
                                    >());

    }
