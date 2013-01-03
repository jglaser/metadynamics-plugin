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
      m_radius(1),
      m_is_first_step(true),
      m_cv_last_updated(0),
      m_kiss_fft(NULL),
      m_kiss_ifft_x(NULL),
      m_kiss_ifft_y(NULL),
      m_kiss_ifft_z(NULL),
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
    m_boxchange_connection = m_pdata->connectBoxChange(boost::bind(&OrderParameterMesh::computeInfluenceFunction, this));

    m_mesh_points = make_uint3(nx, ny, nz);

    // reset virial
    ArrayHandle<Scalar> h_virial(m_virial, access_location::host, access_mode::overwrite);
    memset(h_virial.data, 0, m_virial.getNumElements());

    m_bias = Scalar(1.0);
    }

OrderParameterMesh::~OrderParameterMesh()
    {
    if (m_kiss_fft)
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
    unsigned int i = 1;
    bool is_pow2_x = false;
    while (i <= m_mesh_points.x)
        {
        is_pow2_x = (m_mesh_points.x == i);
        i *= 2;
        }

    i = 1;
    bool is_pow2_y = false;
    while (i <= m_mesh_points.y)
        {
        is_pow2_y = (m_mesh_points.y == i);
        i *= 2;
        }

    i = 1;
    bool is_pow2_z = false;
    while (i <= m_mesh_points.z)
        {
        is_pow2_z = (m_mesh_points.z == i);
        i *= 2;
        }

    if (!is_pow2_x || !is_pow2_y || !is_pow2_z)
        {
        m_exec_conf->msg->warning() << "Mesh dimension not a power of two. Performance may be sub-optimal."
                                    << std::endl << std::endl;
        }

    if ((m_mesh_points.x % 2) || (m_mesh_points.y % 2) || (m_mesh_points.z %2))
        {
        m_exec_conf->msg->warning() << "Number of mesh points must be even."
                                    << std::endl << std::endl;
        throw std::runtime_error("Error setting up mesh.");
        } 

    if ((m_mesh_points.x <8) || (m_mesh_points.y < 8) || (m_mesh_points.z <8))
        {
        m_exec_conf->msg->warning() << "Number of mesh points must be greater or equal 8."
                                    << std::endl << std::endl;
        throw std::runtime_error("Error setting up mesh.");
        }
 
    m_mesh_index = Index3D(m_mesh_points.x, m_mesh_points.y, m_mesh_points.z);
    m_cell_adj_indexer = Index2D((m_radius*2+1)*(m_radius*2+1)*(m_radius*2+1), m_mesh_index.getNumElements());

    // allocate adjacency matrix
    GPUArray<unsigned int> cell_adj(m_cell_adj_indexer.getNumElements(), m_exec_conf);
    m_cell_adj.swap(cell_adj);

    // setup adjacency matrix
    initializeCellAdj();

    // allocate memory for influence function and k values
    GPUArray<Scalar> inf_f(m_mesh_index.getNumElements(), m_exec_conf);
    m_inf_f.swap(inf_f);

    GPUArray<Scalar3> k(m_mesh_index.getNumElements(), m_exec_conf);
    m_k.swap(k);

    initializeFFT();
    } 

void OrderParameterMesh::initializeFFT()
    {
    int dims[3];
    dims[0] = m_mesh_points.x;
    dims[1] = m_mesh_points.y;
    dims[2] = m_mesh_points.z;

    if (m_kiss_fft)
        {
        free(m_kiss_fft);
        free(m_kiss_ifft_x);
        free(m_kiss_ifft_y);
        free(m_kiss_ifft_z);
        }

    m_kiss_fft = kiss_fftnd_alloc(dims, 3, 0, NULL, NULL);
    m_kiss_ifft_x = kiss_fftnd_alloc(dims, 3, 1, NULL, NULL);
    m_kiss_ifft_y = kiss_fftnd_alloc(dims, 3, 1, NULL, NULL);
    m_kiss_ifft_z = kiss_fftnd_alloc(dims, 3, 1, NULL, NULL);

    // allocate mesh and transformed mesh
    unsigned int num_cells = m_mesh_index.getNumElements();
    GPUArray<kiss_fft_cpx> mesh(num_cells,m_exec_conf);
    m_mesh.swap(mesh);

    GPUArray<kiss_fft_cpx> fourier_mesh(num_cells, m_exec_conf);
    m_fourier_mesh.swap(fourier_mesh);

    GPUArray<kiss_fft_cpx> fourier_mesh_G(num_cells, m_exec_conf);
    m_fourier_mesh_G.swap(fourier_mesh_G);

    GPUArray<kiss_fft_cpx> fourier_mesh_x(num_cells, m_exec_conf);
    m_fourier_mesh_x.swap(fourier_mesh_x);

    GPUArray<kiss_fft_cpx> fourier_mesh_y(num_cells, m_exec_conf);
    m_fourier_mesh_y.swap(fourier_mesh_y);

    GPUArray<kiss_fft_cpx> fourier_mesh_z(num_cells, m_exec_conf);
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
    memset(h_inf_f.data, 0, sizeof(Scalar)*m_mesh_index.getNumElements());
    memset(h_k.data, 0, sizeof(Scalar3)*m_mesh_index.getNumElements());

    // maximal integer indices of inner loop
    const int maxnx = 3;
    const int maxny = 3;
    const int maxnz = 3;

    Scalar3 dim_inv = make_scalar3(Scalar(1.0)/(Scalar)m_mesh_points.x,
                                   Scalar(1.0)/(Scalar)m_mesh_points.y,
                                   Scalar(1.0)/(Scalar)m_mesh_points.z);
    const BoxDim& global_box = m_pdata->getGlobalBox();

    // compute reciprocal lattice vectors
    Scalar3 a1 = global_box.getLatticeVector(0);
    Scalar3 a2 = global_box.getLatticeVector(1);
    Scalar3 a3 = global_box.getLatticeVector(2);

    Scalar V_box = global_box.getVolume();
    Scalar3 b1 = Scalar(2.0*M_PI)*make_scalar3(a2.y*a3.z-a2.z*a3.y, a2.z*a3.x-a2.x*a3.z, a2.x*a3.y-a2.y*a3.x)/V_box;
    Scalar3 b2 = Scalar(2.0*M_PI)*make_scalar3(a3.y*a1.z-a3.z*a1.y, a3.z*a1.x-a3.x*a1.z, a3.x*a1.y-a3.y*a1.x)/V_box;
    Scalar3 b3 = Scalar(2.0*M_PI)*make_scalar3(a1.y*a2.z-a1.z*a2.y, a1.z*a2.x-a1.x*a2.z, a1.x*a2.y-a1.y*a2.x)/V_box;

    // particle number
    unsigned int N = m_pdata->getNGlobal();

    m_E_self = Scalar(0.0);

    for (int l = -(int)m_mesh_points.x/2 ; l < (int)m_mesh_points.x/2; ++l)
        for (int m = -(int)m_mesh_points.y/2 ; m < (int)m_mesh_points.y/2; ++m)
            for (int n = -(int)m_mesh_points.z/2 ; n < (int)m_mesh_points.z/2; ++n)
                {
                Scalar3 k = (Scalar)l*b1+(Scalar)m*b2+(Scalar)n*b3;
                Scalar ksq = dot(k,k);

                // accumulate self energy
                m_E_self += exp(-ksq/m_qstarsq*Scalar(1.0/2.0));

                Scalar3 UsqR = make_scalar3(0.0,0.0,0.0);
                Scalar Usq = Scalar(0.0);
                for (int nx = -maxnx; nx <= maxnx; ++nx)
                    for (int ny = -maxny; ny <= maxny; ++ny)
                        for (int nz = -maxnz; nz <= maxnz; ++nz)
                            {
                            Scalar3 kn = k + (Scalar)m_mesh_points.x*(Scalar)nx*b1+
                                           + (Scalar)m_mesh_points.y*(Scalar)ny*b2+
                                           + (Scalar)m_mesh_points.z*(Scalar)nz*b3;

                            Scalar3 knH = Scalar(2.0*M_PI)*(make_scalar3(l,m,n)*dim_inv+make_scalar3(nx,ny,nz));
                            Scalar U = assignTSCFourier(knH.x)*assignTSCFourier(knH.y)*assignTSCFourier(knH.z);
                            Scalar knsq = dot(kn,kn);
                            UsqR += U*U*kn*exp(-knsq/m_qstarsq*Scalar(1.0/2.0))/(Scalar)N/(Scalar)N*V_box;
                            Usq += U*U;
                            }

                Scalar num = dot(k,UsqR);
                Scalar3 kH = Scalar(2.0*M_PI)*make_scalar3(l,m,n)*dim_inv;

                Scalar denom = ksq*Usq*Usq;

                // determine cell idx
                unsigned int ix, iy, iz;
                if (l < 0)
                    ix = l + (int) m_mesh_points.x;
                else
                    ix = l;

                if (m < 0)
                    iy = m + (int) m_mesh_points.y;
                else
                    iy = m;

                if (n < 0)
                    iz = n + (int) m_mesh_points.z;
                else
                    iz = n;

                unsigned int cell_idx = iz + m_mesh_points.z * iy + m_mesh_points.y * m_mesh_points.z * ix;

                if ((l != 0) || (m != 0) || (n!=0)) 
                    h_inf_f.data[cell_idx] = num/denom;
                else
                    // avoid divide by zero
                    h_inf_f.data[cell_idx] = Scalar(1.0)/(Scalar)N/(Scalar)N*V_box;

                h_k.data[cell_idx] = k;
                }

    m_E_self *= Scalar(1.0/2.0)/(Scalar)N;

    if (m_prof) m_prof->pop();
    }
                             

void OrderParameterMesh::initializeCellAdj()
    {
    ArrayHandle<unsigned int> h_cell_adj(m_cell_adj, access_location::host, access_mode::overwrite);

    // loop over all cells
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
                            int wrapi = ni % mx;
                            if (wrapi < 0)
                                wrapi += mx;

                            int wrapj = nj % my;
                            if (wrapj < 0)
                                wrapj += my;

                            int wrapk = nk % mz;
                            if (wrapk < 0)
                                wrapk += mz;

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

/*! \param k Wave vector times mesh size
 * \returns the Fourier transformed TSC assignment function
 */
Scalar OrderParameterMesh::assignTSCFourier(Scalar k)
    {
    Scalar fac = 0;

    if (k*k <= Scalar(1.0))
        {
        Scalar term = Scalar(1.0);
        for (unsigned int i = 0; i < 6; ++i)
            {
            fac += coeff[i] * term;
            term *= k*k/Scalar(4.0);
            }
        }
    else
        {
        fac = Scalar(2.0)*sin(k*Scalar(1.0/2.0))/k;
        }

    return fac*fac*fac;
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
 
    Scalar V_cell = global_box.getVolume()/(Scalar)m_mesh_index.getNumElements();
 
    // set mesh to zero
    memset(h_mesh.data, 0, sizeof(kiss_fft_cpx)*m_mesh.getNumElements());
 
    // inverse dimensions
    Scalar3 dim_inv = make_scalar3(Scalar(1.0)/(Scalar)m_mesh_points.x,
                                   Scalar(1.0)/(Scalar)m_mesh_points.y,
                                   Scalar(1.0)/(Scalar)m_mesh_points.z); 

    for (unsigned int idx = 0; idx < m_pdata->getN(); ++idx)
        {
        Scalar4 postype = h_postype.data[idx];

        Scalar3 pos = make_scalar3(postype.x, postype.y, postype.z);
        unsigned int type = __scalar_as_int(postype.w);

        // compute coordinates in units of the mesh size
        Scalar3 f = global_box.makeFraction(pos);
        Scalar3 reduced_pos = make_scalar3(f.x * (Scalar) m_mesh_points.x,
                                           f.y * (Scalar) m_mesh_points.y,
                                           f.z * (Scalar) m_mesh_points.z);

        // find cell the particle is in
        unsigned int ix = reduced_pos.x;
        unsigned int iy = reduced_pos.y;
        unsigned int iz = reduced_pos.z;

        // handle particles on the boundary
        if (ix == m_mesh_points.x)
            ix = 0;
        if (iy == m_mesh_points.y)
            iy = 0;
        if (iz == m_mesh_points.z)
            iz = 0;

        // center of cell (in units of the mesh size)
        unsigned int my_cell = m_mesh_index(ix,iy,iz);

        // assign particle to cell and next neighbors
        for (unsigned int k = 0; k < m_cell_adj_indexer.getW(); ++k)
            {
            unsigned int neigh_cell = h_cell_adj.data[m_cell_adj_indexer(k, my_cell)];

            uint3 cell_coord = m_mesh_index.getTriple(neigh_cell); 

            Scalar3 neigh_frac = make_scalar3((Scalar) cell_coord.x + Scalar(0.5),
                                             (Scalar) cell_coord.y + Scalar(0.5),
                                             (Scalar) cell_coord.z + Scalar(0.5));
           
            // coordinates of the neighboring cell between 0..1 
            Scalar3 neigh_frac_box = neigh_frac * dim_inv;
            Scalar3 neigh_pos = global_box.makeCoordinates(neigh_frac_box);

            // compute distance between particle and cell center in fractional coordinates using minimum image
            Scalar3 dx = global_box.minImage(neigh_pos - pos);
            Scalar3 dx_frac_box = global_box.makeFraction(dx) - make_scalar3(0.5,0.5,0.5);
            Scalar3 dx_frac = dx_frac_box*make_scalar3(m_mesh_points.x, m_mesh_points.y, m_mesh_points.z);

            // compute fraction of particle density assigned to cell
            Scalar density_fraction = assignTSC(dx_frac.x)*assignTSC(dx_frac.y)*assignTSC(dx_frac.z)/V_cell;
            unsigned int cell_idx = cell_coord.z + m_mesh_points.z * cell_coord.y
                                    + m_mesh_points.y * m_mesh_points.z * cell_coord.x;
            h_mesh.data[cell_idx].r += h_mode.data[type]*density_fraction;
            }
             
        }  // end of loop over particles

    if (m_prof) m_prof->pop();
    }

void OrderParameterMesh::updateMeshes()
    {
    if (m_prof) m_prof->push("FFT");

    ArrayHandle<kiss_fft_cpx> h_mesh(m_mesh, access_location::host, access_mode::read);
    ArrayHandle<kiss_fft_cpx> h_fourier_mesh(m_fourier_mesh, access_location::host, access_mode::overwrite);
    ArrayHandle<kiss_fft_cpx> h_fourier_mesh_G(m_fourier_mesh_G, access_location::host, access_mode::overwrite);

    ArrayHandle<kiss_fft_cpx> h_fourier_mesh_x(m_fourier_mesh_x, access_location::host, access_mode::overwrite);
    ArrayHandle<kiss_fft_cpx> h_fourier_mesh_y(m_fourier_mesh_y, access_location::host, access_mode::overwrite);
    ArrayHandle<kiss_fft_cpx> h_fourier_mesh_z(m_fourier_mesh_z, access_location::host, access_mode::overwrite);

    ArrayHandle<kiss_fft_cpx> h_force_mesh_x(m_force_mesh_x, access_location::host, access_mode::overwrite);
    ArrayHandle<kiss_fft_cpx> h_force_mesh_y(m_force_mesh_y, access_location::host, access_mode::overwrite);
    ArrayHandle<kiss_fft_cpx> h_force_mesh_z(m_force_mesh_z, access_location::host, access_mode::overwrite);

 
    ArrayHandle<Scalar> h_inf_f(m_inf_f, access_location::host, access_mode::read);
    ArrayHandle<Scalar3> h_k(m_k, access_location::host, access_mode::read);

    // transform the particle mesh
    kiss_fftnd(m_kiss_fft, h_mesh.data, h_fourier_mesh.data);

    Scalar V = m_pdata->getGlobalBox().getVolume();
    Scalar V_cell = V / (Scalar)m_mesh_index.getNumElements();

    // multiply with influence function
    for (unsigned int k = 0; k < m_mesh_index.getNumElements(); ++k)
        {
        h_fourier_mesh.data[k].r *= V_cell;
        h_fourier_mesh.data[k].i *= V_cell;

        h_fourier_mesh_G.data[k].r =h_fourier_mesh.data[k].r * h_inf_f.data[k];
        h_fourier_mesh_G.data[k].i =h_fourier_mesh.data[k].i * h_inf_f.data[k];

        Scalar3 kval = h_k.data[k];
        h_fourier_mesh_x.data[k].r = -h_fourier_mesh_G.data[k].i*kval.x;
        h_fourier_mesh_x.data[k].i = h_fourier_mesh_G.data[k].r*kval.x;

        h_fourier_mesh_y.data[k].r = -h_fourier_mesh_G.data[k].i*kval.y;
        h_fourier_mesh_y.data[k].i = h_fourier_mesh_G.data[k].r*kval.y;

        h_fourier_mesh_z.data[k].r = -h_fourier_mesh_G.data[k].i*kval.z;
        h_fourier_mesh_z.data[k].i = h_fourier_mesh_G.data[k].r*kval.z;
        }

    // Inverse transform force on mesh points 
    kiss_fftnd(m_kiss_ifft_x, h_fourier_mesh_x.data, h_force_mesh_x.data);
    kiss_fftnd(m_kiss_ifft_y, h_fourier_mesh_y.data, h_force_mesh_y.data);
    kiss_fftnd(m_kiss_ifft_z, h_fourier_mesh_z.data, h_force_mesh_z.data);

    if (m_prof) m_prof->pop();
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

    const BoxDim& global_box = m_pdata->getGlobalBox();
    const Scalar V = global_box.getVolume();
 
    // inverse dimensions
    Scalar3 dim_inv = make_scalar3(Scalar(1.0)/(Scalar)m_mesh_points.x,
                                   Scalar(1.0)/(Scalar)m_mesh_points.y,
                                   Scalar(1.0)/(Scalar)m_mesh_points.z); 

    for (unsigned int idx = 0; idx < m_pdata->getN(); ++idx)
        {
        Scalar4 postype = h_postype.data[idx];

        Scalar3 pos = make_scalar3(postype.x, postype.y, postype.z);
        unsigned int type = __scalar_as_int(postype.w);

        // compute coordinates in units of the mesh size
        Scalar3 f = global_box.makeFraction(pos);
        Scalar3 reduced_pos = make_scalar3(f.x * (Scalar) m_mesh_points.x,
                                           f.y * (Scalar) m_mesh_points.y,
                                           f.z * (Scalar) m_mesh_points.z);

        // find cell the particle is in
        unsigned int ix = reduced_pos.x;
        unsigned int iy = reduced_pos.y;
        unsigned int iz = reduced_pos.z;

        // handle particles on the boundary
        if (ix == m_mesh_points.x)
            ix = 0;
        if (iy == m_mesh_points.y)
            iy = 0;
        if (iz == m_mesh_points.z)
            iz = 0;

        // center of cell (in units of the mesh size)
        unsigned int my_cell = m_mesh_index(ix,iy,iz);
        
        Scalar3 force = make_scalar3(0.0,0.0,0.0);

        // interpolate mesh forces from cell and next neighbors
        for (unsigned int k = 0; k < m_cell_adj_indexer.getW(); ++k)
            {
            unsigned int neigh_cell = h_cell_adj.data[m_cell_adj_indexer(k, my_cell)];

            uint3 cell_coord = m_mesh_index.getTriple(neigh_cell); 

            Scalar3 neigh_frac = make_scalar3((Scalar) cell_coord.x + Scalar(0.5),
                                             (Scalar) cell_coord.y + Scalar(0.5),
                                             (Scalar) cell_coord.z + Scalar(0.5));
           
            // coordinates of the neighboring cell between 0..1 
            Scalar3 neigh_frac_box = neigh_frac * dim_inv;
            Scalar3 neigh_pos = global_box.makeCoordinates(neigh_frac_box);

            // compute distance between particle and cell center in fractional coordinates using minimum image
            Scalar3 dx = global_box.minImage(neigh_pos - pos);
            Scalar3 dx_frac_box = global_box.makeFraction(dx) - make_scalar3(0.5,0.5,0.5);
            Scalar3 dx_frac = dx_frac_box*make_scalar3(m_mesh_points.x, m_mesh_points.y, m_mesh_points.z);

            unsigned int cell_idx = cell_coord.z + m_mesh_points.z * cell_coord.y
                                    + m_mesh_points.y * m_mesh_points.z * cell_coord.x;

            force += -assignTSC(dx_frac.x)*assignTSC(dx_frac.y)*assignTSC(dx_frac.z)*h_mode.data[type]
                            *make_scalar3(h_force_mesh_x.data[cell_idx].r,
                                          h_force_mesh_y.data[cell_idx].r,
                                          h_force_mesh_z.data[cell_idx].r)/V;
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

    Scalar V = m_pdata->getGlobalBox().getVolume();

    for (unsigned int k = 0; k < m_mesh.getNumElements(); ++k)
        sum += h_fourier_mesh_G.data[k].r * h_fourier_mesh.data[k].r
             + h_fourier_mesh_G.data[k].i * h_fourier_mesh.data[k].i;

    sum *= Scalar(1.0/2.0)/V;

    if (m_prof) m_prof->pop();

    return sum - m_E_self;
    }

Scalar OrderParameterMesh::getCurrentValue(unsigned int timestep)
    {
    if (m_prof) m_prof->push("Mesh");

    if (m_is_first_step)
        {
        // allocate memory and initialize arrays
        setupMesh();

        computeInfluenceFunction();
        m_is_first_step = false;
        }

    assignParticles();

    updateMeshes();

    Scalar val = computeCV();

    m_cv_last_updated = timestep;

    if (m_prof) m_prof->pop();

    return val;
    }
   
void OrderParameterMesh::computeForces(unsigned int timestep)
    {

    if (m_is_first_step || m_cv_last_updated != timestep)
        getCurrentValue(timestep);

    if (m_prof) m_prof->push("Mesh");

    interpolateForces();

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
