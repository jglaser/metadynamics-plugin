/*! \file SteinhardtQl.cc
    \brief Implements the Steinhardt Ql order parameter
 */

#include "SteinhardtQl.h"

#include "spherical_harmonics.hpp"

namespace py = pybind11;

SteinhardtQl::SteinhardtQl(std::shared_ptr<SystemDefinition> sysdef,
            Scalar rcut, unsigned int lmax, std::shared_ptr<NeighborList> nlist,
            unsigned int type,
            const std::string& log_suffix)

    : CollectiveVariable(sysdef,"cv_steinhardt"+log_suffix), m_rcutsq(rcut*rcut), m_lmax(lmax), m_nlist(nlist),
        m_type(type), m_cv_last_updated(0), m_have_computed(false), m_value(0.0)
    {
    m_prof_name = "steinhardt_ql"+log_suffix;

    m_Ql = std::vector<Scalar>(m_lmax, Scalar(0.0));
    }

void SteinhardtQl::computeBiasForces(unsigned int timestep)
    { }

void SteinhardtQl::computeCV(unsigned int timestep)
    {
    if (m_cv_last_updated == timestep && m_have_computed)
        return;

    // start by updating the neighborlist
    m_nlist->compute(timestep);

    // start the profile for this compute
    if (m_prof) m_prof->push(m_prof_name);

    // depending on the neighborlist settings, we can take advantage of newton's third law
    // to reduce computations at the cost of memory access complexity: set that flag now
    bool third_law = m_nlist->getStorageMode() == NeighborList::half;

    // access the neighbor list, particle data, and system box
    ArrayHandle<unsigned int> h_n_neigh(m_nlist->getNNeighArray(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_nlist(m_nlist->getNListArray(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_head_list(m_nlist->getHeadList(), access_location::host, access_mode::read);

    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);

    const BoxDim& box = m_pdata->getBox();

    unsigned int sph_count = ((m_lmax+1)*(m_lmax+2))/2 + (m_lmax*(m_lmax+1))/2;
    std::vector<std::complex<Scalar> > Ylm_pp(sph_count,std::complex<Scalar>(0.0,0.0));
    std::vector<std::complex<Scalar> > Ylm(sph_count,std::complex<Scalar>(0.0,0.0));

    // for each particle
    for (int i = 0; i < (int)m_pdata->getN(); i++)
        {
        // access the particle's position and type (MEM TRANSFER: 4 scalars)
        Scalar3 pi = make_scalar3(h_pos.data[i].x, h_pos.data[i].y, h_pos.data[i].z);
        unsigned int typei = __scalar_as_int(h_pos.data[i].w);

        // sanity check
        assert(typei < m_pdata->getNTypes());

        // only compute for a single particle type
        if (typei != m_type) continue;

        // loop over all of the neighbors of this particle
        const unsigned int myHead = h_head_list.data[i];
        const unsigned int size = (unsigned int)h_n_neigh.data[i];
        for (unsigned int k = 0; k < size; k++)
            {
            // access the index of this neighbor (MEM TRANSFER: 1 scalar)
            unsigned int j = h_nlist.data[myHead + k];
            assert(j < m_pdata->getN() + m_pdata->getNGhosts());

            // calculate dr_ji (MEM TRANSFER: 3 scalars / FLOPS: 3)
            Scalar3 pj = make_scalar3(h_pos.data[j].x, h_pos.data[j].y, h_pos.data[j].z);
            Scalar3 dx = pi - pj;

            // access the type of the neighbor particle (MEM TRANSFER: 1 scalar)
            unsigned int typej = __scalar_as_int(h_pos.data[j].w);
            assert(typej < m_pdata->getNTypes());

            if (typej != m_type) continue;

            // apply periodic boundary conditions
            dx = box.minImage(dx);

            // calculate r_ij squared (FLOPS: 5)
            Scalar rsq = dot(dx, dx);

            if (rsq <= m_rcutsq)
                {
                // compute theta, phi
                Scalar theta = acos(dx.y/sqrt(rsq));
                Scalar phi = atan2(dx.y, dx.x);

                bool negative_m = true;
                fsph::evaluate_SPH<Scalar>(&Ylm_pp.front(), m_lmax, &phi, &theta, 1, negative_m);

                // accumulate
                for (unsigned int n = 0; n < sph_count; ++n)
                    Ylm[n] += Ylm_pp[n];

                // need smoothing
                // .......
                }
            }

        }

    unsigned int Nglobal = m_pdata->getNGlobal();

    // need to reduce Qlm in MPI here
    // ...

    for (int l = 1; l <= (int)m_lmax; ++l)
        {
        m_Ql[l-1] = Scalar(0.0);
        for (int m = -l; m <= l; ++m)
            {
            unsigned int n = fsph::sphIndex(l,m);

            if (third_law)
                {
                if (l % 2 == 0)
                    Ylm[n] *= 2; // assume even parity
                else
                    Ylm[n] = std::complex<Scalar>(0.0,0.0);
                }

            Scalar Qlm = std::abs(Ylm[n])*std::abs(Ylm[n]);
            Qlm /= 4*Nglobal*Nglobal;
            m_Ql[l-1] += Scalar(4.0*M_PI/(2*l+1))*Qlm;
            }
        }

    m_have_computed = true;
    m_cv_last_updated = timestep;

    if (m_prof) m_prof->pop();
    }

void export_SteinhardtQl(py::module& m)
    {
    py::class_<SteinhardtQl, std::shared_ptr<SteinhardtQl> > steinhardt(m, "SteinhardtQl", py::base<CollectiveVariable>() );
    steinhardt.def(py::init< std::shared_ptr<SystemDefinition>, Scalar, unsigned int, std::shared_ptr<NeighborList>, unsigned int, const std::string& > ())
        ;
    }
