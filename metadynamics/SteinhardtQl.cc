/*! \file SteinhardtQl.cc
    \brief Implements the Steinhardt Ql order parameter
 */

#include "SteinhardtQl.h"
#include "spherical_harmonics.hpp"

#include <cmath>

namespace py = pybind11;

SteinhardtQl::SteinhardtQl(std::shared_ptr<SystemDefinition> sysdef,
            Scalar rcut, Scalar ron, unsigned int lmax,
            std::shared_ptr<NeighborList> nlist, unsigned int type,
            const std::vector<Scalar>& Ql_ref,
            const std::string& log_suffix)

    : CollectiveVariable(sysdef,"steinhardt"+log_suffix), m_rcutsq(rcut*rcut), m_ronsq(ron*ron),
        m_lmax(lmax), m_nlist(nlist), m_type(type), m_cv_last_updated(0), m_have_computed(false), m_value(0.0)
    {
    m_prof_name = "steinhardt_Ql"+log_suffix;

    m_Ql = std::vector<Scalar>(m_lmax+1, Scalar(0.0));

    if (Ql_ref.size() != m_lmax+1)
        {
        m_exec_conf->msg->error() << "List of reference Qlm needs to be exactly of length lmax = " << m_lmax+1 << std::endl;
        throw std::runtime_error("Error setting up Steinhardt CV");
        }

    m_Ql_ref.resize(Ql_ref.size());

    std::copy(Ql_ref.begin(), Ql_ref.end(), m_Ql_ref.begin());
    }

inline Scalar fSmooth(Scalar r_onsq, Scalar r_cutsq, Scalar rsq)
    {
    if (rsq <= r_onsq)
        return 1.0;
    if (rsq > r_cutsq)
        return 0.0;

    Scalar r = sqrt(rsq);
    Scalar r_on = sqrt(r_onsq);
    Scalar r_cut = sqrt(r_cutsq);

    return Scalar(0.5)*(cos(M_PI*(r-r_on)/(r_cut-r_on))+1);
    }

inline Scalar fprimeSmooth_divr(Scalar r_onsq, Scalar r_cutsq, Scalar rsq)
    {
    if (rsq <= r_onsq || rsq > r_cutsq)
        return 0.0;

    Scalar r = sqrt(rsq);
    Scalar r_on = sqrt(r_onsq);
    Scalar r_cut = sqrt(r_cutsq);

    return -Scalar(0.5*M_PI)/r/(r_cut-r_on)*sin(M_PI*(r-r_on)/(r_cut-r_on));
    }

void SteinhardtQl::computeCV(unsigned int timestep)
    {
    if (m_cv_last_updated == timestep && m_have_computed)
        return;

    // start by updating the neighborlist
    m_nlist->compute(timestep);

    // start the profile for this compute
    if (m_prof) m_prof->push(m_prof_name);

    if (m_prof) m_prof->push("CV");

    // depending on the neighborlist settings, we can take advantage of newton's third law
    // to reduce computations at the cost of memory access complexity: set that flag now
    bool third_law = m_nlist->getStorageMode() == NeighborList::half;

    // access the neighbor list, particle data, and system box
    ArrayHandle<unsigned int> h_n_neigh(m_nlist->getNNeighArray(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_nlist(m_nlist->getNListArray(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_head_list(m_nlist->getHeadList(), access_location::host, access_mode::read);

    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);

    const BoxDim& box = m_pdata->getBox();

    unsigned int sph_count = (m_lmax+1)*(m_lmax+1);
    std::vector<std::complex<Scalar> > Ylm_pp(sph_count,std::complex<Scalar>(0.0,0.0));
    m_Qlm.resize(sph_count);
    std::fill(m_Qlm.begin(), m_Qlm.end(), std::complex<Scalar>(0.0,0.0));

    // for each particle
    unsigned int N = m_pdata->getN();

    for (int i = 0; i < (int)N; i++)
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
                Scalar f = fSmooth(m_ronsq, m_rcutsq, rsq);

                // compute theta, phi
                Scalar theta = acos(dx.z/sqrt(rsq));
                Scalar phi = atan2(dx.y, dx.x);

                bool negative_m = true;
                // note switching theta and phi due to diffferent convention
                fsph::evaluate_SPH<Scalar>(&Ylm_pp.front(), m_lmax, &theta, &phi, 1, negative_m);

                int n = 0;
                for (int l = 0; l <= (int)m_lmax; ++l)
                    {
                    for (int p = 0; p < 2*l+1; ++p)
                        {
                        int m = (p <= l) ? p : (l-p);
                        int phase = (m > 0 && m % 2) ? -1 : 1; // Condon-Shortley
                        m_Qlm[n] += std::complex<Scalar>(phase)*Ylm_pp[n]*f;
                        n++;
                        }
                    }
                }
            }

        }

    // need to reduce Qlm in MPI here
    // ...

    unsigned int Nglobal = m_pdata->getNGlobal();
    unsigned int nc = 1; // for now

    unsigned int n = 0;
    for (int l = 0; l <= (int)m_lmax; ++l)
        {
        m_Ql[l] = Scalar(0.0);
        for (int p=0; p < 2*l+1; ++p)
            {
            if (third_law)
                {
                if (l % 2 == 0)
                    m_Qlm[n] *= 2; // assume even parity
                else
                    m_Qlm[n] = std::complex<Scalar>(0.0,0.0);
                }

            Scalar Qlm_sq = (std::conj(m_Qlm[n])*m_Qlm[n]).real();
            Qlm_sq *= Scalar(4.0*M_PI/(2*l+1))/(Nglobal*Nglobal*nc*nc);
            m_Ql[l] += Qlm_sq;

            n++;
            }
        }

    // compute collective variable as normalizd dot product
    m_value = Scalar(0.0);
    for (unsigned int l = 0; l <= m_lmax; ++l)
        {
        m_value += m_Ql_ref[l]*m_Ql[l];
        }

    m_have_computed = true;
    m_cv_last_updated = timestep;

    if (m_prof) m_prof->pop();
    if (m_prof) m_prof->pop();
    }

void SteinhardtQl::computeBiasForces(unsigned int timestep)
    {
    // start by updating the neighborlist
    m_nlist->compute(timestep);

    // start the profile for this compute
    if (m_prof) m_prof->push(m_prof_name);

    if (m_prof) m_prof->push("Force");

    // depending on the neighborlist settings, we can take advantage of newton's third law
    // to reduce computations at the cost of memory access complexity: set that flag now
    bool third_law = m_nlist->getStorageMode() == NeighborList::half;

    // access the force array
    ArrayHandle<Scalar4> h_force(m_force, access_location::host, access_mode::overwrite);

    // access the neighbor list, particle data, and system box
    ArrayHandle<unsigned int> h_n_neigh(m_nlist->getNNeighArray(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_nlist(m_nlist->getNListArray(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_head_list(m_nlist->getHeadList(), access_location::host, access_mode::read);

    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);

    const BoxDim& box = m_pdata->getBox();

    unsigned int sph_count = (m_lmax+1)*(m_lmax+1); //l = 0..lmax
    std::vector<std::complex<Scalar> > Ylm_pp(sph_count,std::complex<Scalar>(0.0,0.0));

    unsigned int Nglobal = m_pdata->getNGlobal();
    unsigned int nc = 1; // for now

    // reset force
    memset(h_force.data, 0, sizeof(Scalar4)*m_force.getNumElements());

    // for each particle
    unsigned int N = m_pdata->getN();

    for (int i = 0; i < (int)N; i++)
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

            vec3<Scalar> force(0.0,0.0,0.0);

            if (rsq <= m_rcutsq)
                {
                // compute theta, phi
                std::complex<Scalar> r(sqrt(rsq));
                Scalar theta = acos(dx.z/r.real());
                Scalar phi = atan2(dx.y, dx.x);

                vec3<std::complex<Scalar> > e_theta = vec3<std::complex<Scalar> >(cos(theta)*cos(phi),cos(theta)*sin(phi),-sin(theta));
                vec3<std::complex<Scalar> > e_phi = vec3<std::complex<Scalar> >(-sin(phi),cos(phi),0.0);

                bool negative_m = true;
                // note switching theta and phi due to diffferent convention
                fsph::evaluate_SPH<Scalar>(&Ylm_pp.front(), m_lmax, &theta, &phi, 1, negative_m);

                std::complex<Scalar> fprime_divr(fprimeSmooth_divr(m_ronsq,m_rcutsq, rsq));
                std::complex<Scalar> f(fSmooth(m_ronsq, m_rcutsq, rsq));
                int n = 0;
                for (int l = 0; l <= (int)m_lmax; ++l)
                    {
                    vec3<Scalar> del_Ql_i(0.0,0.0,0.0);
                    for (int p = 0; p < 2*l+1; ++p)
                        {
                        int m = (p <= l) ? p : (l-p);
                        int phase = (m > 0 && m % 2) ? -1 : 1; // Condon-Shortley phase, for derivative formula
                        std::complex<Scalar> Ylm = std::complex<Scalar>(phase)*Ylm_pp[n];
                        std::complex<Scalar> dYlm_dtheta = std::complex<Scalar>(m/tan(theta))*Ylm;
                        if (m < l)
                            {
                            unsigned int m_plus_one = (m < 0) ? (m == -1 ? n-p : n-1) : (n+1);
                            int phase_plus_one = (m + 1 > 0 && (m+1) %2 ) ? -1 : 1;
                            dYlm_dtheta += std::complex<Scalar>(phase_plus_one*sqrt((l-m)*(l+m+1)))*std::exp(std::complex<Scalar>(0,-phi))*Ylm_pp[m_plus_one];
                            }
                        std::complex<Scalar> dYlm_dphi = std::complex<Scalar>(0,m)*Ylm;

                        vec3<std::complex<Scalar> > del_Qlm = vec3<std::complex<Scalar> >(dx)*fprime_divr*Ylm + f/r*e_theta*dYlm_dtheta + f*e_phi/(r*std::complex<Scalar>(sin(theta)))*dYlm_dphi;
                        del_Qlm *= std::conj(m_Qlm[n]);
                        del_Ql_i += Scalar(2.0)*vec3<Scalar>(del_Qlm.x.real(),del_Qlm.y.real(), del_Qlm.z.real());
                        n++;
                        }
                    del_Ql_i *= Scalar(4.0*M_PI/(2*l+1))/(Nglobal*Nglobal)/(nc*nc);

                    force -= m_bias*del_Ql_i*m_Ql_ref[l];
                    }
                }
            h_force.data[i].x += force.x;
            h_force.data[i].y += force.y;
            h_force.data[i].z += force.z;

            if (third_law && j < N)
                {
                h_force.data[j].x -= force.x;
                h_force.data[j].y -= force.y;
                h_force.data[j].z -= force.z;
                }
            }
        }

    if (m_prof) m_prof->pop();
    if (m_prof) m_prof->pop();
    }

void export_SteinhardtQl(py::module& m)
    {
    py::class_<SteinhardtQl, std::shared_ptr<SteinhardtQl> > steinhardt(m, "SteinhardtQl", py::base<CollectiveVariable>() );
    steinhardt.def(py::init< std::shared_ptr<SystemDefinition>, Scalar, Scalar, unsigned int, std::shared_ptr<NeighborList>, unsigned int,
        const std::vector<Scalar>&,  const std::string& > ())
        ;
    }
