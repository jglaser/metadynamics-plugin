/*! \file WellTemperedEnsemble.cc
    \brief Implements the potential energy as a collective variable (well-tempered ensemble)
 */
#include "WellTemperedEnsemble.h"

namespace py = pybind11;

WellTemperedEnsemble::WellTemperedEnsemble(std::shared_ptr<SystemDefinition> sysdef,
                                       const std::string& name)
    : CollectiveVariable(sysdef, name), m_pe(0.0), m_log_name("cv_potential_energy")
    {
    }

void WellTemperedEnsemble::computeCV(unsigned int timestep)
    {
    if (m_prof)
        m_prof->push("Well-Tempered Ensemble");

    ArrayHandle<Scalar4> h_net_force(m_pdata->getNetForce(), access_location::host, access_mode::read);

    unsigned int N = m_pdata->getN();

    m_pe = Scalar(0.0);

    // Sum up potential energy
    for (unsigned int i = 0; i < N; ++i)
        {
        m_pe += h_net_force.data[i].w;
        }

    m_pe += m_pdata->getExternalEnergy();

#ifdef ENABLE_MPI
    // reduce Fourier modes on on all processors
    if (m_pdata->getDomainDecomposition())
        {
        MPI_Allreduce(MPI_IN_PLACE, &m_pe, 1, MPI_HOOMD_SCALAR, MPI_SUM, m_exec_conf->getMPICommunicator());
        }
#endif

    if (m_prof)
        m_prof->pop();
    }

void WellTemperedEnsemble::computeBiasForces(unsigned int timestep)
    {
    if (m_prof)
        m_prof->push("Well-Tempered Ensemble");

    // Note: this Compute operates directly on the net force, therefore it needs to be called
    // after every other force
    ArrayHandle<Scalar4> h_net_force(m_pdata->getNetForce(), access_location::host, access_mode::readwrite);

    unsigned int N = m_pdata->getN();

    // apply bias factor
    Scalar fac = Scalar(1.0)+m_bias;
    for (unsigned int i = 0; i < N; ++i)
        {
        h_net_force.data[i].x *= fac;
        h_net_force.data[i].y *= fac;
        h_net_force.data[i].z *= fac;
        h_net_force.data[i].w *= fac;
        }

    if (m_prof)
        m_prof->pop();
    }


void export_WellTemperedEnsemble(py::module& m)
    {
    py::class_<WellTemperedEnsemble, std::shared_ptr<WellTemperedEnsemble> > (m, "WellTemperedEnsemble", py::base<CollectiveVariable>() )
        .def(py::init< std::shared_ptr<SystemDefinition>, const std::string& > ())
        ;

    }
