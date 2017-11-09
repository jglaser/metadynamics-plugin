#include "Density.h"

namespace py = pybind11;

Density::Density(std::shared_ptr<SystemDefinition> sysdef,
    std::shared_ptr<ParticleGroup> group,
    const std::string& suffix)
    : CollectiveVariable(sysdef, "cv_density"+(suffix != "" ? "_"+suffix : "")), m_group(group)
    {
    // reset force and virial
    ArrayHandle<Scalar4> h_force(m_force, access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar> h_virial(m_virial, access_location::host, access_mode::overwrite);

    memset(h_force.data, 0, sizeof(Scalar4)*m_force.getNumElements());
    memset(h_virial.data, 0, sizeof(Scalar)*m_virial.getNumElements());

    m_log_name = m_cv_name;
    }

Scalar Density::getCurrentValue(unsigned int timestep)
    {
    Scalar V = m_pdata->getGlobalBox().getVolume(m_sysdef->getNDimensions()==2);
    unsigned int N = m_group->getNumMembersGlobal();

    Scalar rho = N/V;
    return rho;
    }

void Density::computeBiasForces(unsigned int timestep)
    {
    #ifdef ENABLE_MPI
    // only add contribution to external virial once (on rank zero)
    if (m_exec_conf->getRank()) return;
    #endif

    const BoxDim& global_box = m_pdata->getGlobalBox();
    Scalar V = global_box.getVolume(m_sysdef->getNDimensions()==2);
    unsigned int N = m_group->getNumMembersGlobal();

    Scalar Lx = global_box.getL().x;
    Scalar Ly = global_box.getL().y;
    Scalar Lz = global_box.getL().z;

    // derivative of rho w.r.t. V
    Scalar fac = -N/(V*V);

    // from Martyna Tobias Klein 1994 Eq. (2.20)
    m_external_virial[0] = - m_bias*fac*Lx*Ly*Lz;         // xx
    m_external_virial[1] = 0;
    m_external_virial[2] = 0;
    m_external_virial[3] = - m_bias*fac*Lx*Ly*Lz;         // yy
    m_external_virial[4] = 0;
    m_external_virial[5] = - m_bias*fac*Lx*Ly*Lz;         // zz
    }

void export_Density(py::module& m)
    {
    py::class_<Density, std::shared_ptr<Density> >(m, "Density", py::base<CollectiveVariable>() )
        .def(py::init< std::shared_ptr<SystemDefinition>,
                       std::shared_ptr<ParticleGroup>,
                       const std::string& >() );
        ;
    }

