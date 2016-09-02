/*! \file CollectiveVariable.cc
    \brief Partially implements the CollectiveVariable class
 */

#include "CollectiveVariable.h"

namespace py = pybind11;

CollectiveVariable::CollectiveVariable(std::shared_ptr<SystemDefinition> sysdef,
                                       const std::string& name)
    : ForceCompute(sysdef),
      m_bias(0.0),
      m_cv_name(name),
      m_umbrella(no_umbrella),
      m_cv0(0.0),
      m_kappa(1.0),
      m_width_flat(0.0),
      m_scale(1.0)
    {
    }

void CollectiveVariable::computeForces(unsigned int timestep)
    {
    if (m_umbrella != no_umbrella)
        {
        Scalar val = getCurrentValue(timestep);
        if ((val < m_cv0 + m_width_flat/Scalar(2.0)) &&
            (val > m_cv0 - m_width_flat/Scalar(2.0)))
            {
            setBiasFactor(0.0);
            }
        else
            {
            Scalar delta(0.0);
            if (val > m_cv0)
                delta = val - m_cv0 - m_width_flat/Scalar(2.0);
            else
                delta = val - m_cv0 + m_width_flat/Scalar(2.0);

            if (m_umbrella == linear)
                {
                setBiasFactor(m_scale*Scalar(1.0));
                }
            else if (m_umbrella == harmonic)
                {
                setBiasFactor(m_kappa*delta);
                }
            else if (m_umbrella == wall)
                {
                setBiasFactor(m_scale*Scalar(12.0)*pow(delta/m_kappa,Scalar(11.0))/m_kappa);
                }
            else if (m_umbrella == gaussian)
                {
                setBiasFactor(-m_scale*(val-m_cv0)*exp(-(val-m_cv0)*(val-m_cv0)/m_kappa/m_kappa/Scalar(2.0)));
                }
            }
        }

    computeBiasForces(timestep);

    }

Scalar CollectiveVariable::getUmbrellaPotential(unsigned int timestep)
    {
    if (m_umbrella != no_umbrella)
        {
        Scalar val = getCurrentValue(timestep);
        if ((val < m_cv0 + m_width_flat/Scalar(2.0)) && 
            (val > m_cv0 - m_width_flat/Scalar(2.0)))
            {
            return Scalar(0.0);
            }
        else
            {
            Scalar delta(0.0);
            if (val > m_cv0)
                delta = val - m_cv0 - m_width_flat/Scalar(2.0);
            else if (val < m_cv0)
                delta = val - m_cv0 + m_width_flat/Scalar(2.0);

            if (m_umbrella == linear)
                {
                return m_scale*delta;
                }
            else if (m_umbrella == harmonic)
                {
                return Scalar(1.0/2.0)*delta*delta*m_kappa;
                }
            else if (m_umbrella == wall)
                {
                return m_scale*pow(delta/m_kappa,Scalar(12.0));
                }
            else if (m_umbrella == gaussian)
                {
                return m_scale*exp(-(val-m_cv0)*(val-m_cv0)/m_kappa/m_kappa/Scalar(2.0))-m_scale;
                }
            }
        }

    return Scalar(0.0);
    }


void export_CollectiveVariable(py::module& m)
    {
    py::class_<CollectiveVariable, std::shared_ptr<CollectiveVariable> > collective_variable(m, "CollectiveVariable", py::base<ForceCompute>() );
    collective_variable.def(py::init< std::shared_ptr<SystemDefinition>, const std::string& > ())
        .def("getCurrentValue", &CollectiveVariable::getCurrentValue)
        .def("setUmbrella", &CollectiveVariable::setUmbrella)
        .def("setKappa", &CollectiveVariable::setKappa)
        .def("setWidthFlat", &CollectiveVariable::setWidthFlat)
        .def("setMinimum", &CollectiveVariable::setMinimum)
        .def("setScale", &CollectiveVariable::setScale)
        .def("requiresNetForce", &CollectiveVariable::requiresNetForce)
        ;

    py::enum_<CollectiveVariable::umbrella_Enum>(collective_variable,"umbrella")
    .value("no_umbrella", CollectiveVariable::no_umbrella)
    .value("linear", CollectiveVariable::linear)
    .value("harmonic", CollectiveVariable::harmonic)
    .value("wall", CollectiveVariable::wall)
    .value("gaussian", CollectiveVariable::gaussian)
    .export_values()
    ;
    }
