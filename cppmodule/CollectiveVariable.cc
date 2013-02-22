/*! \file CollectiveVariable.cc
    \brief Partially implements the CollectiveVariable class
 */
#include <boost/python.hpp>

#include "CollectiveVariable.h"

using namespace boost::python;

CollectiveVariable::CollectiveVariable(boost::shared_ptr<SystemDefinition> sysdef,
                                       const std::string& name)
    : ForceCompute(sysdef),
      m_bias(0.0),
      m_cv_name(name),
      m_umbrella(no_umbrella),
      m_cv0(0.0),
      m_kappa(0.0),
      m_width_flat(0.0)
    {
    }

void CollectiveVariable::computeForces(unsigned int timestep)
    {
    if (m_umbrella != no_umbrella)
        {
        Scalar val = getCurrentValue(timestep);
        if ((val < m_cv0 + m_width_flat/Scalar(2.0)) && 
            (val > m_cv0 - m_width_flat/Scalar(2.0)))
            m_bias = Scalar(0.0);
        else
            {
            Scalar delta(0.0);
            if (val > m_cv0)
                delta = val - m_cv0 - m_width_flat/Scalar(2.0);
            else 
                delta = val - m_cv0 + m_width_flat/Scalar(2.0);

            if (m_umbrella == harmonic)
                {
                m_bias = delta/m_kappa/m_kappa;
                }
            else if (m_umbrella == wall)
                {
                m_bias = Scalar(12.0)*pow(delta/m_kappa,Scalar(11.0))/m_kappa;
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
            return Scalar(0.0);
        else
            {
            Scalar delta(0.0);
            if (val > m_cv0)
                delta = val - m_cv0 - m_width_flat/Scalar(2.0);
            else if (val < m_cv0)
                delta = val - m_cv0 + m_width_flat/Scalar(2.0);

            if (m_umbrella == harmonic)
                {
                return Scalar(1.0/2.0)*delta*delta/m_kappa/m_kappa;
                }
            else if (m_umbrella == wall)
                {
                return pow(delta/m_kappa,Scalar(12.0));
                }
            }
        }

    return Scalar(0.0);
    }

//! Wrapper for abstract class CollectiveVariable
class CollectiveVariableWrap : public CollectiveVariable, public wrapper<CollectiveVariable>
    {
    public:
        //! Constructs a CollectiveVariableWrap
        /*! \param sysdef The system definition
            \param name Name of the collective variable
         */
        CollectiveVariableWrap(boost::shared_ptr<SystemDefinition> sysdef,
                               const std::string& name) 
            : CollectiveVariable(sysdef, name)
            { }

        /*! Compute the forces
            \param timestep Current value of the timestep
         */
        void computeBiasForces(unsigned int timestep)
            {
            this->get_override("computeBiasForces")(timestep);
            }

        /*! Returns the current value of the collective variable
            \param timestep Current value of the timestep
         */
        Scalar getCurrentValue(unsigned int timestep)
            {
            return this->get_override("getCurrentValue")(timestep);
            }
    };              

void export_CollectiveVariable()
    {
    class_<CollectiveVariableWrap, boost::shared_ptr<CollectiveVariableWrap>, bases<ForceCompute>, boost::noncopyable>
        ("CollectiveVariable", init< boost::shared_ptr<SystemDefinition>, const std::string& > ())
        .def("getCurrentValue", pure_virtual(&CollectiveVariable::getCurrentValue))
        .def("setUmbrella", &CollectiveVariable::setUmbrella)
        .def("setKappa", &CollectiveVariable::setKappa)
        .def("setWidthFlat", &CollectiveVariable::setWidthFlat)
        .def("setMinimum", &CollectiveVariable::setMinimum)
        ;

    enum_<CollectiveVariableWrap::umbrella_Enum>("umbrella")
    .value("no_umbrella", CollectiveVariableWrap::no_umbrella)
    .value("harmonic", CollectiveVariableWrap::harmonic)
    .value("wall", CollectiveVariableWrap::wall)
    ;
    }
