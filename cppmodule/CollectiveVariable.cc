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
      m_kappa(0.0)
    {
    }

void CollectiveVariable::computeForces(unsigned int timestep)
    {
    if (m_umbrella == harmonic)
        {
        Scalar val = getCurrentValue(timestep);
        
        m_bias = (val-m_cv0)*m_kappa;
        }
    else if (m_umbrella == wall)
        {
        Scalar val = getCurrentValue(timestep);

        m_bias = Scalar(12.0)*m_kappa*pow(val-m_cv0,Scalar(11.0));
        }

    computeBiasForces(timestep);
    }

Scalar CollectiveVariable::getUmbrellaPotential(unsigned int timestep)
    {
    Scalar val = getCurrentValue(timestep);

    if (m_umbrella == harmonic)
        return Scalar(1.0/2.0)*m_kappa*(val-m_cv0)*(val-m_cv0);
    else if (m_umbrella == wall)
        return m_kappa*pow(val-m_cv0,Scalar(12.0));

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
        .def("setMinimum", &CollectiveVariable::setMinimum)
        ;

    enum_<CollectiveVariableWrap::umbrella_Enum>("umbrella")
    .value("no_umbrella", CollectiveVariableWrap::no_umbrella)
    .value("harmonic", CollectiveVariableWrap::harmonic)
    .value("wall", CollectiveVariableWrap::wall)
    ;
    }
