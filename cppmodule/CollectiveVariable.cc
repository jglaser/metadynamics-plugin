/*! \file CollectiveVariable.cc
    \brief Partially implements the CollectiveVariable class
 */
#include <boost/python.hpp>

#include "CollectiveVariable.h"

using namespace boost::python;

CollectiveVariable::CollectiveVariable(boost::shared_ptr<SystemDefinition> sysdef,
                                       const std::string& name)
    : ForceCompute(sysdef), m_bias(0.0), m_cv_name(name), m_harmonic(false), m_cv0(0.0), m_kappa(0.0)
    {
    }

void CollectiveVariable::computeForces(unsigned int timestep)
    {
    if (m_harmonic)
        {
        Scalar val = getCurrentValue(timestep);
        
        m_bias = (val-m_cv0)*m_kappa;
        }
    computeBiasForces(timestep);
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
        .def("computeForces", &CollectiveVariable::computeForces)
        .def("getCurrentValue", pure_virtual(&CollectiveVariable::getCurrentValue))
        .def("setHarmonic", &CollectiveVariable::setHarmonic)
        .def("setKappa", &CollectiveVariable::setKappa)
        .def("setMinimum", &CollectiveVariable::setMinimum)
        ;
    }
