#ifndef __COLLECTIVE_CALLBACK_H__
#define __COLLECTIVE_CALLBACK_H__

#include "CollectiveVariable.h"

#include <pybind11/pybind11.h>

#include <hoomd/GlobalArray.h>

/*
 * A CollectiveVariable that gets its values from a python callback
 */

class CollectiveCallback : public CollectiveVariable
    {
    public:
        /*! Constructs the collective variable
            \param sysdef The system definition
            \param name The name of this collective variable
            \param cb The python callback object
         */
        CollectiveCallback(std::shared_ptr<SystemDefinition> sysdef, const std::string& name, pybind11::object cb)
            : CollectiveVariable(sysdef, name), m_cb(cb)
            { }

        virtual ~CollectiveCallback() {}

        /*! Returns the current value of the collective variable
         *  \param timestep The currnt value of the timestep
         */
        virtual Scalar getCurrentValue(unsigned int timestep)
            {
            Scalar cv(0.0);
            if (!m_cb.is(pybind11::none()))
                {
                pybind11::object rv = m_cb(timestep);
                try
                    {
                    cv = pybind11::cast<Scalar>(rv);
                    }
                catch (const std::exception& e)
                    {
                    throw std::runtime_error("Expected a scalar CV value as return value.");
                    }
                }
            return cv;
            }

    protected:
        pybind11::object m_cb; //!< The python callback

        virtual void computeBiasForces(unsigned int timestep)
            {
            // noop
            }
    };

//! Export the CollectiveVariable class to python
inline void export_CollectiveCallback(pybind11::module& m)
    {
    pybind11::class_<CollectiveCallback, std::shared_ptr<CollectiveCallback> > (m, "CollectiveCallback", pybind11::base<CollectiveVariable>() )
            .def(pybind11::init< std::shared_ptr<SystemDefinition>, const std::string&, pybind11::object > ())
            ;

    }
#endif // __COLLECTIVE_CALLBACK_H__
