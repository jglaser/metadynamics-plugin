#ifndef __WELL_TEMPERED_ENSEMBLE_H__
#define __WELL_TEMPERED_ENSEMBLE_H__

/*! \file CollectiveVariable.h
    \brief Declares the CollectiveVariable abstract class
 */

#include "CollectiveVariable.h"

/*! Class to implement the potential energy as a collective variable (Well-tempered Ensemble)

    see Bonomi, Parrinello PRL 104:190601 (2010)
*/

class WellTemperedEnsemble : public CollectiveVariable
    {
    public:
        /*! Constructs the collective variable
            \param sysdef The system definition
            \param name The name of this collective variable
         */
        WellTemperedEnsemble(std::shared_ptr<SystemDefinition> sysdef, const std::string& name);
        virtual ~WellTemperedEnsemble() {}

        /*! Requires the evaluation of other variables first */
        virtual bool requiresNetForce()
            {
            return true;
            }

        /*! Returns the names of provided log quantities.
         */
        virtual std::vector<std::string> getProvidedLogQuantities()
            {
            std::vector<std::string> list = CollectiveVariable::getProvidedLogQuantities();
            list.push_back(m_log_name);
            return list;
            }

        /*! Returns the current value of the collective variable
         *  \param timestep The currnt value of the timestep
         */
        virtual Scalar getCurrentValue(unsigned int timestep)
            {
            computeCV(timestep);
            return m_pe;
            }

        /*! Returns the value of a specific log quantity.
         * \param quantity The name of the quantity to return the value of
         * \param timestep The current value of the time step
         */
        Scalar getLogValue(const std::string& quantity, unsigned int timestep)
            {
            if (quantity == m_log_name)
                {
                computeCV(timestep);
                return m_pe;
                }

            // nothing found, turn to base class
            return CollectiveVariable::getLogValue(quantity, timestep);
            }

    protected:
        Scalar m_pe;                //!< The potential energy
        std::string m_log_name;     //!< Name of log quantity

        virtual void computeCV(unsigned int timestep);

        /*! Compute the biased forces for this collective variable.
            The force that is written to the force arrays must be
            multiplied by the bias factor.

            \param timestep The current value of the time step
         */
        virtual void computeBiasForces(unsigned int timestep);
    };

//! Export the CollectiveVariable class to python
void export_WellTemperedEnsemble(pybind11::module& m);

#endif // __WELL_TEMPERED_ENSEMBLE_H__
