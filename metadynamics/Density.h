#include "CollectiveVariable.h"

#include <hoomd/ParticleGroup.h>

class Density :  public CollectiveVariable
    {
    public:
        Density(std::shared_ptr<SystemDefinition> sysdef, std::shared_ptr<ParticleGroup> group,
            const std::string& suffix);
        virtual ~Density() {}

        virtual Scalar getCurrentValue(unsigned int timestep);

        /*! Returns the names of provided log quantities.
         */
        std::vector<std::string> getProvidedLogQuantities()
            {
            std::vector<std::string> list = CollectiveVariable::getProvidedLogQuantities();
            list.push_back(m_log_name);
            return list;
            }

        /*! Returns the value of a specific log quantity.
         * \param quantity The name of the quantity to return the value of
         * \param timestep The current value of the time step
         */
        Scalar getLogValue(const std::string& quantity, unsigned int timestep)
            {
            if (quantity == m_log_name)
                return getCurrentValue(timestep);

            return CollectiveVariable::getLogValue(quantity, timestep);
            }

        /*! Returns true if the collective variable can compute derivatives
         *  w.r.t. particle coordinates
         */
        virtual bool canComputeDerivatives()
            {
            return false;
            }

    private:
        /*! Compute the biased forces for this collective variable.
            The force that is written to the force arrays must be
            multiplied by the bias factor.

            \param timestep The current value of the time step
        */
        virtual void computeBiasForces(unsigned int timestep);

        std::string m_log_name; //!< The name of the collective variable
        std::shared_ptr<ParticleGroup> m_group; //!< The group to count the number of particles
    };

//! Export Density to Python
void export_Density(pybind11::module& m);
