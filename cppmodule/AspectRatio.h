#include "CollectiveVariable.h"

#include <hoomd/hoomd.h>

class AspectRatio :  public CollectiveVariable
    {
    public:
        AspectRatio(boost::shared_ptr<SystemDefinition> sysdef, const unsigned int dir1, unsigned int dir2);
        virtual ~AspectRatio() {}

        virtual void computeBiasForces(unsigned int timestep);

        virtual Scalar getCurrentValue(unsigned int timestep);

        /*! Returns the names of provided log quantities.
         */
        std::vector<std::string> getProvidedLogQuantities()
            {
            std::vector<std::string> list;
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
            else
                {
                m_exec_conf->msg->error() << "cv.aspect_ratio: " << quantity << " is not a valid log quantity." << std::endl;;
                throw std::runtime_error("Error getting log value");
                }
            }

    private:
        unsigned int m_dir1; //!< The cartesian index of the first direction
        unsigned int m_dir2; //!< The cartesian index of the second direction
        std::string m_log_name; //!< The name of the collective variable
    };

//! Export AspectRatio to Python
void export_AspectRatio();
