#ifndef __PATH_COLLECTIVE_VARIABLE_H__
#define __PATH_COLLECTIVE_VARIABLE_H__

#include <hoomd/hoomd.h>

#include "CollectiveVariable.h"

struct PathComponent
    {
    boost::shared_ptr<CollectiveVariable> m_cv;
    };

struct PathCollectiveVariable : public CollectiveVariable
    {
    enum Enum {
        parallel = 0,
        transverse
        };

    public:
        PathCollectiveVariable(boost::shared_ptr<SystemDefinition> sysdef,
                               Enum dir,
                               const unsigned int num_frames,
                               const Scalar scale,
                               const std::string name);
        virtual ~PathCollectiveVariable() {}

        void registerPathComponent(boost::shared_ptr<CollectiveVariable> cv,
                                   std::vector<Scalar> frames)
            {
            PathComponent component;
            
            component.m_cv = cv;

            m_path_components.push_back(component);
            
            if (frames.size() != m_num_frames)
                {
                m_exec_conf->msg->error() << "metadynamics.cv.path: Invalid number of frames for CV "
                                          << cv->getName() << "." << std::endl << std::endl;
                throw std::runtime_error("Error setting up path collective variable.");
                }
            m_frames.push_back(frames);
            }

        void removeAllPathComponents()
            {
            m_path_components.clear();
            m_frames.clear();
            }

        virtual Scalar getCurrentValue(unsigned int timestep);

        /*! Path collective variables do not have their own forces, so they
         *  cannot compute derivatives with respect to particle coordinates
         */
        virtual bool canComputeDerivatives()
            {
            return false;
            }

        /*! Returns the names of log quantitites provided
         */
        std::vector< std::string > getProvidedLogQuantities()
            {
            std::vector< std::string> ret;
            ret.push_back(m_log_name);
            return ret;
            }

        /*! Obtain the value of a specific log quantity
            \param quantity The quantity to obtain the value of
            \param timestep The current value of the time step
         */
        Scalar getLogValue(const std::string& quantity, unsigned int timestep)
            {
            // check if log quantity exists in base class
            if (quantity == m_log_name)
                {
                return m_cv;
                }
            else
                { 
                // default: throw exception
                std::cerr << std::endl << "***Error! " << quantity << " is not a valid log quantity for PathCollectiveVariable"
                          << std::endl << std::endl;
                throw std::runtime_error("Error getting log value");
                }
            }


    protected:
        virtual void computeBiasForces(unsigned int timestep)
            {
            // we do not compute our own forces
            }

    private:
        std::vector<PathComponent> m_path_components;  //!< List of path components (CVs)
        std::vector<std::vector<Scalar> > m_frames;    //!< Values of CV for every frame
        Enum m_direction;                              //!< Which direction of path CV to evaluate
        unsigned int m_num_frames;                     //!< Number of frames along the path
        Scalar m_scale;                                //!< Scale for distance from path
        unsigned int m_cv_last_updated;                //!< Last timestep of path CV update
        Scalar m_cv;                                   //!< Current value
        std::vector<Scalar> m_norm;                    //!< Distance metric of CVs for every frame
        std::string m_log_name;                        //!< Name of logging quantity
    };

void export_PathCollectiveVariable();
#endif
