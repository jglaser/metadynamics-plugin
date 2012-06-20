#ifndef __INTEGRATOR_METADYNAMICS_H__
#define __INTEGRATOR_METADYNAMICS_H__

#include <hoomd/hoomd.h>

#include <boost/shared_ptr.hpp>

#include "CollectiveVariable.h"
#include "IndexGrid.h"

struct CollectiveVariableItem
    {
    boost::shared_ptr<CollectiveVariable> m_cv; //!< The collective variable
    Scalar m_sigma;                             //!< Width of compensating gaussians for this variable
    Scalar m_cv_min;                            //!< Minium value of collective variable (if using grid)
    Scalar m_cv_max;                            //!< Maximum value of collective variable (if using grid)
    Scalar m_num_points;                        //!< Number of grid points for this collective variable
    };

//! Implements a metadynamics update scheme
class IntegratorMetaDynamics : public IntegratorTwoStep
    {
    public:

        IntegratorMetaDynamics(boost::shared_ptr<SystemDefinition> sysdef,
                   Scalar deltaT,
                   Scalar W,
                   Scalar T_shift,
                   unsigned int stride,
                   bool add_hills = true,
                   const std::string& filename = "",
                   bool overwrite = false,
                   bool use_grid = false);

        virtual ~IntegratorMetaDynamics() {}

        //! Sample the collective variables and update their biasing coefficients
        virtual void update(unsigned int timestep);

        //! Prepare for the run
        virtual void prepRun(unsigned int timestep);

        virtual void printStats() {};

        void registerCollectiveVariable(boost::shared_ptr<CollectiveVariable> cv, Scalar sigma, Scalar cv_min=Scalar(0.0), Scalar cv_max=Scalar(0.0), int num_points=0)
            {
            assert(cv);
            assert(sigma > 0);

            CollectiveVariableItem cv_item;

            cv_item.m_cv = cv;
            cv_item.m_sigma = sigma;
           
            cv_item.m_cv_min = cv_min;
            cv_item.m_cv_max = cv_max;
            cv_item.m_num_points = (unsigned int) num_points;

            m_variables.push_back(cv_item);
            }

        void removeAllVariables()
            {
            // Issue a warning if we are already initialized
            if (m_is_initialized)
                {
                m_exec_conf->msg->warning() << "integrate.mode_metadynamics: Removing collective after initialization. Results may be inconsistent." << endl;
                }
            m_variables.clear();
            }

        std::vector< std::string > getProvidedLogQuantities()
            {
            return m_log_names;
            }

        Scalar getLogValue(const std::string& quantity, unsigned int timestep)
            {
            if (quantity == m_log_names[0])
                {
                return m_curr_bias_potential;
                }
            else
                { 
                // default: throw exception
                std::cerr << std::endl << "***Error! " << quantity << " is not a valid log quantity for IntegratorMetaDynamics"
                          << std::endl << std::endl;
                throw std::runtime_error("Error getting log value");
                }
            }

        void setGrid(bool use_grid);

        bool isInitialized()
            {
            return m_is_initialized;
            }

        void dumpGrid(const std::string& filename);

        void restartFromGridFile(const std::string &filename)
            {
            m_restart_filename = filename;
            }

        void setAddHills(bool add_hills)
            {
            m_add_hills = add_hills;
            }

    private:
        Scalar m_W;                                       //!< Height of Gaussians
        Scalar m_T_shift;                                 //!< Temperature shift
        unsigned int m_stride;                            //!< Number of timesteps between Gaussian depositions
        std::vector<CollectiveVariableItem> m_variables;  //!< The list of collective variables

        std::vector<std::vector<Scalar> > m_cv_values;    //!< History of CV values

        unsigned int m_num_update_steps;                  //!< Number of update steps performed in this run thus far

        std::vector<std::string> m_log_names;             //!< Names of logging quantities
        std::string m_suffix;                             //!< Suffix for unbiased variables
        Scalar m_curr_bias_potential;                     //!< The sum of Gaussians
        std::vector<Scalar> m_bias_potential;             //!< List of values of the bias potential

        bool m_is_initialized;                            //!< True if history-dependent potential has been initialized
        const std::string m_filename;                     //!< Name of output file
        bool m_overwrite;                                 //!< True if the file should be overwritten
        bool m_is_appending;                              //!< True if we are appending to the file
        std::ofstream m_file;                             //!< Output log file
        std::string m_delimiter;                          //!< Delimiting string

        bool m_use_grid;                                  //!< True if we are using a grid
        std::vector<Scalar> m_grid;                       //!< d-dimensional grid to store values of bias potential
        IndexGrid m_grid_index;                           //!< Indexer for the d-dimensional grid

        bool m_add_hills;                                 //!< True if hills should be added during the simulation
        std::string m_restart_filename;                   //!< Name of file to read restart information from

        void updateBiasPotential(unsigned int timestep);

        void addCVForces();

        //! Helper function to open output file for logging
        void openOutputFile();

        //! Helper function to write file header
        void writeFileHeader();

        //! Helper function to initialize the grid
        void setupGrid();

        //! Helper function to get value of bias potential by multilinear interpolation
        Scalar interpolateBiasPotential(const std::vector<Scalar>& val);

        //! Helper function to read in data from grid file
        void readGrid(const std::string& filename);

    };

//! Export to python
void export_IntegratorMetaDynamics();

#endif // __INTEGRATOR_METADYNAMICS_H__
