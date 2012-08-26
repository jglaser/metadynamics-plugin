#ifndef __INTEGRATOR_METADYNAMICS_H__
#define __INTEGRATOR_METADYNAMICS_H__

#include <hoomd/hoomd.h>

#include <boost/shared_ptr.hpp>

#include "CollectiveVariable.h"
#include "IndexGrid.h"

/*! \file IntegratorMetaDynamics.h
    \brief Declares the IntegratorMetaDynamics class
*/

//! Structure to hold information about a collective variable
struct CollectiveVariableItem
    {
    boost::shared_ptr<CollectiveVariable> m_cv; //!< The collective variable
    Scalar m_sigma;                             //!< Width of compensating gaussians for this variable
    Scalar m_cv_min;                            //!< Minium value of collective variable (if using grid)
    Scalar m_cv_max;                            //!< Maximum value of collective variable (if using grid)
    Scalar m_num_points;                        //!< Number of grid points for this collective variable
    };

//! Implements a metadynamics update scheme
/*! This class implements an integration scheme for metadynamics.
 
    For a review of metadynamics, a free-energy measurement technique,
    see Barducci et al., Metadynamics,
    Wiley Interdiscipl. Rev.: Comput. Mol. Sci. 5, pp. 826-843 (2011).

    Some of the features of this class are loosely inspired by the
    PLUMED plugin for Metadynamics, http://www.plumed-code.org/.

    The IntegratorMetaDynamics class takes a number of CollectiveVariables,
    and uses their values during the course of the simulation to update
    the bias potential, a sum of Gaussians, which disfavors revisiting
    visited states. The derivative of the bias potential (with respect
    to the collective coordinate) in turn multiplies the force on the
    particles as defined by the collective variable.

    The bias potential is updated every stride steps.

    Well-tempered metadynamics is supported by default, a value of T_shift
    defines the temperature shift entering the scaling factor for the
    bias potential (see above ref. for details). Very large values of T_shift
    correspond to standard metadynamics, T_shift->0 corresponds to standard
    molecular dynamics.

    Two modes of operation are supported: in the first mode, the Gaussians
    are resummed every time step. With increasing simulation time,
    this process will take longer, and slow down the simulation.
    In the second mode, a grid mode, a current grid of values of the
    collective variables is maintained and updated whenever a new 
    Gaussian is deposited. The value of the bias potential and its
    derivatives of the potential are calculated by using numerical
    interpolation. In this mode, the time per step is O(1), so this mode
    is preferrable for long simulations.

    It is possible to output the grid after the simulation, and to restart
    from the grid file. It is also possible to restart from the grid file
    and turn off the deposition of new Gaussians, e.g. to equilibrate
    the system in the bias potential landscape and measure the histogram of
    the collective variable, to correct for errors.
*/ 
class IntegratorMetaDynamics : public IntegratorTwoStep
    {
    public:

        /*! Constructor
           \param sysdef System definition
           \param deltaT Time step
           \param W Weight of Gaussians (multiplicative factor)
           \param T_shift Temperature shift for well-tempered metadynamics
           \param stride Number of time steps between deposition of Gaussians
           \param add_hills True if new Gaussians should be added during the simulation
           \param filename Name of file to output hill information to
           \param overwrite True f the file should be overwritten when it exists already
           \param use_grid True if grid should be used
        */
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

        /*! Sample collective variables, update bias potential and derivatives
           \param timestep The current value of the timestep
         */
        virtual void update(unsigned int timestep);

        /*! Prepare for the run
           \param timestep The current value of the timestep
         */
        virtual void prepRun(unsigned int timestep);

        /*! Output statistics at end of run (unimplemented as of now)
         */
        virtual void printStats() {};

        /*! Register a new collective variable
            \param cv The collective variable
            \param sigma The standard deviation of Gaussians for this collective variable
            \param cv_min Minimum value, if using grid
            \param cv_max Maximum value, if using grid
            \param num_points Number of grid points to use for interpolation
         */
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

        /*! Remove all collective variables
         */
        void removeAllVariables()
            {
            // Issue a warning if we are already initialized
            if (m_is_initialized)
                {
                m_exec_conf->msg->warning() << "integrate.mode_metadynamics: Removing collective after initialization. Results may be inconsistent." << endl;
                }
            m_variables.clear();
            }

        /*! Returns the names of log quantitites provided by this integrator
         */
        std::vector< std::string > getProvidedLogQuantities()
            {
            std::vector< std::string> ret = m_log_names;
            std::vector< std::string> q = Integrator::getProvidedLogQuantities();

            ret.insert(ret.end(), q.begin(), q.end());
            return ret;
            }

        /*! Obtain the value of a specific log quantity
            \param quantity The quantity to obtain the value of
            \param timestep The current value of the time step
         */
        Scalar getLogValue(const std::string& quantity, unsigned int timestep)
            {
            // check if log quantity exists in base class
            std::vector< std::string> q = Integrator::getProvidedLogQuantities();

            std::vector< std::string >::iterator it;
            for (it = q.begin(); it != q.end(); it++)
                if (quantity == *it) return Integrator::getLogValue(quantity, timestep);
                
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

        /*! Set up grid interpolation
            \param use_grid True if the grid is to be used
         */
        void setGrid(bool use_grid);

        /*! Returns true if the integration has been initialized (i.e.
            the simulation has been run at least once)
         */
        bool isInitialized()
            {
            return m_is_initialized;
            }

        /*! Output the grid to a file
            \param filename Name of file to dump grid to
         */
        void dumpGrid(const std::string& filename);

        /*! Restart from a grid file. Upon running the simulation,
            the information will be read in.

            \param filename The name of the file that contains the grid information
         */
        void restartFromGridFile(const std::string &filename)
            {
            m_restart_filename = filename;
            }

        /*! Set a flag that controls deposition of new Gaussian hills
            \param add_hills True if hills should be generated during the simulation
         */
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

        //! Internal helper function to update the bias potential
        void updateBiasPotential(unsigned int timestep);

        //! Helper function to open output file for logging
        void openOutputFile();

        //! Helper function to write file header
        void writeFileHeader();

        //! Helper function to initialize the grid
        void setupGrid();

        //! Helper function to get value of bias potential by multilinear interpolation
        Scalar interpolateBiasPotential(const std::vector<Scalar>& val);

        //! Helper function to calculate the partial derivative of the bias potential in direction cv
        Scalar biasPotentialDerivative(unsigned int cv, const std::vector<Scalar>& val);

        //! Helper function to read in data from grid file
        void readGrid(const std::string& filename);
    };

//! Export to python
void export_IntegratorMetaDynamics();

#endif // __INTEGRATOR_METADYNAMICS_H__
