#include <hoomd/hoomd.h>

#include <boost/shared_ptr.hpp>

struct CollectiveVariableItem
    {
    boost::shared_ptr<ForceCompute> m_cv;       // The collective variable
    Scalar m_sigma;                             // Width of compensating gaussians for this variable
    Scalar m_curr_bias;                         // current partial deriviative of the biasing potential
    };

//! Implements a metadynamics update scheme
class IntegratorMetaDynamics : public IntegratorTwoStep
    {
    public:

        IntegratorMetaDynamics(boost::shared_ptr<SystemDefinition> sysdef,
                   Scalar deltaT,
                   Scalar W,
                   Scalar T_shift,
                   unsigned int stride);

        virtual ~IntegratorMetaDynamics() {}

        //! Sample the collective variables and update their biasing coefficients
        virtual void update(unsigned int timestep);

        //! Reset metadynamics potential
        virtual void resetStats()
            {
            m_cv_values.resize(m_variables.size());
            std::vector< std::vector<Scalar> >::iterator it;

            for (it = m_cv_values.begin(); it != m_cv_values.end(); ++it)
                it->clear();

            m_num_update_steps = 0;
            m_log_weight.clear();
            m_reweighting_factor = 1.0;
            m_Vdot_avg_sum = 0.0;
            m_bias_potential.clear();
            }

        virtual void printStats() {};

        void registerCollectiveVariable(boost::shared_ptr<ForceCompute> cv, Scalar sigma)
            {
            assert(cv);
            assert(gaussian_width > 0);

            CollectiveVariableItem cv_item;

            cv_item.m_cv = cv;
            cv_item.m_sigma = sigma;

            m_variables.push_back(cv_item);
            }

        void removeAllVariables()
            {
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
            else if (quantity == m_log_names[1])
                {
                return m_reweighting_factor;
                }
            else
                { 
                // default: throw exception
                std::cerr << std::endl << "***Error! " << quantity << " is not a valid log quantity for IntegratorMetaDynamics"
                          << std::endl << std::endl;
                throw std::runtime_error("Error getting log value");
                }
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
        std::vector<double> m_log_weight;                 //!< Logarithm of reweighting factors
        double m_reweighting_factor;                      //!< Ratio between unbiased and biased ensemble
        double m_Vdot_avg_sum;                            //!< Logarithm of reweighting factor
        std::vector<Scalar> m_bias_potential;             //!< List of values of the bias potential

        void updateBiasPotential(unsigned int timestep);

        void addCVForces();
    };

//! Export to python
void export_IntegratorMetaDynamics();

