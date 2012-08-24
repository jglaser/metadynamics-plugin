/*! \file IntegratorMetaDynamics.cc
    \brief Implements the IntegratorMetaDynamics class
 */

#include "IntegratorMetaDynamics.h"

#include <stdio.h>
#include <iomanip>
#include <sstream>

using namespace std;

#include <boost/python.hpp>
#include <boost/filesystem.hpp>

#ifdef ENABLE_MPI
#include <boost/mpi.hpp>
#endif

using namespace boost::python;
using namespace boost::filesystem;


IntegratorMetaDynamics::IntegratorMetaDynamics(boost::shared_ptr<SystemDefinition> sysdef,
            Scalar deltaT,
            Scalar W,
            Scalar T_shift,
            unsigned int stride,
            bool add_hills,
            const std::string& filename,
            bool overwrite,
            bool use_grid)
    : IntegratorTwoStep(sysdef, deltaT),
      m_W(W),
      m_T_shift(T_shift),
      m_stride(stride),
      m_num_update_steps(0),
      m_curr_bias_potential(0.0),
      m_is_initialized(false),
      m_filename(filename),
      m_overwrite(overwrite),
      m_is_appending(false),
      m_delimiter("\t"),
      m_use_grid(use_grid),
      m_add_hills(add_hills),
      m_restart_filename("")
    {
    assert(m_T_shift>0);
    assert(m_W > 0);

    m_log_names.push_back("bias_potential");
    }

void IntegratorMetaDynamics::openOutputFile()
    {
    if (exists(m_filename) && !m_overwrite)
        {
        m_exec_conf->msg->notice(3) << "integrate.mode_metadynamics: Appending log to existing file \"" << m_filename << "\"" << endl;
        m_file.open(m_filename.c_str(), ios_base::in | ios_base::out | ios_base::ate);
        m_is_appending = true;
        }
    else
        {
        m_exec_conf->msg->notice(3) << "integrate.mode_metadynamics: Creating new log in file \"" << m_filename << "\"" << endl;
        m_file.open(m_filename.c_str(), ios_base::out);
        m_is_appending = false;
        }
    if (!m_file.good())
        {
        m_exec_conf->msg->error() << "integrate.mode_metadynamics: Error opening log file " << m_filename << endl;
        throw runtime_error("Error initializing IntegratorMetadynamics");
        }
    }

void IntegratorMetaDynamics::writeFileHeader()
    {
    assert(m_variables.size());
    assert(m_file);

    m_file << "timestep" << m_delimiter << "W" << m_delimiter;

    std::vector<CollectiveVariableItem>::iterator it;
    for (it = m_variables.begin(); it != m_variables.end(); ++it)
        {
        m_file << it->m_cv->getName();
        m_file << m_delimiter << "sigma_" << it->m_cv->getName();

        if (it != m_variables.end())
            m_file << m_delimiter;
        }

    m_file << endl;
    }

void IntegratorMetaDynamics::prepRun(unsigned int timestep)
    {
#ifdef ENABLE_MPI
    bool is_root = true;

    if (m_pdata->getDomainDecomposition())
        is_root = m_exec_conf->isMPIRoot();

    if (is_root)
#endif
        {
        // Set up file output
        if (! m_is_initialized && m_filename != "")
            {
            openOutputFile();
            if (! m_is_appending)
                writeFileHeader();
            }
        
     
        // Set up colllective variables
        if (! m_is_initialized)
            {
            m_cv_values.resize(m_variables.size());
            std::vector< std::vector<Scalar> >::iterator it;

            for (it = m_cv_values.begin(); it != m_cv_values.end(); ++it)
                it->clear();

            m_num_update_steps = 0;
            m_bias_potential.clear();
            }

        // Set up grid if necessary
        if (! m_is_initialized && m_use_grid)
            {
            setupGrid();

            if (m_restart_filename != "")
                {
                // restart from file
                m_exec_conf->msg->notice(2) << "integrate.mode_metadynamics: Restarting from grid file \"" << m_restart_filename << "\"" << endl;

                readGrid(m_restart_filename);

                m_restart_filename = "";
                }
            }

        } // endif isMPIRoot()

    m_is_initialized = true;
#ifdef ENABLE_MPI
    if (m_comm)
        {
        // perform all necessary communication steps. This ensures
        // a) that particles have migrated to the correct domains
        // b) that forces are calculated correctly
        m_comm->communicate(timestep);
        }
#endif

    // initial update of the potential
    updateBiasPotential(timestep);

    IntegratorTwoStep::prepRun(timestep);
    }

void IntegratorMetaDynamics::update(unsigned int timestep)
    {
    // issue a warning if no integration methods are set
    if (!m_gave_warning && m_methods.size() == 0)
        {
        m_exec_conf->msg->warning() << "No integration methods are set, continuing anyways." << endl;
        m_gave_warning = true;
        }
    
    // ensure that prepRun() has been called
    assert(this->m_prepared);
    
    if (m_prof)
        m_prof->push("Integrate");
    
    // perform the first step of the integration on all groups
    std::vector< boost::shared_ptr<IntegrationMethodTwoStep> >::iterator method;
    for (method = m_methods.begin(); method != m_methods.end(); ++method)
        (*method)->integrateStepOne(timestep);

    // Update the rigid body particle positions and velocities if they are present
    if (m_sysdef->getRigidData()->getNumBodies() > 0)
        m_sysdef->getRigidData()->setRV(true);

    if (m_prof)
        m_prof->pop();

#ifdef ENABLE_MPI
    if (m_comm)
        {
        // perform all necessary communication steps. This ensures
        // a) that particles have migrated to the correct domains
        // b) that forces are calculated correctly
        m_comm->communicate(timestep+1);
        }
#endif

    // update bias potential
    updateBiasPotential(timestep+1);

    // compute the net force on all particles
#ifdef ENABLE_CUDA
    if (exec_conf->exec_mode == ExecutionConfiguration::GPU)
        computeNetForceGPU(timestep+1);
    else
#endif
        computeNetForce(timestep+1);

    if (m_prof)
        m_prof->push("Integrate");

    // if the virial needs to be computed and there are rigid bodies, perform the virial correction
    PDataFlags flags = m_pdata->getFlags();
    if (flags[pdata_flag::isotropic_virial] && m_sysdef->getRigidData()->getNumBodies() > 0)
        m_sysdef->getRigidData()->computeVirialCorrectionStart();

    // perform the second step of the integration on all groups
    for (method = m_methods.begin(); method != m_methods.end(); ++method)
        (*method)->integrateStepTwo(timestep);

    // Update the rigid body particle velocities if they are present
    if (m_sysdef->getRigidData()->getNumBodies() > 0)
       m_sysdef->getRigidData()->setRV(false);

    // if the virial needs to be computed and there are rigid bodies, perform the virial correction
    if (flags[pdata_flag::isotropic_virial] && m_sysdef->getRigidData()->getNumBodies() > 0)
        m_sysdef->getRigidData()->computeVirialCorrectionEnd(m_deltaT/2.0);

    if (m_prof)
        m_prof->pop();
    } 

void IntegratorMetaDynamics::updateBiasPotential(unsigned int timestep)
    {
    // exit early if there are no collective variables
    if (m_variables.size() == 0)
        return;

    // collect values of collective variables
    std::vector< Scalar> current_val;
    std::vector<CollectiveVariableItem>::iterator it;
    for (it = m_variables.begin(); it != m_variables.end(); ++it)
        {
        Scalar val = it->m_cv->getCurrentValue(timestep);
        current_val.push_back(val);
        }

        std::vector<double> bias(m_variables.size(), 0.0); 

    bool is_root = true;

    if (m_prof)
        m_prof->push("Metadynamics");

#ifdef ENABLE_MPI
    if (m_pdata->getDomainDecomposition())
        is_root = m_exec_conf->isMPIRoot();

    if (is_root)
#endif
        {
        if (! m_use_grid && (m_num_update_steps % m_stride == 0))
            {
            // record history of CV values every m_stride steps
            std::vector<CollectiveVariableItem>::iterator it;
            for (it = m_variables.begin(); it != m_variables.end(); ++it)
                {
                unsigned int cv_index = it - m_variables.begin();
                m_cv_values[cv_index].push_back(current_val[cv_index]);
                }
            }

        // update biasing weights by summing up partial derivivatives of Gaussians deposited every m_stride steps
        m_curr_bias_potential = 0.0;

        if (m_use_grid)
            {
            // interpolate current value of bias potential
            Scalar V = interpolateBiasPotential(current_val);
            m_curr_bias_potential = V;

            if (m_add_hills && (m_num_update_steps % m_stride == 0))
                {
                // add Gaussian to grid
                
                // scaling factor for well-tempered MetaD
                Scalar scal = exp(-V/m_T_shift);

                // loop over grid
                unsigned int len = m_grid_index.getNumElements();
                std::vector<unsigned int> coords(m_grid_index.getDimension()); 
                for (unsigned int grid_idx = 0; grid_idx < len; grid_idx++)
                    {
                    // obtain d-dimensional coordinates
                    m_grid_index.getCoordinates(grid_idx, coords);

                    Scalar gauss_exp(0.0);
                    // evaluate Gaussian on grid point
                    for (unsigned int cv_idx = 0; cv_idx < m_variables.size(); ++cv_idx)
                        {
                        Scalar delta = (m_variables[cv_idx].m_cv_max - m_variables[cv_idx].m_cv_min)/
                                       (m_variables[cv_idx].m_num_points - 1);
                        Scalar val = m_variables[cv_idx].m_cv_min + coords[cv_idx]*delta;
                        Scalar sigma = m_variables[cv_idx].m_sigma;
                        double d = val - current_val[cv_idx];
                        gauss_exp += d*d/2.0/sigma/sigma;
                        }
                    double gauss = exp(-gauss_exp);

                    // add Gaussian to grid
                    m_grid[grid_idx] += m_W*scal*gauss;
                    }
                }

            // calculate partial derivatives numerically
            for (unsigned int cv_idx = 0; cv_idx < m_variables.size(); ++cv_idx)
                bias[cv_idx] = biasPotentialDerivative(cv_idx, current_val);

            } 
        else
            {
            // sum up all Gaussians accumulated until now
            for (unsigned int gauss_idx = 0; gauss_idx < m_bias_potential.size(); ++gauss_idx)
                {
                double gauss_exp = 0.0;
                // calculate Gaussian contribution from t'=gauss_idx*m_stride
                std::vector<Scalar>::iterator val_it;
                for (val_it = current_val.begin(); val_it != current_val.end(); ++val_it)
                    {
                    Scalar val = *val_it;
                    unsigned int cv_index = val_it - current_val.begin();
                    Scalar sigma = m_variables[cv_index].m_sigma;
                    double delta = val - m_cv_values[cv_index][gauss_idx];
                    gauss_exp += delta*delta/2.0/sigma/sigma;
                    }
                double gauss = exp(-gauss_exp);

                // calculate partial derivatives
                std::vector<CollectiveVariableItem>::iterator cv_item;
                Scalar scal = exp(-m_bias_potential[gauss_idx]/m_T_shift);
                for (cv_item = m_variables.begin(); cv_item != m_variables.end(); ++cv_item)
                    {
                    unsigned int cv_index = cv_item - m_variables.begin();
                    Scalar val = current_val[cv_index];
                    Scalar sigma = m_variables[cv_index].m_sigma;
                    bias[cv_index] -= m_W*scal/sigma/sigma*(val - m_cv_values[cv_index][gauss_idx])*gauss;
                    }

                m_curr_bias_potential += m_W*scal*gauss;
                }
            }

        // write hills information
        if (m_is_initialized && (m_num_update_steps % m_stride == 0) && m_add_hills)
            {
            Scalar W = m_W*exp(-m_curr_bias_potential/m_T_shift);
            m_file << setprecision(10) << timestep << m_delimiter;
            m_file << setprecision(10) << W << m_delimiter;

            std::vector<Scalar>::iterator cv;
            for (cv = current_val.begin(); cv != current_val.end(); ++cv)
                {
                unsigned int cv_index = cv - current_val.begin();
                m_file << setprecision(10) << *cv << m_delimiter;
                m_file << setprecision(10) << m_variables[cv_index].m_sigma;
                if (cv != current_val.end() -1) m_file << m_delimiter;
                }

            m_file << endl;
            }
       
        if (m_add_hills && (! m_use_grid) && (m_num_update_steps % m_stride == 0))
            m_bias_potential.push_back(m_curr_bias_potential);

        // increment number of updated steps
        m_num_update_steps++;
     
        } // endif root processor

#ifdef ENABLE_MPI
    // broadcast bias factors
    if (m_pdata->getDomainDecomposition())
        boost::mpi::broadcast(*m_exec_conf->getMPICommunicator(), bias, 0);
#endif

    // update current bias potential derivative for every collective variable
    std::vector<CollectiveVariableItem>::iterator cv_item;
    for (cv_item = m_variables.begin(); cv_item != m_variables.end(); ++cv_item)
        {
        unsigned int cv_index = cv_item - m_variables.begin();
        cv_item->m_cv->setBiasFactor(bias[cv_index]);
        }

   if (m_prof)
        m_prof->pop();
    }

void IntegratorMetaDynamics::setupGrid()
    {
    assert(! m_is_initialized);
    assert(m_variables.size());

    std::vector< CollectiveVariableItem >::iterator it;

    std::vector< unsigned int > lengths(m_variables.size());

    for (it = m_variables.begin(); it != m_variables.end(); ++it)
        {
        lengths[it - m_variables.begin()] = it->m_num_points;
        }

    m_grid_index.setLengths(lengths);
    m_grid.resize(m_grid_index.getNumElements(),0.0);
    } 

Scalar IntegratorMetaDynamics::interpolateBiasPotential(const std::vector<Scalar>& val)
    {
    assert(val.size() == m_grid_index.getDimension());

    // find closest d-dimensional sub-block
    std::vector<unsigned int> lower_idx(m_grid_index.getDimension());
    std::vector<unsigned int> upper_idx(m_grid_index.getDimension());
    std::vector<Scalar> rel_delta(m_grid_index.getDimension());

    for (unsigned int cv_idx = 0; cv_idx < m_variables.size(); cv_idx++)
        {
        Scalar delta = (m_variables[cv_idx].m_cv_max - m_variables[cv_idx].m_cv_min)/(m_variables[cv_idx].m_num_points - 1);
        int lower = (int) ((val[cv_idx] - m_variables[cv_idx].m_cv_min)/delta);
        int upper = lower+1;

        if (lower < 0 || upper >= m_variables[cv_idx].m_num_points)
            {
            m_exec_conf->msg->warning() << "integrate.mode_metadynamics: Value " << val[cv_idx]
                                        << " of collective variable " << cv_idx << " out of bounds." << endl
                                        << "Assuming bias potential of zero." << endl;
            return Scalar(0.0);
            }

        Scalar lower_bound = m_variables[cv_idx].m_cv_min + delta * lower;
        Scalar upper_bound = m_variables[cv_idx].m_cv_min + delta * upper;
        lower_idx[cv_idx] = lower;
        upper_idx[cv_idx] = upper;
        rel_delta[cv_idx] = (val[cv_idx]-lower_bound)/(upper_bound-lower_bound);
        }

    // construct multilinear interpolation
    unsigned int n_term = 1 << m_grid_index.getDimension();
    Scalar res(0.0);
    for (unsigned int bits = 0; bits < n_term; ++bits)
        {
        std::vector<unsigned int> coords(m_grid_index.getDimension());
        Scalar term(1.0);
        for (unsigned int i = 0; i < m_grid_index.getDimension(); i++)
            {
            if (bits & (1 << i))
                {
                coords[i] = lower_idx[i];
                term *= (Scalar(1.0) - rel_delta[i]);
                }
            else
                {
                coords[i] = upper_idx[i];
                term *= rel_delta[i];
                }
            }
        
        term *= m_grid[m_grid_index.getIndex(coords)];
        res += term;
        }

    return res;
    }

Scalar IntegratorMetaDynamics::biasPotentialDerivative(unsigned int cv, const std::vector<Scalar>& val)
    {
    Scalar delta = (m_variables[cv].m_cv_max - m_variables[cv].m_cv_min)/
                   (m_variables[cv].m_num_points - 1);

    if (val[cv] - delta < m_variables[cv].m_cv_min) 
        {
        // forward difference
        std::vector<Scalar> val2 = val;
        val2[cv] += delta;

        Scalar y2 = interpolateBiasPotential(val2);
        Scalar y1 = interpolateBiasPotential(val);
        return (y2-y1)/delta;
        }
    else if (val[cv] + delta > m_variables[cv].m_cv_max)
        {
        // backward difference
        std::vector<Scalar> val2 = val;
        val2[cv] -= delta;
        Scalar y1 = interpolateBiasPotential(val2);
        Scalar y2 = interpolateBiasPotential(val);
        return (y2-y1)/delta;
        }
    else
        {
        // central difference
        std::vector<Scalar> val2 = val;
        std::vector<Scalar> val1 = val;
        val1[cv] -= delta;
        val2[cv] += delta;
        Scalar y1 = interpolateBiasPotential(val1);
        Scalar y2 = interpolateBiasPotential(val2);
        return (y2 - y1)/(Scalar(2.0)*delta);
        }
    }

void IntegratorMetaDynamics::setGrid(bool use_grid)
    {
#ifdef ENABLE_MPI
    // Only on root processor
    if (m_pdata->getDomainDecomposition())
        if (! m_exec_conf->isMPIRoot()) return;
#endif
    if (m_is_initialized)
        {
        m_exec_conf->msg->error() << "integrate.mode_metadynamics: Cannot change grid mode after initialization." << endl;
        throw std::runtime_error("Error setting up metadynamics parameters.");
        }

    m_use_grid = use_grid;

    if (use_grid)
        {
        // Check for some input errors
        std::vector<CollectiveVariableItem>::iterator it;

        for (it = m_variables.begin(); it != m_variables.end(); ++it)
            {
            if (it->m_cv_min >= it->m_cv_max)
                {
                m_exec_conf->msg->error() << "integrate.mode_metadyanmics: Maximum grid value of collective variable has to be greater than minimum value.";
                throw std::runtime_error("Error creating collective variable.");
                
                }

            if (it->m_num_points < 2)
                {
                m_exec_conf->msg->error() << "integrate.mode_metadynamics: Number of grid points for collective variable has to be at least two.";
                throw std::runtime_error("Error creating collective variable.");
                }
            }
        }
    }

void IntegratorMetaDynamics::dumpGrid(const std::string& filename)
    {
#ifdef ENABLE_MPI
    // Only on root processor
    if (m_pdata->getDomainDecomposition())
        if (! m_exec_conf->isMPIRoot()) return;
#endif

    if (! m_use_grid)
        {
        m_exec_conf->msg->error() << "integrate.mode_metadynamics: Grid information can only be dumped if grid is enabled.";
        throw std::runtime_error("Error dumping grid.");
        }
 
    std::ofstream file;

    // open output file
    file.open(filename.c_str(), ios_base::out);

    // write file header
    file << "#n_cv: " << m_grid_index.getDimension() << std::endl;
    file << "#dim:";
    
    for (unsigned int i= 0; i < m_grid_index.getDimension(); i++)
        file << " " << m_grid_index.getLength(i);

    file << std::endl;

    file << "grid_value";

    for (unsigned int i = 0; i < m_grid_index.getDimension(); i++)
        file << m_delimiter << "cv" << i;

    file << std::endl;

    // loop over grid
    unsigned int len = m_grid_index.getNumElements();
    std::vector<unsigned int> coords(m_grid_index.getDimension()); 
    for (unsigned int grid_idx = 0; grid_idx < len; grid_idx++)
        {
        // obtain d-dimensional coordinates
        m_grid_index.getCoordinates(grid_idx, coords);

        file << setprecision(10) << m_grid[grid_idx];

        for (unsigned int cv_idx = 0; cv_idx < m_variables.size(); ++cv_idx)
            {
            Scalar delta = (m_variables[cv_idx].m_cv_max - m_variables[cv_idx].m_cv_min)/
                           (m_variables[cv_idx].m_num_points - 1);
            Scalar val = m_variables[cv_idx].m_cv_min + coords[cv_idx]*delta;

            file << m_delimiter << setprecision(10) << val;
            }

        file << std::endl;
        }

    file.close();
    }

void IntegratorMetaDynamics::readGrid(const std::string& filename)
    {
#ifdef ENABLE_MPI
    // Only on root processor
    if (m_pdata->getDomainDecomposition())
        if (! m_exec_conf->isMPIRoot()) return;
#endif

    if (! m_use_grid)
        {
        m_exec_conf->msg->error() << "integrate.mode_metadynamics: Grid information can only be read if grid is enabled.";
        throw std::runtime_error("Error reading grid.");
        }
    std::ifstream file;

    // open grid file
    file.open(filename.c_str());

    std::string line; 

    // Skip file header
    getline(file, line);
    getline(file, line);
    getline(file, line);

    unsigned int len = m_grid_index.getNumElements();
    std::vector<unsigned int> coords(m_grid_index.getDimension()); 
    for (unsigned int grid_idx = 0; grid_idx < len; grid_idx++)
        {
        if (! file.good())
            {
            m_exec_conf->msg->error() << "integrate.mode_metadynamics: Premature end of grid file.";
            throw std::runtime_error("Error reading grid.");
            }
     
        getline(file, line);
        istringstream iss(line);
        iss >> m_grid[grid_idx];
        }
    
    file.close();
    } 

void IntegratorMetaDynamics::testInterpolation(const std::string& filename, const std::vector<unsigned int>& dim)
    {
    std::vector<unsigned int> coords(m_grid_index.getDimension()); 

    std::vector<Scalar> val(m_variables.size());

    IndexGrid test_grid(dim);
    std::ofstream f;
    f.open(filename.c_str(), std::ios::out);
    for (unsigned int i = 0; i < test_grid.getNumElements(); i++)
        {
        test_grid.getCoordinates(i, coords);

        // evaluate Gaussian on grid point
        for (unsigned int cv_idx = 0; cv_idx < m_variables.size(); ++cv_idx)
            {
            Scalar delta = (m_variables[cv_idx].m_cv_max - m_variables[cv_idx].m_cv_min)/ dim[cv_idx];
            val[cv_idx] = m_variables[cv_idx].m_cv_min + coords[cv_idx]*delta;

            f << val[cv_idx];
            }

        f << " ";

        f << interpolateBiasPotential(val);

        for (unsigned int cv_idx = 0; cv_idx < m_variables.size(); ++cv_idx)
            f << " " << biasPotentialDerivative(cv_idx, val);

        f << std::endl;
        }
    }


void export_IntegratorMetaDynamics()
    {
    class_<IntegratorMetaDynamics, boost::shared_ptr<IntegratorMetaDynamics>, bases<IntegratorTwoStep>, boost::noncopyable>
    ("IntegratorMetaDynamics", init< boost::shared_ptr<SystemDefinition>,
                          Scalar,
                          Scalar,
                          Scalar,
                          unsigned int,
                          bool,
                          const std::string&,
                          bool>())
    .def("registerCollectiveVariable", &IntegratorMetaDynamics::registerCollectiveVariable)
    .def("removeAllVariables", &IntegratorMetaDynamics::removeAllVariables)
    .def("isInitialized", &IntegratorMetaDynamics::isInitialized)
    .def("setGrid", &IntegratorMetaDynamics::setGrid)
    .def("dumpGrid", &IntegratorMetaDynamics::dumpGrid)
    .def("restartFromGridFile", &IntegratorMetaDynamics::restartFromGridFile)
    .def("setAddHills", &IntegratorMetaDynamics::setAddHills)
    .def("testInterpolation", &IntegratorMetaDynamics::testInterpolation)
    ;
    }
