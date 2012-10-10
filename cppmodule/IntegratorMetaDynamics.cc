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

#ifdef ENABLE_CUDA
#include "IntegratorMetaDynamics.cuh"
#endif 
using namespace boost::python;
using namespace boost::filesystem;


IntegratorMetaDynamics::IntegratorMetaDynamics(boost::shared_ptr<SystemDefinition> sysdef,
            Scalar deltaT,
            Scalar W,
            Scalar T_shift,
            Scalar T,
            unsigned int stride,
            bool add_bias,
            const std::string& filename,
            bool overwrite,
            const Enum mode)
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
      m_use_grid(false),
      m_add_bias(add_bias),
      m_restart_filename(""),
      m_grid_fname1(""),
      m_grid_fname2(""),
      m_grid_period(0),
      m_cur_file(0),
      m_temp(T),
      m_mode(mode),
      m_stride_multiply(1),
      m_num_label_change(0),
      m_min_label_change(0)
    {
    assert(m_T_shift>0);
    assert(m_W > 0);

    m_log_names.push_back("bias_potential");

    // Initial state for flux-tempered MetaD
    m_compute_histograms = false;
    m_walker_state = true;
    m_num_histogram_entries = 0;
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
        is_root = m_exec_conf->isRoot();

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

#ifdef ENABLE_CUDA
            if (m_exec_conf->isCUDAEnabled())
                {
                // initialize GPU mirror values for collective variable data
                GPUArray<Scalar> cv_min(m_variables.size(), m_exec_conf);
                m_cv_min.swap(cv_min);

                GPUArray<Scalar> cv_max(m_variables.size(), m_exec_conf);
                m_cv_max.swap(cv_max);

                GPUArray<Scalar> current_val(m_variables.size(), m_exec_conf);
                m_current_val.swap(current_val);

                GPUArray<unsigned int> lengths(m_variables.size(), m_exec_conf);
                m_lengths.swap(lengths);

                GPUArray<Scalar> sigma(m_variables.size(), m_exec_conf);
                m_sigma.swap(sigma);
    
                ArrayHandle<Scalar> h_cv_min(m_cv_min, access_location::host, access_mode::overwrite);
                ArrayHandle<Scalar> h_cv_max(m_cv_max, access_location::host, access_mode::overwrite);
                ArrayHandle<unsigned int> h_lengths(m_lengths, access_location::host, access_mode::overwrite);
                ArrayHandle<Scalar> h_sigma(m_sigma, access_location::host, access_mode::overwrite);
                
                for (unsigned int cv_idx = 0; cv_idx < m_variables.size(); cv_idx++)
                    {
                    h_cv_min.data[cv_idx] = m_variables[cv_idx].m_cv_min;
                    h_cv_max.data[cv_idx] = m_variables[cv_idx].m_cv_max;
                    h_sigma.data[cv_idx] = m_variables[cv_idx].m_sigma;
                    h_lengths.data[cv_idx] = m_variables[cv_idx].m_num_points;
                    }
                }
#endif
            m_num_update_steps = 0;
            m_bias_potential.clear();
            }

        // Set up histograms if necessary
        if (! m_is_initialized && m_compute_histograms)
            {
            if (m_variables.size() != 1)
                {
                m_exec_conf->msg->error() << "integrate.mode_metadynamics: With histogram computation, exactly one CV is required." << std::endl;
                throw std::runtime_error("Error initializing metadynamics");
                }

            setupHistograms();
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

        } // endif isRoot()

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

    std::vector<Scalar> bias(m_variables.size(), 0.0); 

    bool is_root = true;

    if (m_prof)
        m_prof->push("Metadynamics");

#ifdef ENABLE_MPI
    if (m_pdata->getDomainDecomposition())
        is_root = m_exec_conf->isRoot();

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

        if (m_compute_histograms)
            {
            assert(m_variables.size() == 1);
            Scalar val = current_val[0];

            // change walker state if necessary
            Scalar min = m_variables[0].m_ftm_min;
            Scalar max = m_variables[0].m_ftm_max;
            if ( val <= min && m_walker_state == true)
                {
                m_walker_state = false;
                m_num_label_change++;
                }

            if ( val >= max && m_walker_state == false)
                {
                m_walker_state = true;
                m_num_label_change++;
                }

            // record histograms
            #ifdef ENABLE_CUDA
            if (m_exec_conf->isCUDAEnabled())
                sampleHistogramsGPU(val, m_walker_state);
            else
                sampleHistograms(val, m_walker_state);
            #else
            sampleHistograms(val, m_walker_state);
            #endif

            m_num_histogram_entries++;
            }

        // update biasing weights by summing up partial derivivatives of Gaussians deposited every m_stride steps
        m_curr_bias_potential = 0.0;

        if (m_use_grid)
            {
            // interpolate current value of bias potential
            Scalar V = interpolateBiasPotential(current_val);
            m_curr_bias_potential = V;

            if (m_add_bias && (m_num_update_steps % m_stride == 0)
                && (! (m_mode == mode_flux_tempered) || m_num_label_change >= m_min_label_change))
                {
                // add Gaussian to grid
               
                // scaling factor for well-tempered MetaD
                Scalar scal = Scalar(1.0);
                if (m_mode == mode_well_tempered)
                    scal = exp(-V/m_T_shift);

#ifdef ENABLE_CUDA
                if (m_exec_conf->isCUDAEnabled())
                    updateGridGPU(current_val, scal);
                else
                    updateGrid(current_val, scal);
#else
                updateGrid(current_val, scal);
#endif

                // reset statistics
                if (m_mode == mode_flux_tempered)
                    {
                    resetHistograms();
                    m_num_label_change = 0;
                    }
                }

            // calculate partial derivatives numerically
            for (unsigned int cv_idx = 0; cv_idx < m_variables.size(); ++cv_idx)
                bias[cv_idx] = biasPotentialDerivative(cv_idx, current_val);

            } 
        else
            {
            if (m_mode == mode_flux_tempered)
                {
                m_exec_conf->msg->error() << "integrate.mode_metadynamics: Flux-tempered MetaD is only supported in grid-mode" << std::endl;
                throw std::runtime_error("Error in metadynamics integration.");
                }

            // sum up all Gaussians accumulated until now
            for (unsigned int gauss_idx = 0; gauss_idx < m_bias_potential.size(); ++gauss_idx)
                {
                Scalar gauss_exp = 0.0;
                // calculate Gaussian contribution from t'=gauss_idx*m_stride
                std::vector<Scalar>::iterator val_it;
                for (val_it = current_val.begin(); val_it != current_val.end(); ++val_it)
                    {
                    Scalar val = *val_it;
                    unsigned int cv_index = val_it - current_val.begin();
                    Scalar sigma = m_variables[cv_index].m_sigma;
                    Scalar delta = val - m_cv_values[cv_index][gauss_idx];
                    gauss_exp += delta*delta/2.0/sigma/sigma;
                    }
                Scalar gauss = exp(-gauss_exp);

                // calculate partial derivatives
                std::vector<CollectiveVariableItem>::iterator cv_item;

                // scaling factor for well-tempered MetaD
                Scalar scal = Scalar(1.0);
                if (m_mode == mode_well_tempered)
                    scal = exp(-m_bias_potential[gauss_idx]/m_T_shift);

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
        if (m_is_initialized && (m_num_update_steps % m_stride == 0) && m_add_bias && m_file.is_open())
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
       
        if (m_add_bias && (! m_use_grid) && (m_num_update_steps % m_stride == 0))
            m_bias_potential.push_back(m_curr_bias_potential);

        // update stride
        if (m_num_update_steps && (m_num_update_steps % m_stride == 0))
            m_stride *= m_stride_multiply;

        // dump grid information if required using alternating scheme
        if (m_grid_period && (timestep % m_grid_period == 0))
            {
            if (m_grid_fname2 != "")
                {
                writeGrid(m_cur_file ? m_grid_fname2 : m_grid_fname1);
                m_cur_file = m_cur_file ? 0 : 1;
                }
            else
                writeGrid(m_grid_fname1);
            }

        // increment number of updated steps
        m_num_update_steps++;
     
        } // endif root processor

#ifdef ENABLE_MPI
    // broadcast bias factors
    if (m_pdata->getDomainDecomposition())
        MPI_Bcast(&bias.front(), bias.size(), MPI_HOOMD_SCALAR, 0, m_exec_conf->getMPICommunicator());

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

    GPUArray<Scalar> grid(m_grid_index.getNumElements(),m_exec_conf);
    m_grid.swap(grid);

    // reset grid
    ArrayHandle<Scalar> h_grid(m_grid, access_location::host, access_mode::overwrite);
    memset(h_grid.data, 0, sizeof(Scalar)*m_grid.getNumElements());
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

    ArrayHandle<Scalar> h_grid(m_grid, access_location::host, access_mode::read);

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
        
        term *= h_grid.data[m_grid_index.getIndex(coords)];
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
        if (! m_exec_conf->isRoot()) return;
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

void IntegratorMetaDynamics::setHistograms(bool compute_histograms)
    {
#ifdef ENABLE_MPI
    // Only on root processor
    if (m_pdata->getDomainDecomposition())
        if (! m_exec_conf->isRoot()) return;
#endif
    if (m_is_initialized)
        {
        m_exec_conf->msg->error() << "integrate.mode_metadynamics: Cannot change histogram mode after initialization." << endl;
        throw std::runtime_error("Error setting up metadynamics parameters.");
        }

    m_compute_histograms = compute_histograms;
    }

void IntegratorMetaDynamics::dumpGrid(const std::string& filename1, const std::string& filename2, unsigned int period)
    {
    if (period == 0)
        {
        // dump grid immediately
        writeGrid(filename1);
        return;
        }

    m_grid_period = period;
    m_grid_fname1 = filename1;
    m_grid_fname2 = filename2;
    }

void IntegratorMetaDynamics::writeGrid(const std::string& filename)
    {
    std::ofstream file;

#ifdef ENABLE_MPI
    // Only on root processor
    if (m_pdata->getDomainDecomposition())
        if (! m_exec_conf->isRoot()) return;
#endif

    if (! m_use_grid)
        {
        m_exec_conf->msg->error() << "integrate.mode_metadynamics: Grid information can only be dumped if grid is enabled.";
        throw std::runtime_error("Error dumping grid.");
        }

 
    // open output file
    file.open(filename.c_str(), ios_base::out);

    // write file header
    file << "#n_cv: " << m_grid_index.getDimension() << std::endl;
    file << "#dim: ";
    
    for (unsigned int i= 0; i < m_grid_index.getDimension(); i++)
        file << " " << m_grid_index.getLength(i);

    file << std::endl;

    file << "#num_histogram_entries: " << m_num_histogram_entries << std::endl;

    for (unsigned int i = 0; i < m_grid_index.getDimension(); i++)
        file << m_variables[i].m_cv->getName() << m_delimiter;

    file << "grid_value";

    file << std::endl;

    // loop over grid
    ArrayHandle<Scalar> h_grid(m_grid, access_location::host, access_mode::read);
    unsigned int len = m_grid_index.getNumElements();
    std::vector<unsigned int> coords(m_grid_index.getDimension()); 

    for (unsigned int grid_idx = 0; grid_idx < len; grid_idx++)
        {
        // obtain d-dimensional coordinates
        m_grid_index.getCoordinates(grid_idx, coords);

        for (unsigned int cv_idx = 0; cv_idx < m_variables.size(); ++cv_idx)
            {
            Scalar delta = (m_variables[cv_idx].m_cv_max - m_variables[cv_idx].m_cv_min)/
                           (m_variables[cv_idx].m_num_points - 1);
            Scalar val = m_variables[cv_idx].m_cv_min + coords[cv_idx]*delta;

            file << setprecision(10) << val << m_delimiter;
            }

        file << setprecision(10) << h_grid.data[grid_idx];

        file << std::endl;
        }

    if (m_compute_histograms)
        {
        assert(m_variables.size() == 1);
        ArrayHandle<Scalar> h_histogram(m_ftm_histogram, access_location::host, access_mode::read);

        file << std::endl;

        for (unsigned int i = 0; i < m_variables[0].m_num_points; ++i)
            {
            Scalar delta = (m_variables[0].m_cv_max - m_variables[0].m_cv_min)/
                           (m_variables[0].m_num_points - 1);
            Scalar val = m_variables[0].m_cv_min + i*delta;

            file << setprecision(10) << val << m_delimiter;
            file << setprecision(10) << h_histogram.data[i];
            file << std::endl;
            }

        ArrayHandle<Scalar> h_histogram_plus(m_ftm_histogram_plus, access_location::host, access_mode::read);

        file << std::endl;

        for (unsigned int i = 0; i < m_variables[0].m_num_points; ++i)
            {
            Scalar delta = (m_variables[0].m_cv_max - m_variables[0].m_cv_min)/
                           (m_variables[0].m_num_points - 1);
            Scalar val = m_variables[0].m_cv_min + i*delta;

            file << setprecision(10) << val << m_delimiter;
            file << setprecision(10) << h_histogram_plus.data[i];
            file << std::endl;
            }
 
        }

    file.close();
    }

void IntegratorMetaDynamics::readGrid(const std::string& filename)
    {
#ifdef ENABLE_MPI
    // Only on root processor
    if (m_pdata->getDomainDecomposition())
        if (! m_exec_conf->isRoot()) return;
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

    // Skip first two lines of file header
    getline(file, line);
    getline(file, line);
   
    // read number of histogram entries
    getline(file, line);
    istringstream iss(line);
    std::string tmp;
    iss >> tmp >> m_num_histogram_entries;

    // Skip last header line
    getline(file, line);

    unsigned int len = m_grid_index.getNumElements();
    std::vector<unsigned int> coords(m_grid_index.getDimension()); 
    ArrayHandle<Scalar> h_grid(m_grid, access_location::host, access_mode::overwrite);
    for (unsigned int grid_idx = 0; grid_idx < len; grid_idx++)
        {
        if (! file.good())
            {
            m_exec_conf->msg->error() << "integrate.mode_metadynamics: Premature end of grid file.";
            throw std::runtime_error("Error reading grid.");
            }
     
        getline(file, line);
        istringstream iss(line);

        // skip values of collective variables
        for (unsigned int i = 0; i < m_variables.size(); i++)
            iss >> tmp;

        iss >> h_grid.data[grid_idx];
        }
    
    if (m_compute_histograms)
        {
        // read in histograms
        assert(m_variables.size() == 1);

        ArrayHandle<Scalar> h_histogram(m_ftm_histogram, access_location::host, access_mode::overwrite);
        ArrayHandle<Scalar> h_histogram_plus(m_ftm_histogram_plus, access_location::host, access_mode::overwrite);
        // skip one line
        getline(file, line);

        // read in equilibrium histogram
        for (unsigned int idx = 0; idx < len; ++idx)
            {
            if (! file.good())
                {
                m_exec_conf->msg->error() << "integrate.mode_metadynamics: Premature end of grid file.";
                throw std::runtime_error("Error reading grid.");
                }
            getline(file, line);
            istringstream iss(line);

            // skip value of collective variable
            iss >> tmp;

            iss >> h_histogram.data[idx];
            }

        // skip one line
        getline(file, line);

        // read in plus-state histogram
        for (unsigned int idx = 0; idx < len; ++idx)
            {
            if (! file.good())
                {
                m_exec_conf->msg->error() << "integrate.mode_metadynamics: Premature end of grid file.";
                throw std::runtime_error("Error reading grid.");
                }
            getline(file, line);
            istringstream iss(line);

            // skip value of collective variable
            iss >> tmp;

            iss >> h_histogram_plus.data[idx];
            }
        }

    file.close();

    } 

void IntegratorMetaDynamics::updateGrid(std::vector<Scalar>& current_val, Scalar scal)
    {
    if (m_prof) m_prof->push("update grid");

    if (m_mode == mode_flux_tempered && m_compute_histograms == false)
        {
        m_exec_conf->msg->error() << "integrate.mode_metadynamics: Need to enable histograms for flux-tempered metadynamics." << std::endl;
        throw std::runtime_error("Error updating grid.");
        }

    ArrayHandle<Scalar> h_grid(m_grid, access_location::host, access_mode::readwrite);

    ArrayHandle<Scalar> h_histogram(m_ftm_histogram, access_location::host, access_mode::read);

    // loop over grid
    unsigned int len = m_grid_index.getNumElements();
    std::vector<unsigned int> coords(m_grid_index.getDimension()); 
    for (unsigned int grid_idx = 0; grid_idx < len; grid_idx++)
        {
        // obtain d-dimensional coordinates
        m_grid_index.getCoordinates(grid_idx, coords);

        if (m_mode == mode_standard || m_mode == mode_well_tempered)
            {
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
            h_grid.data[grid_idx] += m_W*scal*gauss;
            }
        else if (m_mode == mode_flux_tempered)
            {
            assert(m_variables.size()==1);

            Scalar grid_delta = (m_variables[0].m_cv_max - m_variables[0].m_cv_min)/
                               (Scalar)(m_variables[0].m_num_points - 1);
            Scalar val = m_variables[0].m_cv_min + coords[0]*grid_delta;

            Scalar dfds = fractionDerivative(val);
            Scalar hist = h_histogram.data[coords[0]];

            // normalize histogram
            hist /= m_num_histogram_entries; 

            Scalar del = -Scalar(1.0/2.0)*m_temp*(log(fabsf(dfds)) - log(hist));
            h_grid.data[grid_idx] += del;
            }
        }

    if (m_prof) m_prof->pop();
    }

#ifdef ENABLE_CUDA
void IntegratorMetaDynamics::updateGridGPU(std::vector<Scalar>& current_val, Scalar scal)
    {
    if (m_prof)
        m_prof->push(m_exec_conf, "update grid");

        { 
        // copy current CV values into array
        ArrayHandle<Scalar> h_current_val(m_current_val, access_location::host, access_mode::overwrite);

        for (unsigned int cv_idx = 0; cv_idx < m_variables.size(); cv_idx++)
            h_current_val.data[cv_idx] = current_val[cv_idx];
        }

    ArrayHandle<Scalar> d_grid(m_grid, access_location::device, access_mode::readwrite);
    ArrayHandle<unsigned int> d_lengths(m_lengths, access_location::device, access_mode::read);
    ArrayHandle<Scalar> d_cv_min(m_cv_min, access_location::device, access_mode::read);
    ArrayHandle<Scalar> d_cv_max(m_cv_max, access_location::device, access_mode::read);
    ArrayHandle<Scalar> d_sigma(m_sigma, access_location::device, access_mode::read);
    ArrayHandle<Scalar> d_current_val(m_current_val, access_location::device, access_mode::read);

    ArrayHandle<Scalar> d_histogram(m_ftm_histogram, access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar> d_histogram_plus(m_ftm_histogram_plus, access_location::device, access_mode::readwrite);

    gpu_update_grid(m_grid_index.getNumElements(),
                    d_lengths.data,
                    m_variables.size(),
                    d_current_val.data,
                    d_grid.data,
                    d_cv_min.data,
                    d_cv_max.data,
                    d_sigma.data,
                    scal,
                    m_W,
                    (m_mode == mode_flux_tempered),
                    m_temp,
                    d_histogram.data,
                    d_histogram_plus.data,
                    m_num_histogram_entries,
                    m_variables[0].m_ftm_min,
                    m_variables[0].m_ftm_max);

    if (m_prof) m_prof->pop(m_exec_conf);
    }
#endif


void IntegratorMetaDynamics::setupHistograms()
    {
    assert(m_variables.size() == 1);

    unsigned int num_points = m_variables[0].m_num_points;

    GPUArray<Scalar> histogram(num_points, m_exec_conf);
    m_ftm_histogram.swap(histogram);

    GPUArray<Scalar> histogram_plus(num_points, m_exec_conf);
    m_ftm_histogram_plus.swap(histogram_plus);

    resetHistograms();
    }

void IntegratorMetaDynamics::resetHistograms()
    {
    assert(m_variables.size() == 1);
    unsigned int num_points = m_variables[0].m_num_points;
    ArrayHandle<Scalar> h_histogram(m_ftm_histogram, access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar> h_histogram_plus(m_ftm_histogram_plus, access_location::host, access_mode::overwrite);

    memset(h_histogram.data, 0, num_points*sizeof(Scalar));
    memset(h_histogram_plus.data, 0, num_points*sizeof(Scalar));
    } 

void IntegratorMetaDynamics::sampleHistograms(Scalar val, bool state)
    {
    assert(m_variables.size()==1);

    Scalar min = m_variables[0].m_cv_min;
    Scalar max = m_variables[0].m_cv_max;
    unsigned int num_points = m_variables[0].m_num_points;
    Scalar delta = (max-min)/(Scalar)(num_points-1);
    Scalar sigma = m_variables[0].m_sigma;

    Scalar norm = Scalar(1.0)/sqrt(2.0*M_PI*sigma*sigma);

    ArrayHandle<Scalar> h_histogram(m_ftm_histogram, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> h_histogram_plus(m_ftm_histogram_plus, access_location::host, access_mode::readwrite);

    for (unsigned int i = 0; i < num_points; ++i)
        {
        Scalar val_grid = min + i*delta;
        double d = val - val_grid;
        Scalar gauss_exp = d*d/2.0/sigma/sigma;
        double gauss = norm*exp(-gauss_exp);

        // add Gaussian to grid
        h_histogram.data[i] += gauss;
        if (state == true)
            h_histogram_plus.data[i] += gauss;
        }
    }

#ifdef ENABLE_CUDA
void IntegratorMetaDynamics::sampleHistogramsGPU(Scalar val, bool state)
    {
    assert(m_variables.size()==1);

    Scalar min = m_variables[0].m_cv_min;
    Scalar max = m_variables[0].m_cv_max;
    unsigned int num_points = m_variables[0].m_num_points;
    Scalar delta = (max-min)/(Scalar)(num_points-1);
    Scalar sigma = m_variables[0].m_sigma;

    ArrayHandle<Scalar> d_histogram(m_ftm_histogram, access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar> d_histogram_plus(m_ftm_histogram_plus, access_location::device, access_mode::readwrite);

    gpu_update_histograms(val,
                          min,
                          delta,
                          num_points,
                          sigma,
                          state,
                          d_histogram.data,
                          d_histogram_plus.data);
    }
#endif
 
Scalar IntegratorMetaDynamics::interpolateHistogram(Scalar val,bool fraction)
    {
    assert(m_variables.size()==1);

    Scalar min = m_variables[0].m_cv_min;
    Scalar max = m_variables[0].m_cv_max;
    unsigned int num_points = m_variables[0].m_num_points;

    Scalar delta = (max-min)/(num_points-1);

    int lower_bin = (int) ((val - min)/delta);
    unsigned int upper_bin = lower_bin+1;

    if (lower_bin < 0 || upper_bin >= num_points)
        {
        m_exec_conf->msg->warning() << "integrate.mode_metadynamics: Value " << val
                                    << " of collective variable outside range for histogram." << std::endl
                                    << "Assuming frequency of zero." << std::endl;
        return 0;
        }

    Scalar rel_delta = (val - (Scalar)lower_bin*delta)/delta;

    ArrayHandle<Scalar> h_histogram(m_ftm_histogram, access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_histogram_plus(m_ftm_histogram_plus, access_location::host, access_mode::read);

    Scalar lower_val;
    Scalar upper_val;

    if (fraction)
        {
        lower_val = h_histogram_plus.data[lower_bin]/h_histogram.data[lower_bin];
        upper_val= h_histogram_plus.data[upper_bin]/h_histogram.data[upper_bin];
        }
    else
        {
        lower_val = h_histogram.data[lower_bin];
        upper_val= h_histogram.data[upper_bin];
        }

    return lower_val + rel_delta*(upper_val - lower_val);
    }


Scalar IntegratorMetaDynamics::fractionDerivative(Scalar val)
    {
    assert(m_variables.size() == 1);
    Scalar delta = (m_variables[0].m_cv_max - m_variables[0].m_cv_min)/m_variables[0].m_num_points;

    if (val - delta < m_variables[0].m_cv_min) 
        {
        // forward difference
        Scalar val2 = val + delta;

        Scalar y2 = interpolateHistogram(val2,true);
        Scalar y1 = interpolateHistogram(val,true);
        return (y2-y1)/delta;
        }
    else if (val + delta > m_variables[0].m_cv_max)
        {
        // backward difference
        Scalar val2 = val - delta;
        Scalar y1 = interpolateHistogram(val2,true);
        Scalar y2 = interpolateHistogram(val,true);
        return (y2-y1)/delta;
        }
    else
        {
        // central difference
        Scalar val2 = val + delta;
        Scalar val1 = val - delta;
        Scalar y1 = interpolateHistogram(val1,true);
        Scalar y2 = interpolateHistogram(val2,true);
        return (y2 - y1)/(Scalar(2.0)*delta);
        }
    } 

void export_IntegratorMetaDynamics()
    {
    scope in_metad = class_<IntegratorMetaDynamics, boost::shared_ptr<IntegratorMetaDynamics>, bases<IntegratorTwoStep>, boost::noncopyable>
    ("IntegratorMetaDynamics", init< boost::shared_ptr<SystemDefinition>,
                          Scalar,
                          Scalar,
                          Scalar,
                          Scalar,
                          unsigned int,
                          bool,
                          const std::string&,
                          bool,
                          IntegratorMetaDynamics::Enum>())
    .def("registerCollectiveVariable", &IntegratorMetaDynamics::registerCollectiveVariable)
    .def("removeAllVariables", &IntegratorMetaDynamics::removeAllVariables)
    .def("isInitialized", &IntegratorMetaDynamics::isInitialized)
    .def("setGrid", &IntegratorMetaDynamics::setGrid)
    .def("dumpGrid", &IntegratorMetaDynamics::dumpGrid)
    .def("restartFromGridFile", &IntegratorMetaDynamics::restartFromGridFile)
    .def("setAddHills", &IntegratorMetaDynamics::setAddHills)
    .def("setHistograms", &IntegratorMetaDynamics::setHistograms)
    .def("setMode", &IntegratorMetaDynamics::setMode)
    .def("setStride", &IntegratorMetaDynamics::setStride)
    .def("setStrideMultiply", &IntegratorMetaDynamics::setStrideMultiply)
    .def("setMinimumLabelChanges", &IntegratorMetaDynamics::setMinimumLabelChanges)
    ;

    enum_<IntegratorMetaDynamics::Enum>("mode")
    .value("standard", IntegratorMetaDynamics::mode_standard)
    .value("well_tempered", IntegratorMetaDynamics::mode_well_tempered)
    .value("flux_tempered", IntegratorMetaDynamics::mode_flux_tempered)
    ;
    }
