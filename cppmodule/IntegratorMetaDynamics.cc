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
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/lu.hpp>

#ifdef ENABLE_CUDA
#include "IntegratorMetaDynamics.cuh"
#endif 

using namespace boost::python;
using namespace boost::filesystem;
namespace bnu = boost::numeric::ublas;

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
      m_histograms_initialized(false),
      m_filename(filename),
      m_overwrite(overwrite),
      m_is_appending(false),
      m_delimiter("\t"),
      m_use_grid(false),
      m_num_biased_variables(0),
      m_add_bias(add_bias),
      m_restart_filename(""),
      m_grid_fname1(""),
      m_grid_fname2(""),
      m_grid_period(0),
      m_cur_file(0),
      m_sigma_g(1.0),
      m_adaptive(false),
      m_temp(T),
      m_mode(mode),
      m_stride_multiply(1),
      m_num_label_change(0),
      m_min_label_change(0),
      m_umbrella_factor(Scalar(1.0))
    {
    assert(m_T_shift>0);
    assert(m_W > 0);

    m_log_names.push_back("bias_potential");
    m_log_names.push_back("det_sigma");

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

    std::vector<CollectiveVariableItem>::iterator it,itj;
    for (it = m_variables.begin(); it != m_variables.end(); ++it)
        {
        if (it->m_umbrella) continue;

        m_file << it->m_cv->getName();

        for (itj = m_variables.begin(); itj != m_variables.end(); ++itj)
            m_file << m_delimiter << "sigma_" << it->m_cv->getName() << "_"
                   << it -m_variables.begin() << "_" << itj - m_variables.begin();

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
            m_cv_values.resize(m_num_biased_variables);
            std::vector< std::vector<Scalar> >::iterator it;

            for (it = m_cv_values.begin(); it != m_cv_values.end(); ++it)
                it->clear();

            // initialize GPU mirror values for collective variable data
            GPUArray<Scalar> cv_min(m_num_biased_variables, m_exec_conf);
            m_cv_min.swap(cv_min);

            GPUArray<Scalar> cv_max(m_num_biased_variables, m_exec_conf);
            m_cv_max.swap(cv_max);

            GPUArray<Scalar> current_val(m_num_biased_variables, m_exec_conf);
            m_current_val.swap(current_val);

            GPUArray<unsigned int> lengths(m_num_biased_variables, m_exec_conf);
            m_lengths.swap(lengths);

            GPUArray<Scalar> sigma(m_num_biased_variables*m_num_biased_variables, m_exec_conf);
            m_sigma.swap(sigma);

            ArrayHandle<Scalar> h_cv_min(m_cv_min, access_location::host, access_mode::overwrite);
            ArrayHandle<Scalar> h_cv_max(m_cv_max, access_location::host, access_mode::overwrite);
            ArrayHandle<unsigned int> h_lengths(m_lengths, access_location::host, access_mode::overwrite);
            ArrayHandle<Scalar> h_sigma(m_sigma, access_location::host, access_mode::overwrite);
           
            memset(h_sigma.data, 0, sizeof(Scalar)*m_num_biased_variables*m_num_biased_variables);

            unsigned int idx = 0;
            for (unsigned int cv_idx = 0; cv_idx < m_variables.size(); cv_idx++)
                {
                if (m_variables[cv_idx].m_umbrella) continue;

                h_cv_min.data[idx] = m_variables[cv_idx].m_cv_min;
                h_cv_max.data[idx] = m_variables[cv_idx].m_cv_max;
                h_sigma.data[idx*m_variables.size()+idx] = m_variables[cv_idx].m_sigma;
                h_lengths.data[idx] = m_variables[cv_idx].m_num_points;
                idx++;
                }
            
            m_num_update_steps = 0;
            m_bias_potential.clear();
            }

        // Set up histograms if necessary
        if (! m_histograms_initialized && m_compute_histograms)
            {
            if (m_num_biased_variables != 1)
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
    if (m_num_biased_variables == 0)
        return;

    // collect values of collective variables
    std::vector< Scalar> current_val;
    std::vector<CollectiveVariableItem>::iterator it;
    for (it = m_variables.begin(); it != m_variables.end(); ++it)
        {
        if (it->m_umbrella) continue;
        Scalar val = it->m_cv->getCurrentValue(timestep);
        current_val.push_back(val);
        }

    std::vector<Scalar> bias(m_num_biased_variables, 0.0); 

    bool is_root = true;

   if (m_adaptive && (m_num_update_steps % m_stride == 0))
        {
        // compute derivatives of collective variables
        for (unsigned int i = 0; i < m_num_biased_variables; ++i)
            m_variables[i].m_cv->computeDerivatives(timestep);

        // compute instantaneous estimate of standard deviation matrix
        computeSigma();
        } 

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
            for (unsigned int i = 0; i < m_num_biased_variables; ++i)
                {
                m_cv_values[i].push_back(current_val[i]);
                }
            }

        if (m_compute_histograms)
            {
            assert(m_num_biased_variables == 1);
            Scalar val = current_val[0];

            unsigned int idx = 0;
            while (m_variables[idx].m_umbrella && idx < m_variables.size()) idx++;

            assert(!m_variables[idx].m_umbrella);

            // change walker state if necessary
            Scalar min = m_variables[idx].m_ftm_min;
            Scalar max = m_variables[idx].m_ftm_max;
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

                // add up umbrella potentials for reweighting
                Scalar umbrella_energy = Scalar(0.0);
                for (unsigned int i = 0; i < m_variables.size(); ++i)
                    {
                    if (m_variables[i].m_umbrella)
                        {
                        Scalar val = m_variables[i].m_cv->getUmbrellaPotential(timestep);
                        umbrella_energy += val;
                        }
                    }

                // reweight by Boltzmann factor
                m_umbrella_factor = exp(umbrella_energy/m_temp);

#ifdef ENABLE_CUDA
                if (m_exec_conf->isCUDAEnabled())
                    updateGridGPU(current_val, scal, m_umbrella_factor);
                else
                    updateGrid(current_val, scal,m_umbrella_factor);
#else
                updateGrid(current_val, scal, m_umbrella_factor);
#endif

                // reset statistics
                if (m_mode == mode_flux_tempered)
                    {
                    resetHistograms();
                    m_num_label_change = 0;
                    }
                }

            // calculate partial derivatives numerically
            for (unsigned int cv_idx = 0; cv_idx < m_num_biased_variables; ++cv_idx)
                bias[cv_idx] = biasPotentialDerivative(cv_idx, current_val);

            } 
        else  //!m_use_grid
            {
            if (m_mode == mode_flux_tempered)
                {
                m_exec_conf->msg->error() << "integrate.mode_metadynamics: Flux-tempered MetaD is only supported in grid mode" << std::endl;
                throw std::runtime_error("Error in metadynamics integration.");
                }

            if (m_adaptive)
                {
                m_exec_conf->msg->error() << "integrate.mode_metadynamics: Adaptive Gaussians only available in grid mode" << std::endl;
                throw std::runtime_error("Error in metadynamics integration.");
                }

            if (m_variables.size() != m_num_biased_variables)
                {
                m_exec_conf->msg->error() << "integrate.mode_metadynamics: Reweighting supported only in grid mode." << std::endl;
                throw std::runtime_error("Error in metadynamics integration.");
                }



            ArrayHandle<Scalar> h_sigma(m_sigma, access_location::host, access_mode::read);

            // sum up all Gaussians accumulated until now
            for (unsigned int gauss_idx = 0; gauss_idx < m_bias_potential.size(); ++gauss_idx)
                {
                Scalar gauss_exp = 0.0;
                // calculate Gaussian contribution from t'=gauss_idx*m_stride
                for (unsigned int i = 0; i < m_num_biased_variables; ++i)
                    {
                    Scalar vali = current_val[i];
                    Scalar delta_i = vali - m_cv_values[i][gauss_idx];

                    for (unsigned int j = 0; j < m_num_biased_variables; ++j)
                        {
                        Scalar valj = current_val[j];
                        Scalar delta_j = valj - m_cv_values[j][gauss_idx];

                        Scalar sigmaij = h_sigma.data[i*m_num_biased_variables+j];

                        gauss_exp += delta_i*delta_j*Scalar(1.0/2.0)/(sigmaij*sigmaij);
                        }
                    }
                Scalar gauss = exp(-gauss_exp);

                // calculate partial derivatives

                // scaling factor for well-tempered MetaD
                Scalar scal = Scalar(1.0);
                if (m_mode == mode_well_tempered)
                    scal = exp(-m_bias_potential[gauss_idx]/m_T_shift);

                for (unsigned int i = 0; i < m_num_biased_variables; ++i)
                    {
                    Scalar val_i = current_val[i];

                    for (unsigned int j = 0; j < m_num_biased_variables; ++j)
                        {
                        Scalar val_j = current_val[j];

                        Scalar sigmaij = h_sigma.data[i*m_num_biased_variables+j];
                        
                        bias[i] -= Scalar(1.0/2.0)*m_W*scal/(sigmaij*sigmaij)*(val_j - m_cv_values[j][gauss_idx])*gauss;
                        bias[j] -= Scalar(1.0/2.0)*m_W*scal/(sigmaij*sigmaij)*(val_i - m_cv_values[i][gauss_idx])*gauss;
                        }
                    }

                m_curr_bias_potential += m_W*scal*gauss;
                }
            }

        // write hills information
        if (m_is_initialized && (m_num_update_steps % m_stride == 0) && m_add_bias && m_file.is_open())
            {
            ArrayHandle<Scalar> h_sigma(m_sigma, access_location::host, access_mode::read);

            Scalar W = m_W*exp(-m_curr_bias_potential/m_T_shift);
            m_file << setprecision(10) << timestep << m_delimiter;
            m_file << setprecision(10) << W << m_delimiter;

            std::vector<Scalar>::iterator cv,cvj;
            for (cv = current_val.begin(); cv != current_val.end(); ++cv)
                {
                unsigned int cv_index = cv - current_val.begin();
                m_file << setprecision(10) << *cv << m_delimiter;

                // Write row of sigma matrix
                for (cvj = current_val.begin(); cvj != current_val.end(); ++cvj)
                    {
                    unsigned int cv_index_j = cvj - current_val.begin();
                    Scalar sigmaij = h_sigma.data[cv_index*m_num_biased_variables+cv_index_j];
                    m_file << setprecision(10) << sigmaij;
                    }

                if (cv != current_val.end() -1) m_file << m_delimiter;
                }

            m_file << endl;
            }
       
        if (m_add_bias && (! m_use_grid) && (m_num_update_steps % m_stride == 0))
            m_bias_potential.push_back(m_curr_bias_potential);

        // update stride
        if (m_num_update_steps && (m_num_update_steps % m_stride == 0))
            m_stride *= m_stride_multiply;

        if (m_adaptive && m_use_grid)
            {
            // update sigma grid and histogram
            updateSigmaGrid(current_val,m_umbrella_factor);
            }

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

    
        } // endif root processor

    // increment number of updated steps
    m_num_update_steps++;

#ifdef ENABLE_MPI
    // broadcast bias factors
    if (m_pdata->getDomainDecomposition())
        MPI_Bcast(&bias.front(), bias.size(), MPI_HOOMD_SCALAR, 0, m_exec_conf->getMPICommunicator());

#endif

    // update current bias potential derivative for every collective variable
    std::vector<CollectiveVariableItem>::iterator cv_item;
    unsigned int cv = 0;
    for (cv_item = m_variables.begin(); cv_item != m_variables.end(); ++cv_item)
        {
        if (cv_item->m_umbrella) continue;

        cv_item->m_cv->setBiasFactor(bias[cv]);
        cv++;
        }

    if (m_prof)
        m_prof->pop();
    }

void IntegratorMetaDynamics::setupGrid()
    {
    assert(! m_is_initialized);
    assert(m_num_biased_variables);

    std::vector< CollectiveVariableItem >::iterator it;

    std::vector< unsigned int > lengths(m_num_biased_variables);

    unsigned int idx = 0;
    for (it = m_variables.begin(); it != m_variables.end(); ++it)
        {
        if (it->m_umbrella) continue;
        lengths[idx] = it->m_num_points;
        idx++;
        }

    m_grid_index.setLengths(lengths);

    GPUArray<Scalar> grid(m_grid_index.getNumElements(),m_exec_conf);
    m_grid.swap(grid);

    GPUArray<Scalar> reweighted_grid(m_grid_index.getNumElements(),m_exec_conf);
    m_reweighted_grid.swap(reweighted_grid);

    // reset grid
    ArrayHandle<Scalar> h_grid(m_grid, access_location::host, access_mode::overwrite);
    memset(h_grid.data, 0, sizeof(Scalar)*m_grid.getNumElements());

    ArrayHandle<Scalar> h_reweighted_grid(m_reweighted_grid, access_location::host, access_mode::overwrite);
    memset(h_reweighted_grid.data, 0, sizeof(Scalar)*m_reweighted_grid.getNumElements());
 
    GPUArray<Scalar> sigma_grid(m_grid_index.getNumElements(),m_exec_conf);
    m_sigma_grid.swap(sigma_grid);

    GPUArray<Scalar> sigma_reweight_grid(m_grid_index.getNumElements(),m_exec_conf);
    m_sigma_reweight_grid.swap(sigma_reweight_grid);

    GPUArray<unsigned int> grid_hist(m_grid_index.getNumElements(),m_exec_conf);
    m_grid_hist.swap(grid_hist);

    ArrayHandle<Scalar> h_sigma_grid(m_sigma_grid, access_location::host, access_mode::overwrite);
    memset(h_sigma_grid.data, 0, sizeof(Scalar)*m_sigma_grid.getNumElements());

    ArrayHandle<Scalar> h_sigma_reweight_grid(m_sigma_reweight_grid, access_location::host, access_mode::overwrite);
    memset(h_sigma_reweight_grid.data, 0, sizeof(Scalar)*m_sigma_reweight_grid.getNumElements());

    ArrayHandle<unsigned int> h_grid_hist(m_grid_hist, access_location::host, access_mode::overwrite);
    memset(h_grid_hist.data, 0, sizeof(Scalar)*m_grid_hist.getNumElements());
   } 

Scalar IntegratorMetaDynamics::interpolateBiasPotential(const std::vector<Scalar>& val)
    {
    assert(val.size() == m_grid_index.getDimension());

    // find closest d-dimensional sub-block
    std::vector<unsigned int> lower_idx(m_grid_index.getDimension());
    std::vector<unsigned int> upper_idx(m_grid_index.getDimension());
    std::vector<Scalar> rel_delta(m_grid_index.getDimension());

    unsigned int cv = 0;
    for (unsigned int cv_idx = 0; cv_idx < m_variables.size(); cv_idx++)
        {
        if (m_variables[cv_idx].m_umbrella) continue;

        Scalar delta = (m_variables[cv_idx].m_cv_max - m_variables[cv_idx].m_cv_min)/(m_variables[cv_idx].m_num_points - 1);
        int lower = (int) ((val[cv] - m_variables[cv_idx].m_cv_min)/delta);
        int upper = lower+1;

        if (lower < 0 || upper >= m_variables[cv_idx].m_num_points)
            {
            m_exec_conf->msg->warning() << "integrate.mode_metadynamics: Value " << val[cv]
                                        << " of collective variable " << cv_idx << " out of bounds." << endl
                                        << "Assuming bias potential of zero." << endl;
            return Scalar(0.0);
            }

        Scalar lower_bound = m_variables[cv_idx].m_cv_min + delta * lower;
        Scalar upper_bound = m_variables[cv_idx].m_cv_min + delta * upper;
        lower_idx[cv_idx] = lower;
        upper_idx[cv_idx] = upper;
        rel_delta[cv_idx] = (val[cv]-lower_bound)/(upper_bound-lower_bound);

        cv++;
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
    ArrayHandle<Scalar> h_cv_min(m_cv_min, access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_cv_max(m_cv_max, access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_lengths(m_lengths, access_location::host, access_mode::read);


    unsigned int cv_idx = 0;
    for (unsigned int i = 0; i < m_variables.size(); ++i)
        {
        if (m_variables[i].m_umbrella) continue;
        if (cv_idx == cv) break;
        cv_idx++;
        }

    Scalar delta = (h_cv_max.data[cv_idx] - h_cv_min.data[cv_idx])/
                   (Scalar)(h_lengths.data[cv_idx] - 1);
    if (val[cv] - delta < m_variables[cv_idx].m_cv_min) 
        {
        // forward difference
        std::vector<Scalar> val2 = val;
        val2[cv] += delta;

        Scalar y2 = interpolateBiasPotential(val2);
        Scalar y1 = interpolateBiasPotential(val);
        return (y2-y1)/delta;
        }
    else if (val[cv] + delta > m_variables[cv_idx].m_cv_max)
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
            if (it->m_umbrella) continue;

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
        {
        if (m_variables[i].m_umbrella) continue;

        file << m_variables[i].m_cv->getName() << m_delimiter;
        }


    file << "grid";

    file << m_delimiter << "grid_reweight";
    file << m_delimiter << "det_sigma";
    file << m_delimiter << "det_sigma_reweight";
    file << m_delimiter << "hist";

    file << std::endl;

    // loop over grid
    ArrayHandle<Scalar> h_grid(m_grid, access_location::host, access_mode::read);
    unsigned int len = m_grid_index.getNumElements();
    std::vector<unsigned int> coords(m_grid_index.getDimension()); 

    ArrayHandle<Scalar> h_sigma_grid(m_sigma_grid, access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_sigma_reweight_grid(m_sigma_reweight_grid, access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_grid_hist(m_grid_hist, access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_reweighted_grid(m_reweighted_grid, access_location::host, access_mode::read);

    for (unsigned int grid_idx = 0; grid_idx < len; grid_idx++)
        {
        // obtain d-dimensional coordinates
        m_grid_index.getCoordinates(grid_idx, coords);
        
        unsigned int cv = 0;
        for (unsigned int cv_idx = 0; cv_idx < m_variables.size(); ++cv_idx)
            {
            if (m_variables[cv_idx].m_umbrella) continue;

            Scalar delta = (m_variables[cv_idx].m_cv_max - m_variables[cv_idx].m_cv_min)/
                           (m_variables[cv_idx].m_num_points - 1);
            Scalar val = m_variables[cv_idx].m_cv_min + coords[cv]*delta;

            file << setprecision(10) << val << m_delimiter;
            cv++;
            }

        file << setprecision(10) << h_grid.data[grid_idx];
        file << m_delimiter << setprecision(10) << h_reweighted_grid.data[grid_idx];

        // write average of Gaussian volume
        Scalar val,val_reweight;
        if (h_grid_hist.data[grid_idx] > 0)
            {
            val = h_sigma_grid.data[grid_idx]/(Scalar)h_grid_hist.data[grid_idx];
            val_reweight = h_sigma_reweight_grid.data[grid_idx]/(Scalar)h_grid_hist.data[grid_idx];
            }
        else
            {
            val = Scalar(0.0);
            val_reweight = Scalar(0.0);
            }

        file << m_delimiter << setprecision(10) << val;
        file << m_delimiter << setprecision(10) << val_reweight;
        file << m_delimiter << h_grid_hist.data[grid_idx];

        file << std::endl;
        }

    if (m_compute_histograms)
        {
        assert(m_num_biased_variables == 1);
        ArrayHandle<Scalar> h_histogram(m_ftm_histogram, access_location::host, access_mode::read);

        file << std::endl;
    
        unsigned int idx = 0;
        while (m_variables[idx].m_umbrella && idx < m_variables.size()) idx++;

        assert(! m_variables[idx].m_umbrella);

        for (unsigned int i = 0; i < m_variables[idx].m_num_points; ++i)
            {
            Scalar delta = (m_variables[idx].m_cv_max - m_variables[idx].m_cv_min)/
                           (m_variables[idx].m_num_points - 1);
            Scalar val = m_variables[idx].m_cv_min + i*delta;

            file << setprecision(10) << val << m_delimiter;
            file << setprecision(10) << h_histogram.data[i];
            file << std::endl;
            }

        ArrayHandle<Scalar> h_histogram_plus(m_ftm_histogram_plus, access_location::host, access_mode::read);

        file << std::endl;

        for (unsigned int i = 0; i < m_variables[idx].m_num_points; ++i)
            {
            Scalar delta = (m_variables[idx].m_cv_max - m_variables[idx].m_cv_min)/
                           (m_variables[idx].m_num_points - 1);
            Scalar val = m_variables[idx].m_cv_min + i*delta;

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

    ArrayHandle<Scalar> h_sigma_grid(m_sigma_grid, access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar> h_sigma_reweight_grid(m_sigma_reweight_grid, access_location::host, access_mode::overwrite);
    ArrayHandle<unsigned int> h_grid_hist(m_grid_hist, access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar> h_reweighted_grid(m_reweighted_grid, access_location::host, access_mode::overwrite);

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
        for (unsigned int i = 0; i < m_num_biased_variables; i++)
            iss >> tmp;

        iss >> h_grid.data[grid_idx];

        iss >> h_reweighted_grid.data[grid_idx];
        iss >> h_sigma_grid.data[grid_idx];
        iss >> h_sigma_reweight_grid.data[grid_idx];
        iss >> h_grid_hist.data[grid_idx];

        h_sigma_grid.data[grid_idx] *= h_grid_hist.data[grid_idx];
        h_sigma_reweight_grid.data[grid_idx] *= h_grid_hist.data[grid_idx];
        }
    
    if (m_compute_histograms)
        {
        // read in histograms
        assert(m_num_biased_variables == 1);

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

void IntegratorMetaDynamics::updateGrid(std::vector<Scalar>& current_val, Scalar scal, Scalar reweight )
    {
    if (m_prof) m_prof->push("update grid");

    if (m_mode == mode_flux_tempered && m_compute_histograms == false)
        {
        m_exec_conf->msg->error() << "integrate.mode_metadynamics: Need to enable histograms for flux-tempered metadynamics." << std::endl;
        throw std::runtime_error("Error updating grid.");
        }

    ArrayHandle<Scalar> h_grid(m_grid, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> h_reweighted_grid(m_reweighted_grid, access_location::host, access_mode::readwrite);

    ArrayHandle<Scalar> h_histogram(m_ftm_histogram, access_location::host, access_mode::read);

    // loop over grid
    unsigned int len = m_grid_index.getNumElements();
    std::vector<unsigned int> coords(m_grid_index.getDimension()); 

    ArrayHandle<Scalar> h_sigma(m_sigma, access_location::host, access_mode::read);

    unsigned int idx = 0;
    if (m_mode == mode_flux_tempered)
        while (m_variables[idx].m_umbrella && idx < m_variables.size()) idx++;


    for (unsigned int grid_idx = 0; grid_idx < len; grid_idx++)
        {
        // obtain d-dimensional coordinates
        m_grid_index.getCoordinates(grid_idx, coords);

        if (m_mode == mode_standard || m_mode == mode_well_tempered)
            {
            Scalar gauss_exp(0.0);
            // evaluate Gaussian on grid point
            for (unsigned int cv_i = 0; cv_i < m_variables.size(); ++cv_i)
                {
                // do not include umbrella variables in metadynamics
                if (m_variables[cv_i].m_umbrella) continue;

                Scalar delta_i = (m_variables[cv_i].m_cv_max - m_variables[cv_i].m_cv_min)/
                               (m_variables[cv_i].m_num_points - 1);
                Scalar val_i = m_variables[cv_i].m_cv_min + coords[cv_i]*delta_i;
                double d_i = val_i - current_val[cv_i];

                for (unsigned int cv_j = 0; cv_j < m_variables.size(); ++cv_j)
                    {
                    if (m_variables[cv_j].m_umbrella) continue;

                    Scalar delta_j = (m_variables[cv_j].m_cv_max - m_variables[cv_j].m_cv_min)/
                                   (m_variables[cv_j].m_num_points - 1);
                    Scalar val_j = m_variables[cv_j].m_cv_min + coords[cv_j]*delta_j;
                    double d_j = val_j - current_val[cv_j];

                    Scalar sigma_ij = h_sigma.data[cv_i*m_variables.size()+cv_j];

                    gauss_exp += d_i*d_j*Scalar(1.0/2.0)/(sigma_ij*sigma_ij);
                    }
                }
            double gauss = exp(-gauss_exp);

            // add Gaussian to grid
            h_grid.data[grid_idx] += m_W*scal*gauss;
            h_reweighted_grid.data[grid_idx] += m_W*reweight*scal*gauss;
            }
        else if (m_mode == mode_flux_tempered)
            {
            assert(m_num_biased_variables==1);

            Scalar grid_delta = (m_variables[idx].m_cv_max - m_variables[idx].m_cv_min)/
                               (Scalar)(m_variables[idx].m_num_points - 1);
            Scalar val = m_variables[idx].m_cv_min + coords[idx]*grid_delta;

            Scalar dfds = fractionDerivative(val);
            Scalar hist = h_histogram.data[coords[idx]];

            // normalize histogram
            hist /= m_num_histogram_entries; 

            Scalar del = -Scalar(1.0/2.0)*m_temp*(log(fabsf(dfds)) - log(hist));
            h_grid.data[grid_idx] += del;
            h_reweighted_grid.data[grid_idx] += reweight*del;
            }
        }

    if (m_prof) m_prof->pop();
    }

void IntegratorMetaDynamics::updateSigmaGrid(std::vector<Scalar>& current_val, Scalar reweight)
    {
    if (m_prof) m_prof->push("update sigma grid");

    ArrayHandle<Scalar> h_sigma_grid(m_sigma_grid, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> h_sigma_reweight_grid(m_sigma_reweight_grid, access_location::host, access_mode::readwrite);
    ArrayHandle<unsigned int> h_grid_hist(m_grid_hist, access_location::host, access_mode::readwrite);

    assert(h_sigma_grid.data);
    assert(h_sigma_reweight_grid.data);

    std::vector<unsigned int> grid_coord(m_variables.size());

    // add current value of determinant of standard deviation matrix to grid
    bool on_grid = true;
    unsigned int cv = 0;
    for (unsigned int cv_i = 0; cv_i < m_variables.size(); ++cv_i)
        {
        if (m_variables[cv_i].m_umbrella) continue;

        Scalar delta = (m_variables[cv_i].m_cv_max - m_variables[cv_i].m_cv_min)/
                       (m_variables[cv_i].m_num_points - 1);
        grid_coord[cv_i] = (current_val[cv] - m_variables[cv_i].m_cv_min)/delta;
        if (grid_coord[cv] >= m_variables[cv_i].m_num_points)
            on_grid = false;
        cv++;
        }

    // add Gaussian volume to grid
    if (on_grid)
        {
        unsigned int grid_idx = m_grid_index.getIndex(grid_coord);
        h_sigma_grid.data[grid_idx] += sigmaDeterminant();
        h_sigma_reweight_grid.data[grid_idx] += reweight*sigmaDeterminant();
        h_grid_hist.data[grid_idx]++;
        }

    if (m_prof) m_prof->pop();
    }


#ifdef ENABLE_CUDA
void IntegratorMetaDynamics::updateGridGPU(std::vector<Scalar>& current_val, Scalar scal, Scalar reweight)
    {
    if (m_prof)
        m_prof->push(m_exec_conf, "update grid");

        { 
        // copy current CV values into array
        ArrayHandle<Scalar> h_current_val(m_current_val, access_location::host, access_mode::overwrite);

        for (unsigned int cv = 0; cv < current_val.size(); cv++)
            h_current_val.data[cv] = current_val[cv];
        }

    ArrayHandle<Scalar> d_grid(m_grid, access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar> d_reweighted_grid(m_reweighted_grid, access_location::device, access_mode::readwrite);
    ArrayHandle<unsigned int> d_lengths(m_lengths, access_location::device, access_mode::read);
    ArrayHandle<Scalar> d_cv_min(m_cv_min, access_location::device, access_mode::read);
    ArrayHandle<Scalar> d_cv_max(m_cv_max, access_location::device, access_mode::read);
    ArrayHandle<Scalar> d_sigma(m_sigma, access_location::device, access_mode::read);
    ArrayHandle<Scalar> d_current_val(m_current_val, access_location::device, access_mode::read);

    ArrayHandle<Scalar> d_histogram(m_ftm_histogram, access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar> d_histogram_plus(m_ftm_histogram_plus, access_location::device, access_mode::readwrite);

    unsigned int idx = 0;
    if (m_mode == mode_flux_tempered)
        while (m_variables[idx].m_umbrella && idx < m_variables.size()) idx++;

    gpu_update_grid(m_grid_index.getNumElements(),
                    d_lengths.data,
                    m_num_biased_variables,
                    d_current_val.data,
                    d_grid.data,
                    d_reweighted_grid.data,
                    d_cv_min.data,
                    d_cv_max.data,
                    d_sigma.data,
                    scal,
                    reweight,
                    m_W,
                    (m_mode == mode_flux_tempered),
                    m_temp,
                    d_histogram.data,
                    d_histogram_plus.data,
                    m_num_histogram_entries,
                    m_variables[idx].m_ftm_min,
                    m_variables[idx].m_ftm_max);

    if (m_prof) m_prof->pop(m_exec_conf);
    }
#endif


void IntegratorMetaDynamics::setupHistograms()
    {
    assert(m_variables.size() == 1);

    unsigned int idx = 0;
    while (m_variables[idx].m_umbrella && idx < m_variables.size()) idx++;

    unsigned int num_points = m_variables[idx].m_num_points;

    GPUArray<Scalar> histogram(num_points, m_exec_conf);
    m_ftm_histogram.swap(histogram);

    GPUArray<Scalar> histogram_plus(num_points, m_exec_conf);
    m_ftm_histogram_plus.swap(histogram_plus);

    resetHistograms();
    }

void IntegratorMetaDynamics::resetHistograms()
    {
    assert(m_variables.size() == 1);

    if (m_compute_histograms)
        {
        unsigned int idx = 0;
        while (m_variables[idx].m_umbrella && idx < m_variables.size()) idx++;

        unsigned int num_points = m_variables[idx].m_num_points;
        ArrayHandle<Scalar> h_histogram(m_ftm_histogram, access_location::host, access_mode::overwrite);
        ArrayHandle<Scalar> h_histogram_plus(m_ftm_histogram_plus, access_location::host, access_mode::overwrite);

        memset(h_histogram.data, 0, num_points*sizeof(Scalar));
        memset(h_histogram_plus.data, 0, num_points*sizeof(Scalar));
        m_num_histogram_entries = 0;
        }

    ArrayHandle<unsigned int> h_grid_hist(m_grid_hist, access_location::host, access_mode::overwrite);
    memset(h_grid_hist.data, 0, sizeof(Scalar)*m_grid_hist.getNumElements());
    } 

void IntegratorMetaDynamics::sampleHistograms(Scalar val, bool state)
    {
    assert(m_variables.size()==1);

    unsigned int idx = 0;
    while (m_variables[idx].m_umbrella && idx < m_variables.size()) idx++;

    Scalar min = m_variables[idx].m_cv_min;
    Scalar max = m_variables[idx].m_cv_max;
    unsigned int num_points = m_variables[idx].m_num_points;
    Scalar delta = (max-min)/(Scalar)(num_points-1);
    Scalar sigma = m_variables[idx].m_sigma;

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

    unsigned int idx = 0;
    while (m_variables[idx].m_umbrella && idx < m_variables.size()) idx++;

    Scalar min = m_variables[idx].m_cv_min;
    Scalar max = m_variables[idx].m_cv_max;
    unsigned int num_points = m_variables[idx].m_num_points;
    Scalar delta = (max-min)/(Scalar)(num_points-1);
    Scalar sigma = m_variables[idx].m_sigma;

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
    assert(m_num_biased_variables ==1);

    unsigned int idx = 0;
    while (m_variables[idx].m_umbrella && idx < m_variables.size()) idx++;

    Scalar min = m_variables[idx].m_cv_min;
    Scalar max = m_variables[idx].m_cv_max;
    unsigned int num_points = m_variables[idx].m_num_points;

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
    assert(m_num_biased_variables==1);
    unsigned int idx = 0;
    while (m_variables[idx].m_umbrella && idx < m_variables.size()) idx++;

    Scalar delta = (m_variables[idx].m_cv_max - m_variables[idx].m_cv_min)/m_variables[idx].m_num_points;

    if (val - delta < m_variables[idx].m_cv_min) 
        {
        // forward difference
        Scalar val2 = val + delta;

        Scalar y2 = interpolateHistogram(val2,true);
        Scalar y1 = interpolateHistogram(val,true);
        return (y2-y1)/delta;
        }
    else if (val + delta > m_variables[idx].m_cv_max)
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

void IntegratorMetaDynamics::computeSigma()
    {
    std::vector<CollectiveVariableItem>::iterator iti,itj;

    unsigned int ncv = m_num_biased_variables;

    Scalar *sigma = new Scalar[ncv*ncv];

    for (iti = m_variables.begin(); iti != m_variables.end(); ++iti)
        {
        if (iti->m_umbrella) continue;

        ArrayHandle<Scalar4> handle_i = ArrayHandle<Scalar4>(iti->m_cv->getForceArray(), access_location::host, access_mode::read);
        unsigned int i = iti - m_variables.begin();

        for (itj = m_variables.begin(); itj != m_variables.end(); ++itj)
            {
            if (itj->m_umbrella) continue;

            unsigned int j = itj - m_variables.begin();

            // this releases an array twice, so may create problems in debug mode
            ArrayHandle<Scalar4> handle_j = (i != j) ?
            ArrayHandle<Scalar4>(itj->m_cv->getForceArray(), access_location::host, access_mode::read) : handle_i;

            Scalar sigmasq(0.0);
            // sum up products of derviatives
            for (unsigned int n = 0; n < m_pdata->getN(); ++n)
                {
                Scalar4 f_i = handle_i.data[n];
                Scalar4 f_j = handle_j.data[n];
                Scalar3 force_i = make_scalar3(f_i.x,f_i.y,f_i.z);
                Scalar3 force_j = make_scalar3(f_j.x,f_j.y,f_j.z);
                sigmasq += m_sigma_g*m_sigma_g*dot(force_i,force_j);
                }

            sigma[i*ncv+j] = sqrt(sigmasq);
            j++;
            } 
        i++;
        }

    #ifdef ENABLE_MPI
    if (m_pdata->getDomainDecomposition())
        {
        MPI_Allreduce(MPI_IN_PLACE,
                   &sigma[0],
                   ncv*ncv,
                   MPI_HOOMD_SCALAR,
                   MPI_SUM,
                   m_exec_conf->getMPICommunicator()); 
    }
    #endif

    bool is_root = m_exec_conf->getRank() == 0;

    if (is_root)
        {
        // write out result
        ArrayHandle<Scalar> h_sigma(m_sigma, access_location::host, access_mode::overwrite);
        for (unsigned int i = 0; i < ncv*ncv; ++i)
            h_sigma.data[i] = sigma[i];
        }

    delete[] sigma;
    }

int determinant_sign(const bnu::permutation_matrix<std ::size_t>& pm)
{
    int pm_sign=1;
    size_t size = pm.size();
    for (size_t i = 0; i < size; ++i)
        if (i != pm(i))
            pm_sign *= -1.0;
    return pm_sign;
}
 
Scalar IntegratorMetaDynamics::sigmaDeterminant()
    {
    ArrayHandle<Scalar> h_sigma(m_sigma, access_location::host, access_mode::read);

    unsigned int n_cv =m_num_biased_variables;

    bnu::matrix<Scalar> m(n_cv,n_cv);
    for (unsigned int i = 0; i < n_cv; ++i)
        for (unsigned int j = 0 ; j < n_cv; ++j)
            {
            m(i,j) = h_sigma.data[i*n_cv+j];
            }

    bnu::permutation_matrix<size_t> pm(m.size1());
    Scalar det(1.0);
    if( bnu::lu_factorize(m,pm) )
        {
        det = 0.0;
        }
    else
        {
        for(unsigned int i = 0; i < m.size1(); i++)
            det *= m(i,i); // multiply by elements on diagonal
        det = det * determinant_sign( pm );
        }

    return det;
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
    .def("setAdaptive", &IntegratorMetaDynamics::setAdaptive)
    .def("setSigmaG", &IntegratorMetaDynamics::setSigmaG)
    .def("resetHistograms", &IntegratorMetaDynamics::resetHistograms)
    ;

    enum_<IntegratorMetaDynamics::Enum>("mode")
    .value("standard", IntegratorMetaDynamics::mode_standard)
    .value("well_tempered", IntegratorMetaDynamics::mode_well_tempered)
    .value("flux_tempered", IntegratorMetaDynamics::mode_flux_tempered)
    ;
    }
