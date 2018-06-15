/*! \file IntegratorMetaDynamics.cc
    \brief Implements the IntegratorMetaDynamics class
 */

#include "IntegratorMetaDynamics.h"

#include <stdio.h>
#include <iomanip>
#include <sstream>
#include <sys/stat.h>

namespace py = pybind11;

#include <hoomd/extern/Eigen/Eigen/Dense>

using namespace std;

#ifdef ENABLE_CUDA
#include "IntegratorMetaDynamics.cuh"
#endif

//! Constructor
IntegratorMetaDynamics::IntegratorMetaDynamics(std::shared_ptr<SystemDefinition> sysdef,
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
      m_num_gaussians(0),
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
      m_sigma_g(1.0),
      m_adaptive(false),
      m_temp(T),
      m_mode(mode),
      m_multiple_walkers(false),
      m_curr_reweight(1.0)
    {
    assert(m_T_shift>0);
    assert(m_W > 0);

    m_log_names.push_back("bias");
    m_log_names.push_back("det_sigma");
    m_log_names.push_back("weight");

    #ifdef ENABLE_MPI
    // create partition communicator
    MPI_Comm_split(MPI_COMM_WORLD,
        (m_exec_conf->getRank() == 0) ? 0 : MPI_UNDEFINED,
         m_exec_conf->getPartition(),
        &m_partition_comm);
    #endif
    }

void IntegratorMetaDynamics::openOutputFile()
    {
    struct stat buffer;
    bool file_exists  = stat(m_filename.c_str(), &buffer) == 0;

    if (file_exists && !m_overwrite)
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
            m_cv_values.resize(m_variables.size());
            std::vector< std::vector<Scalar> >::iterator it;

            for (it = m_cv_values.begin(); it != m_cv_values.end(); ++it)
                it->clear();

            // initialize GPU mirror values for collective variable data
            GPUArray<Scalar> cv_min(m_variables.size(), m_exec_conf);
            m_cv_min.swap(cv_min);

            GPUArray<Scalar> cv_max(m_variables.size(), m_exec_conf);
            m_cv_max.swap(cv_max);

            GPUArray<Scalar> current_val(m_variables.size(), m_exec_conf);
            m_current_val.swap(current_val);

            GPUArray<unsigned int> lengths(m_variables.size(), m_exec_conf);
            m_lengths.swap(lengths);

            GPUArray<Scalar> sigma_inv(m_variables.size()*m_variables.size(), m_exec_conf);
            m_sigma_inv.swap(sigma_inv);

            GPUArray<Scalar> M(m_variables.size()*m_variables.size(), m_exec_conf);
            m_M.swap(M);

            ArrayHandle<Scalar> h_cv_min(m_cv_min, access_location::host, access_mode::overwrite);
            ArrayHandle<Scalar> h_cv_max(m_cv_max, access_location::host, access_mode::overwrite);
            ArrayHandle<unsigned int> h_lengths(m_lengths, access_location::host, access_mode::overwrite);
            ArrayHandle<Scalar> h_sigma_inv(m_sigma_inv, access_location::host, access_mode::overwrite);
            ArrayHandle<Scalar> h_M(m_M, access_location::host, access_mode::overwrite);

            memset(h_sigma_inv.data, 0, sizeof(Scalar)*m_variables.size()*m_variables.size());
            memset(h_M.data, 0, sizeof(Scalar)*m_variables.size()*m_variables.size());

            for (unsigned int cv_idx = 0; cv_idx < m_variables.size(); cv_idx++)
                {
                h_cv_min.data[cv_idx] = m_variables[cv_idx].m_cv_min;
                h_cv_max.data[cv_idx] = m_variables[cv_idx].m_cv_max;
                h_sigma_inv.data[cv_idx*m_variables.size()+cv_idx] = Scalar(1.0)/m_variables[cv_idx].m_sigma;
                h_lengths.data[cv_idx] = m_variables[cv_idx].m_num_points;
                }

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
    std::vector< std::shared_ptr<IntegrationMethodTwoStep> >::iterator method;
    for (method = m_methods.begin(); method != m_methods.end(); ++method)
        (*method)->integrateStepOne(timestep);

    if (m_prof)
        m_prof->pop();


#ifdef ENABLE_MPI
    if (m_comm)
        {
        // perform all necessary communication steps. This ensures
        // a) that particles have migrated to the correct domains
        // b) that forces are calculated correctly, if ghost atom positions are updated every time step

        // also updates rigid bodies after ghost updating
        m_comm->communicate(timestep+1);
        }
    else
#endif
        {
        updateRigidBodies(timestep+1);
        }

    bool net_force_first = (m_variables.size() == 1 && m_variables[0].m_cv->requiresNetForce());

    if (! net_force_first)
        {
        // check sanity
        for (auto it = m_variables.begin(); it != m_variables.end(); ++it)
            {
            if (it->m_cv->requiresNetForce())
                {
                throw std::runtime_error("Only one collective variable requiring the potential energy may be defined.\n");
                }
            }
        }

    if (net_force_first)
        {
        // compute the net force on all particles
        #ifdef ENABLE_CUDA
        if (m_exec_conf->exec_mode == ExecutionConfiguration::GPU)
            computeNetForceGPU(timestep+1);
        else
        #endif
            computeNetForce(timestep+1);
        }

    // update bias potential
    updateBiasPotential(timestep+1);

    if (! net_force_first)
        {
        // compute the net force on all particles
        #ifdef ENABLE_CUDA
        if (m_exec_conf->exec_mode == ExecutionConfiguration::GPU)
            computeNetForceGPU(timestep+1);
        else
        #endif
            computeNetForce(timestep+1);
        }
    else
        {
        // compute bias forces *after* everything else
        m_variables[0].m_cv->compute(timestep);
        }

    if (m_prof)
        m_prof->push("Integrate");

    // perform the second step of the integration on all groups
    for (method = m_methods.begin(); method != m_methods.end(); ++method)
        (*method)->integrateStepTwo(timestep);

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

    if (timestep % m_stride == 0)
        {
        // compute derivatives of collective variables
        for (unsigned int i = 0; i < m_variables.size(); ++i)
            m_variables[i].m_cv->computeDerivatives(timestep);

        // compute instantaneous estimate of standard deviation matrix
        computeSigma();
        }

    if (m_prof)
        m_prof->push("Metadynamics");

#ifdef ENABLE_MPI
    if (m_pdata->getDomainDecomposition())
        is_root = m_exec_conf->isRoot();
#endif

    if (is_root)
        {
        if (! m_use_grid && (timestep % m_stride == 0))
            {
            // record history of CV values every m_stride steps
            std::vector<CollectiveVariableItem>::iterator it;
            for (unsigned int i = 0; i < m_variables.size(); ++i)
                {
                m_cv_values[i].push_back(current_val[i]);
                }
            }

        if (m_use_grid)
            {
            // update histogram
            updateHistogram(current_val);

            if (m_add_bias && (timestep % m_stride == 0))
                {
                // update sigma grid
                updateSigmaGrid(current_val);

                // scaling factor for well-tempered MetaD
                Scalar scal = Scalar(1.0);
                if (m_mode == mode_well_tempered)
                    {
                    Scalar V = interpolateGrid(current_val,false);
                    scal = exp(-V/m_T_shift);
                    }

                m_exec_conf->msg->notice(3) << "integrate.mode_metadynamics: Updating grid." << std::endl;

                #ifdef ENABLE_CUDA
                if (m_exec_conf->isCUDAEnabled())
                    updateGridGPU(current_val, scal);
                else
                    updateGrid(current_val, scal);
                #else
                updateGrid(current_val, scal);
                #endif

                #ifdef ENABLE_MPI
                if (m_multiple_walkers)
                    {
                    // sum up increments
                    ArrayHandle<Scalar> h_grid_delta(m_grid_delta, access_location::host, access_mode::readwrite);
                    ArrayHandle<Scalar> h_sigma_grid_delta(m_sigma_grid_delta, access_location::host, access_mode::readwrite);
                    ArrayHandle<Scalar> h_M_grid_delta(m_M_grid_delta, access_location::host, access_mode::readwrite);
                    ArrayHandle<unsigned int> h_grid_hist_delta(m_grid_hist_delta, access_location::host, access_mode::readwrite);
                    ArrayHandle<unsigned int> h_grid_hist_gauss_delta(m_grid_hist_gauss_delta, access_location::host, access_mode::readwrite);

                    MPI_Allreduce(MPI_IN_PLACE, h_grid_delta.data, m_grid_delta.getNumElements(),
                        MPI_HOOMD_SCALAR, MPI_SUM, m_partition_comm);
                    MPI_Allreduce(MPI_IN_PLACE, h_sigma_grid_delta.data, m_sigma_grid_delta.getNumElements(),
                        MPI_HOOMD_SCALAR, MPI_SUM, m_partition_comm);
                    MPI_Allreduce(MPI_IN_PLACE, h_M_grid_delta.data, m_M_grid_delta.getNumElements(),
                        MPI_HOOMD_SCALAR, MPI_SUM, m_partition_comm);
                    MPI_Allreduce(MPI_IN_PLACE, h_grid_hist_delta.data,m_grid_hist_delta.getNumElements(),
                        MPI_INT, MPI_SUM, m_partition_comm);
                    MPI_Allreduce(MPI_IN_PLACE, h_grid_hist_gauss_delta.data,m_grid_hist_gauss_delta.getNumElements(),
                        MPI_INT, MPI_SUM, m_partition_comm);
                    }
                #endif

                // use deltaV and grid histogram to update estimator of unbiased CV histogram
                updateReweightedEstimator(current_val);

                    {
                    // add deltas to grid
                    ArrayHandle<Scalar> h_grid(m_grid, access_location::host, access_mode::readwrite);
                    ArrayHandle<Scalar> h_grid_delta(m_grid_delta, access_location::host, access_mode::readwrite);
                    ArrayHandle<Scalar> h_sigma_grid(m_sigma_grid, access_location::host, access_mode::readwrite);
                    ArrayHandle<Scalar> h_sigma_grid_delta(m_sigma_grid_delta, access_location::host, access_mode::readwrite);
                    ArrayHandle<Scalar> h_M_grid(m_M_grid, access_location::host, access_mode::readwrite);
                    ArrayHandle<Scalar> h_M_grid_delta(m_M_grid_delta, access_location::host, access_mode::readwrite);
                    ArrayHandle<unsigned int> h_grid_hist(m_grid_hist, access_location::host, access_mode::readwrite);
                    ArrayHandle<unsigned int> h_grid_hist_delta(m_grid_hist_delta, access_location::host, access_mode::readwrite);
                    ArrayHandle<unsigned int> h_grid_hist_gauss(m_grid_hist_gauss, access_location::host, access_mode::readwrite);
                    ArrayHandle<unsigned int> h_grid_hist_gauss_delta(m_grid_hist_gauss_delta, access_location::host, access_mode::readwrite);

                    for (unsigned int i = 0; i < m_grid.getNumElements(); ++i)
                        {
                        h_grid.data[i] += h_grid_delta.data[i];
                        h_sigma_grid.data[i] += h_sigma_grid_delta.data[i];
                        h_grid_hist.data[i] += h_grid_hist_delta.data[i];
                        h_grid_hist_gauss.data[i] += h_grid_hist_gauss_delta.data[i];

                        h_grid_delta.data[i] = Scalar(0.0);
                        h_sigma_grid_delta.data[i] = Scalar(0.0);
                        h_grid_hist_delta.data[i] = 0;
                        h_grid_hist_gauss_delta.data[i] = 0;
                        }
                    for (unsigned int i = 0; i < m_M_grid.getNumElements(); ++i)
                        {
                        h_M_grid.data[i] += h_M_grid_delta.data[i];
                        h_M_grid_delta.data[i] = Scalar(0.0);
                        }
                    } // end ArrayHandle scope

                m_num_gaussians++;
                } // end update

            // calculate partial derivatives numerically
            for (unsigned int cv_idx = 0; cv_idx < m_variables.size(); ++cv_idx)
                bias[cv_idx] = biasPotentialDerivative(cv_idx, current_val);

            // current bias potential
            m_curr_bias_potential = interpolateGrid(current_val, false);

            // current reweighting factor
            m_curr_reweight = interpolateGrid(current_val, true);
            }
        else  //!m_use_grid
            {
            // update biasing weights by summing up partial derivivatives
            // of Gaussians deposited every m_stride steps
            m_curr_bias_potential = 0.0;

            if (m_adaptive)
                {
                m_exec_conf->msg->error() << "integrate.mode_metadynamics: Adaptive Gaussians only available in grid mode" << std::endl;
                throw std::runtime_error("Error in metadynamics integration.");
                }

            if (m_variables.size() != m_variables.size())
                {
                m_exec_conf->msg->error() << "integrate.mode_metadynamics: Reweighting supported only in grid mode." << std::endl;
                throw std::runtime_error("Error in metadynamics integration.");
                }



            ArrayHandle<Scalar> h_sigma_inv(m_sigma_inv, access_location::host, access_mode::read);

            // sum up all Gaussians accumulated until now
            for (unsigned int gauss_idx = 0; gauss_idx < m_bias_potential.size(); ++gauss_idx)
                {
                Scalar gauss_exp = 0.0;
                // calculate Gaussian contribution from t'=gauss_idx*m_stride
                for (unsigned int i = 0; i < m_variables.size(); ++i)
                    {
                    Scalar vali = current_val[i];
                    Scalar delta_i = vali - m_cv_values[i][gauss_idx];

                    for (unsigned int j = 0; j < m_variables.size(); ++j)
                        {
                        Scalar valj = current_val[j];
                        Scalar delta_j = valj - m_cv_values[j][gauss_idx];

                        Scalar sigma_inv_ij = h_sigma_inv.data[i*m_variables.size()+j];

                        gauss_exp += delta_i*delta_j*Scalar(1.0/2.0)*(sigma_inv_ij*sigma_inv_ij);
                        }
                    }
                Scalar gauss = exp(-gauss_exp);

                // calculate partial derivatives

                // scaling factor for well-tempered MetaD
                Scalar scal = Scalar(1.0);
                if (m_mode == mode_well_tempered)
                    scal = exp(-m_bias_potential[gauss_idx]/m_T_shift);

                for (unsigned int i = 0; i < m_variables.size(); ++i)
                    {
                    Scalar val_i = current_val[i];

                    for (unsigned int j = 0; j < m_variables.size(); ++j)
                        {
                        Scalar val_j = current_val[j];

                        Scalar sigma_inv_ij = h_sigma_inv.data[i*m_variables.size()+j];

                        bias[i] -= Scalar(1.0/2.0)*m_W*scal*(sigma_inv_ij*sigma_inv_ij)*(val_j - m_cv_values[j][gauss_idx])*gauss;
                        bias[j] -= Scalar(1.0/2.0)*m_W*scal*(sigma_inv_ij*sigma_inv_ij)*(val_i - m_cv_values[i][gauss_idx])*gauss;
                        }
                    }

                m_curr_bias_potential += m_W*scal*gauss;
                }
            }

        // write hills information
        if (m_is_initialized && (timestep % m_stride == 0) && m_add_bias && m_file.is_open())
            {
            ArrayHandle<Scalar> h_sigma_inv(m_sigma_inv, access_location::host, access_mode::read);

            Scalar W = m_W*exp(-m_curr_bias_potential/m_T_shift);
            m_file << setprecision(10) << timestep << m_delimiter;
            m_file << setprecision(10) << W << m_delimiter;

            std::vector<Scalar>::iterator cv,cvj;
            for (cv = current_val.begin(); cv != current_val.end(); ++cv)
                {
                unsigned int cv_index = cv - current_val.begin();
                m_file << setprecision(10) << *cv << m_delimiter;

                // Write row of inverse sigma matrix
                for (cvj = current_val.begin(); cvj != current_val.end(); ++cvj)
                    {
                    unsigned int cv_index_j = cvj - current_val.begin();
                    Scalar sigma_inv_ij = h_sigma_inv.data[cv_index*m_variables.size()+cv_index_j];
                    m_file << setprecision(10) << sigma_inv_ij;
                    }

                if (cv != current_val.end() -1) m_file << m_delimiter;
                }

            m_file << endl;
            }

        if (m_add_bias && (! m_use_grid) && (timestep % m_stride == 0))
            m_bias_potential.push_back(m_curr_bias_potential);

        // dump grid information if required using alternating scheme
        if (m_grid_period && (timestep % m_grid_period == 0))
            {
            if (m_grid_fname2 != "")
                {
                writeGrid(m_cur_file ? m_grid_fname2 : m_grid_fname1, timestep);
                m_cur_file = m_cur_file ? 0 : 1;
                }
            else
                writeGrid(m_grid_fname1, timestep);
            }

    
        } // endif root processor

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
        cv_item->m_cv->setBiasFactor(bias[cv]);
        cv++;
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

    unsigned int idx = 0;
    for (it = m_variables.begin(); it != m_variables.end(); ++it)
        {
        lengths[idx] = it->m_num_points;
        idx++;
        }

    m_grid_index.setLengths(lengths);

    GPUArray<Scalar> grid(m_grid_index.getNumElements(),m_exec_conf);
    m_grid.swap(grid);

    GPUArray<Scalar> grid_delta(m_grid_index.getNumElements(),m_exec_conf);
    m_grid_delta.swap(grid_delta);

    GPUArray<Scalar> grid_reweighted(m_grid_index.getNumElements(),m_exec_conf);
    m_grid_reweighted.swap(grid_reweighted);

    GPUArray<Scalar> grid_weight(m_grid_index.getNumElements(),m_exec_conf);
    m_grid_weight.swap(grid_weight);

    // reset grid
    ArrayHandle<Scalar> h_grid(m_grid, access_location::host, access_mode::overwrite);
    memset(h_grid.data, 0, sizeof(Scalar)*m_grid.getNumElements());

    ArrayHandle<Scalar> h_grid_delta(m_grid_delta, access_location::host, access_mode::overwrite);
    memset(h_grid_delta.data, 0, sizeof(Scalar)*m_grid_delta.getNumElements());

    GPUArray<Scalar> sigma_grid(m_grid_index.getNumElements(),m_exec_conf);
    m_sigma_grid.swap(sigma_grid);

    GPUArray<Scalar> sigma_grid_delta(m_grid_index.getNumElements(),m_exec_conf);
    m_sigma_grid_delta.swap(sigma_grid_delta);

    GPUArray<Scalar> M_grid(m_grid_index.getNumElements()*m_variables.size()*m_variables.size(),m_exec_conf);
    m_M_grid.swap(M_grid);

    GPUArray<Scalar> M_grid_delta(m_grid_index.getNumElements()*m_variables.size()*m_variables.size(),m_exec_conf);
    m_M_grid_delta.swap(M_grid_delta);

        {
        ArrayHandle<Scalar> h_M_grid(m_M_grid, access_location::host, access_mode::overwrite);
        ArrayHandle<Scalar> h_M_grid_delta(m_M_grid_delta, access_location::host, access_mode::overwrite);

        for (unsigned int i = 0; i < m_M_grid.getNumElements(); ++i)
            {
            h_M_grid.data[i] = 0.0;
            h_M_grid_delta.data[i] = 0.0;
            }
        }

    GPUArray<unsigned int> grid_hist(m_grid_index.getNumElements(),m_exec_conf);
    m_grid_hist.swap(grid_hist);

    GPUArray<unsigned int> grid_hist_delta(m_grid_index.getNumElements(),m_exec_conf);
    m_grid_hist_delta.swap(grid_hist_delta);

    GPUArray<unsigned int> grid_hist_gauss(m_grid_index.getNumElements(),m_exec_conf);
    m_grid_hist_gauss.swap(grid_hist_gauss);

    GPUArray<unsigned int> grid_hist_gauss_delta(m_grid_index.getNumElements(),m_exec_conf);
    m_grid_hist_gauss_delta.swap(grid_hist_gauss_delta);

    ArrayHandle<Scalar> h_grid_reweighted(m_grid_reweighted, access_location::host, access_mode::overwrite);
    memset(h_grid_reweighted.data, 0, sizeof(Scalar)*m_grid.getNumElements());

    ArrayHandle<unsigned int> h_grid_hist_gauss(m_grid_hist_gauss, access_location::host, access_mode::overwrite);
    memset(h_grid_hist_gauss.data,0, sizeof(unsigned int)*m_grid_hist_gauss.getNumElements());

    ArrayHandle<unsigned int> h_grid_hist_gauss_delta(m_grid_hist_gauss_delta, access_location::host, access_mode::overwrite);
    memset(h_grid_hist_gauss_delta.data,0, sizeof(unsigned int)*m_grid_hist_gauss_delta.getNumElements());

    // reset to one
    ArrayHandle<Scalar> h_grid_weight(m_grid_weight, access_location::host, access_mode::overwrite);

    for (unsigned int i = 0; i < m_grid_weight.getNumElements(); ++i)
        h_grid_weight.data[i] = Scalar(1.0);

    resetHistogram();
    }

Scalar IntegratorMetaDynamics::interpolateGrid(const std::vector<Scalar>& val, bool reweight)
    {
    assert(val.size() == m_grid_index.getDimension());

    // find closest d-dimensional sub-block
    std::vector<unsigned int> lower_idx(m_grid_index.getDimension());
    std::vector<unsigned int> upper_idx(m_grid_index.getDimension());
    std::vector<Scalar> rel_delta(m_grid_index.getDimension());

    unsigned int cv = 0;
    for (unsigned int cv_idx = 0; cv_idx < m_variables.size(); cv_idx++)
        {
        Scalar delta = (m_variables[cv_idx].m_cv_max - m_variables[cv_idx].m_cv_min)/(m_variables[cv_idx].m_num_points - 1);
        int lower = (int) ((val[cv] - m_variables[cv_idx].m_cv_min)/delta);
        int upper = lower+1;

        if (lower < 0 || upper >= m_variables[cv_idx].m_num_points)
            {
            m_exec_conf->msg->warning() << "integrate.mode_metadynamics: Value " << val[cv]
                                        << " of collective variable " << m_variables[cv_idx].m_cv->getName() << " out of bounds." << endl
                                        << "Assuming bias potential of zero." << endl;
            return Scalar(0.0);
            }

        Scalar lower_bound = m_variables[cv_idx].m_cv_min + delta * lower;
        Scalar upper_bound = m_variables[cv_idx].m_cv_min + delta * upper;
        lower_idx[cv] = lower;
        upper_idx[cv] = upper;
        rel_delta[cv] = (val[cv]-lower_bound)/(upper_bound-lower_bound);

        cv++;
        }

    // construct multilinear interpolation
    unsigned int n_term = 1 << m_grid_index.getDimension();
    Scalar res(0.0);

    ArrayHandle<Scalar> h_grid(m_grid, access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_grid_weight(m_grid_weight, access_location::host, access_mode::read);

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
      
        unsigned int idx = m_grid_index.getIndex(coords);
        Scalar val = (reweight ? h_grid_weight.data[idx] : h_grid.data[idx]);
        term *= val;
        res += term;
        }

    return res;
    }

Scalar IntegratorMetaDynamics::biasPotentialDerivative(unsigned int cv, const std::vector<Scalar>& val)
    {
    ArrayHandle<Scalar> h_cv_min(m_cv_min, access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_cv_max(m_cv_max, access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_lengths(m_lengths, access_location::host, access_mode::read);

    Scalar delta = (h_cv_max.data[cv] - h_cv_min.data[cv])/
                   (Scalar)(h_lengths.data[cv] - 1);
    if (val[cv] - delta < m_variables[cv].m_cv_min) 
        {
        // forward difference
        std::vector<Scalar> val2 = val;
        val2[cv] += delta;

        Scalar y2 = interpolateGrid(val2,false);
        Scalar y1 = interpolateGrid(val,false);
        return (y2-y1)/delta;
        }
    else if (val[cv] + delta > m_variables[cv].m_cv_max)
        {
        // backward difference
        std::vector<Scalar> val2 = val;
        val2[cv] -= delta;
        Scalar y1 = interpolateGrid(val2,false);
        Scalar y2 = interpolateGrid(val,false);
        return (y2-y1)/delta;
        }
    else
        {
        // central difference
        std::vector<Scalar> val2 = val;
        std::vector<Scalar> val1 = val;
        val1[cv] -= delta;
        val2[cv] += delta;
        Scalar y1 = interpolateGrid(val1,false);
        Scalar y2 = interpolateGrid(val2,false);
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

void IntegratorMetaDynamics::dumpGrid(const std::string& filename1, const std::string& filename2, unsigned int period)
    {
    if (period == 0)
        {
        // dump grid immediately
        writeGrid(filename1, 0);
        return;
        }

    m_grid_period = period;
    m_grid_fname1 = filename1;
    m_grid_fname2 = filename2;
    }

void IntegratorMetaDynamics::writeGrid(const std::string& filename, unsigned int timestep)
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
    file.open((filename+"_"+std::to_string(timestep)).c_str(), ios_base::out);

    // write file header
    file << "#n_cv: " << m_grid_index.getDimension() << std::endl;
    file << "#dim: ";

    for (unsigned int i= 0; i < m_grid_index.getDimension(); i++)
        file << " " << m_grid_index.getLength(i);

    file << std::endl;

    file << "#num_gaussians: " << m_num_gaussians << std::endl;

    for (unsigned int i = 0; i < m_variables.size(); i++)
        {
        file << m_variables[i].m_cv->getName() << m_delimiter;
        }


    file << "grid_value";

    file << m_delimiter << "det_sigma";
    file << m_delimiter << "num_gaussians";
    file << m_delimiter << "hist";
    file << m_delimiter << "hist_reweight";
    file << m_delimiter << "weight";

    // header for derivative tensor
    for (unsigned int i = 0; i < m_variables.size(); ++i)
        for (unsigned int j = 0; j < m_variables.size(); ++j)
            file << m_delimiter << "M_" << i << j << std::endl;

    file << std::endl;

    // loop over grid
    ArrayHandle<Scalar> h_grid(m_grid, access_location::host, access_mode::read);
    unsigned int len = m_grid_index.getNumElements();
    std::vector<unsigned int> coords(m_grid_index.getDimension());

    ArrayHandle<Scalar> h_sigma_grid(m_sigma_grid, access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_M_grid(m_M_grid, access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_grid_hist(m_grid_hist, access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_grid_hist_gauss(m_grid_hist_gauss, access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_grid_reweighted(m_grid_reweighted, access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_grid_weight(m_grid_weight, access_location::host, access_mode::read);

    unsigned int ncv = m_variables.size();

    for (unsigned int grid_idx = 0; grid_idx < len; grid_idx++)
        {
        // obtain d-dimensional coordinates
        m_grid_index.getCoordinates(grid_idx, coords);

        unsigned int cv = 0;
        for (unsigned int cv_idx = 0; cv_idx < m_variables.size(); ++cv_idx)
            {
            Scalar delta = (m_variables[cv_idx].m_cv_max - m_variables[cv_idx].m_cv_min)/
                           (m_variables[cv_idx].m_num_points - 1);
            Scalar val = m_variables[cv_idx].m_cv_min + coords[cv]*delta;

            file << setprecision(10) << val << m_delimiter;
            cv++;
            }

        file << setprecision(10) << h_grid.data[grid_idx];

        // write average of Gaussian volume
        Scalar val;
        if (h_grid_hist_gauss.data[grid_idx] > 0)
            {
            val = h_sigma_grid.data[grid_idx]/(Scalar)h_grid_hist_gauss.data[grid_idx];
            }
        else
            val = Scalar(0.0);

        file << m_delimiter << setprecision(10) << val;
        file << m_delimiter << h_grid_hist_gauss.data[grid_idx];
        file << m_delimiter << h_grid_hist.data[grid_idx];

        file << m_delimiter << setprecision(10) << h_grid_reweighted.data[grid_idx];
        file << m_delimiter << setprecision(10) << h_grid_weight.data[grid_idx];

        for (unsigned int i = 0; i < ncv; ++i)
            for (unsigned int j = 0; j < ncv; ++j)
                {
                // write average of deriviative tensor
                Scalar val;
                if (h_grid_hist_gauss.data[grid_idx] > 0)
                    {
                    val = h_M_grid.data[(i*ncv+j)*len+grid_idx]/(Scalar)h_grid_hist_gauss.data[grid_idx];
                    }
                else
                    val = Scalar(0.0);

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
    iss >> tmp >> m_num_gaussians;

    // Skip last header line
    getline(file, line);

    unsigned int len = m_grid_index.getNumElements();
    std::vector<unsigned int> coords(m_grid_index.getDimension()); 
    ArrayHandle<Scalar> h_grid(m_grid, access_location::host, access_mode::overwrite);

    ArrayHandle<Scalar> h_sigma_grid(m_sigma_grid, access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar> h_M_grid(m_M_grid, access_location::host, access_mode::overwrite);
    ArrayHandle<unsigned int> h_grid_hist(m_grid_hist, access_location::host, access_mode::overwrite);
    ArrayHandle<unsigned int> h_grid_hist_gauss(m_grid_hist_gauss, access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar> h_grid_reweighted(m_grid_reweighted, access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar> h_grid_weight(m_grid_weight, access_location::host, access_mode::overwrite);

    unsigned int ncv = m_variables.size();

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

        iss >> h_sigma_grid.data[grid_idx];
        iss >> h_grid_hist_gauss.data[grid_idx];
        iss >> h_grid_hist.data[grid_idx];

        h_sigma_grid.data[grid_idx] *= h_grid_hist_gauss.data[grid_idx];

        iss >> h_grid_reweighted.data[grid_idx];
        iss >> h_grid_weight.data[grid_idx];

        for (unsigned int i = 0; i < ncv; ++i)
            for (unsigned int j = 0; j < ncv; ++j)
                {
                iss >> h_M_grid.data[(i*ncv+j)*len+grid_idx];
                h_M_grid.data[(i*ncv+j)*len+grid_idx] *= h_grid_hist_gauss.data[grid_idx];
                }
        }

    file.close();

    }

void IntegratorMetaDynamics::updateGrid(std::vector<Scalar>& current_val, Scalar scal )
    {
    if (m_prof) m_prof->push("update grid");

    ArrayHandle<Scalar> h_grid_delta(m_grid_delta, access_location::host, access_mode::overwrite);

    // loop over grid
    unsigned int len = m_grid_index.getNumElements();
    std::vector<unsigned int> coords(m_grid_index.getDimension());

    ArrayHandle<Scalar> h_sigma_inv(m_sigma_inv, access_location::host, access_mode::read);

    for (unsigned int grid_idx = 0; grid_idx < len; grid_idx++)
        {
        // obtain d-dimensional coordinates
        m_grid_index.getCoordinates(grid_idx, coords);

        Scalar gauss_exp(0.0);
        // evaluate Gaussian on grid point
        for (unsigned int cv_i = 0; cv_i < m_variables.size(); ++cv_i)
            {
            Scalar delta_i = (m_variables[cv_i].m_cv_max - m_variables[cv_i].m_cv_min)/
                           (m_variables[cv_i].m_num_points - 1);
            Scalar val_i = m_variables[cv_i].m_cv_min + coords[cv_i]*delta_i;
            double d_i = val_i - current_val[cv_i];

            for (unsigned int cv_j = 0; cv_j < m_variables.size(); ++cv_j)
                {
                Scalar delta_j = (m_variables[cv_j].m_cv_max - m_variables[cv_j].m_cv_min)/
                               (m_variables[cv_j].m_num_points - 1);
                Scalar val_j = m_variables[cv_j].m_cv_min + coords[cv_j]*delta_j;
                double d_j = val_j - current_val[cv_j];

                Scalar sigma_inv_ij = h_sigma_inv.data[cv_i*m_variables.size()+cv_j];

                gauss_exp += d_i*d_j*Scalar(1.0/2.0)*(sigma_inv_ij*sigma_inv_ij);
                }
            }
        double gauss = exp(-gauss_exp);

        // add Gaussian to grid
        h_grid_delta.data[grid_idx] = m_W*scal*gauss;
        }

    if (m_prof) m_prof->pop();
    }

/*! \param val List of current CV values 
 *
 * Called every time a Gaussian is deposted
 */
void IntegratorMetaDynamics::updateReweightedEstimator(std::vector<Scalar>& current_val)
    {
    if (m_prof) m_prof->push("update grid");

    ArrayHandle<Scalar> h_grid_reweighted(m_grid_reweighted, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> h_grid_weight(m_grid_weight, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> h_grid_delta(m_grid_delta, access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_grid_hist_delta(m_grid_hist_delta, access_location::host, access_mode::read);

    // loop over grid
    unsigned int len = m_grid_index.getNumElements();
    std::vector<unsigned int> coords(m_grid_index.getDimension()); 

    Scalar avg_delta_V(0.0);
    Scalar norm(0.0);

    // compute ensemble-averaged temporal bias potential derivative
    for (unsigned int grid_idx = 0; grid_idx < len; grid_idx++)
        {
        h_grid_reweighted.data[grid_idx] += (Scalar) h_grid_hist_delta.data[grid_idx];
        avg_delta_V += h_grid_reweighted.data[grid_idx]*h_grid_delta.data[grid_idx];
        norm += h_grid_reweighted.data[grid_idx];
        }

    avg_delta_V /= norm; 

    for (unsigned int grid_idx = 0; grid_idx < len; grid_idx++)
        {
        double delta_V = h_grid_delta.data[grid_idx];

        // evolve estimator and grid of reweighting factors
        Scalar fac = exp(-(delta_V-avg_delta_V)/m_temp);
        h_grid_reweighted.data[grid_idx] *= fac;
        h_grid_weight.data[grid_idx] /= fac;
        }

    if (m_prof) m_prof->pop();
    }

void IntegratorMetaDynamics::updateHistogram(std::vector<Scalar>& current_val)
    {
    if (m_prof) m_prof->push("update grid");

    ArrayHandle<unsigned int> h_grid_hist_delta(m_grid_hist_delta, access_location::host, access_mode::readwrite);

    std::vector<unsigned int> grid_coord(m_variables.size());

    // increment histogram of CV values
    bool on_grid = true;
    for (unsigned int cv_i = 0; cv_i < m_variables.size(); ++cv_i)
        {
        Scalar delta = (m_variables[cv_i].m_cv_max - m_variables[cv_i].m_cv_min)/
                       (m_variables[cv_i].m_num_points - 1);
        grid_coord[cv_i] = (current_val[cv_i] - m_variables[cv_i].m_cv_min)/delta;
        if (grid_coord[cv_i] >= m_variables[cv_i].m_num_points)
            on_grid = false;
        }

    // add to histogram
    if (on_grid)
        {
        unsigned int grid_idx = m_grid_index.getIndex(grid_coord);
        h_grid_hist_delta.data[grid_idx]++;
        }

    if (m_prof) m_prof->pop();
    }


void IntegratorMetaDynamics::updateSigmaGrid(std::vector<Scalar>& current_val)
    {
    if (m_prof) m_prof->push("update grid");

    ArrayHandle<Scalar> h_sigma_grid_delta(m_sigma_grid_delta, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> h_M(m_M, access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_M_grid_delta(m_M_grid_delta, access_location::host, access_mode::readwrite);
    ArrayHandle<unsigned int> h_grid_hist_gauss_delta(m_grid_hist_gauss_delta, access_location::host, access_mode::readwrite);

    assert(h_sigma_grid_delta.data);
    assert(h_M_grid_delta.data);

    std::vector<unsigned int> grid_coord(m_variables.size());

    // add current value of determinant of standard deviation matrix to grid
    bool on_grid = true;
    unsigned int cv = 0;
    for (unsigned int cv_i = 0; cv_i < m_variables.size(); ++cv_i)
        {
        Scalar delta = (m_variables[cv_i].m_cv_max - m_variables[cv_i].m_cv_min)/
                       (m_variables[cv_i].m_num_points - 1);
        grid_coord[cv] = (current_val[cv] - m_variables[cv_i].m_cv_min)/delta;
        if (grid_coord[cv] >= m_variables[cv_i].m_num_points)
            on_grid = false;
        cv++;
        }

    // add Gaussian to grid
    if (on_grid)
        {
        unsigned int grid_idx = m_grid_index.getIndex(grid_coord);
        h_sigma_grid_delta.data[grid_idx] += sigmaDeterminant();
        h_grid_hist_gauss_delta.data[grid_idx]++;

        unsigned int ncv = m_variables.size();
        for (unsigned int i = 0; i < ncv; ++i)
            for (unsigned int j = 0; j < ncv; ++j)
                h_M_grid_delta.data[(i*ncv+j)*m_grid.getNumElements()+grid_idx] += h_M.data[i*m_variables.size()+j];
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

        for (unsigned int cv = 0; cv < current_val.size(); cv++)
            h_current_val.data[cv] = current_val[cv];
        }

    ArrayHandle<Scalar> d_grid_delta(m_grid_delta, access_location::device, access_mode::readwrite);
    ArrayHandle<unsigned int> d_lengths(m_lengths, access_location::device, access_mode::read);
    ArrayHandle<Scalar> d_cv_min(m_cv_min, access_location::device, access_mode::read);
    ArrayHandle<Scalar> d_cv_max(m_cv_max, access_location::device, access_mode::read);
    ArrayHandle<Scalar> d_sigma_inv(m_sigma_inv, access_location::device, access_mode::read);
    ArrayHandle<Scalar> d_current_val(m_current_val, access_location::device, access_mode::read);

    gpu_update_grid(m_grid_index.getNumElements(),
                    d_lengths.data,
                    m_variables.size(),
                    d_current_val.data,
                    d_grid_delta.data,
                    d_cv_min.data,
                    d_cv_max.data,
                    d_sigma_inv.data,
                    scal,
                    m_W,
                    m_temp);

    if (m_prof) m_prof->pop(m_exec_conf);
    }
#endif

void IntegratorMetaDynamics::resetHistogram()
    {
    ArrayHandle<unsigned int> h_grid_hist(m_grid_hist, access_location::host, access_mode::overwrite);
    memset(h_grid_hist.data, 0, sizeof(unsigned int)*m_grid_hist.getNumElements());

    ArrayHandle<unsigned int> h_grid_hist_delta(m_grid_hist_delta, access_location::host, access_mode::overwrite);
    memset(h_grid_hist_delta.data, 0, sizeof(unsigned int)*m_grid_hist_delta.getNumElements());

    }

void IntegratorMetaDynamics::computeSigma()
    {
    if (m_prof)
        m_prof->push(m_exec_conf,"Derivatives");

    std::vector<CollectiveVariableItem>::iterator iti,itj;

    unsigned int ncv = m_variables.size();

    Scalar *sigmasq = new Scalar[ncv*ncv];

    bool is_root = m_exec_conf->getRank() == 0;

    unsigned int i = 0;
    unsigned int j = 0;

    std::vector< ArrayHandle<Scalar4>* > handles;

    for (iti = m_variables.begin(); iti != m_variables.end(); ++iti)
        handles.push_back(new ArrayHandle<Scalar4>(iti->m_cv->getForceArray(), access_location::host, access_mode::read));

    ArrayHandle<Scalar> h_M(m_M, access_location::host, access_mode::overwrite);

    for (iti = m_variables.begin(); iti != m_variables.end(); ++iti)
        {
        ArrayHandle<Scalar4>& handle_i = *handles[i];

        j = 0;
        for (itj = m_variables.begin(); itj != m_variables.end(); ++itj)
            {
            sigmasq[i*ncv+j] = Scalar(0.0);
            h_M.data[i*ncv+j] = Scalar(0.0);
            if (iti->m_cv->canComputeDerivatives() && itj->m_cv->canComputeDerivatives())
                {
                // this releases an array twice, so may create problems in debug mode
                ArrayHandle<Scalar4>& handle_j = *handles[j];

                // sum up products of derviatives
                for (unsigned int n = 0; n < m_pdata->getN(); ++n)
                    {
                    Scalar4 f_i = handle_i.data[n];
                    Scalar4 f_j = handle_j.data[n];
                    Scalar3 force_i = make_scalar3(f_i.x,f_i.y,f_i.z);
                    Scalar3 force_j = make_scalar3(f_j.x,f_j.y,f_j.z);
                    sigmasq[i*ncv+j] += m_sigma_g*m_sigma_g*dot(force_i,force_j);
                    h_M.data[i*ncv+j] += dot(force_i,force_j);
                    }
                }
            else if (i==j && is_root) sigmasq[i*ncv+j] = iti->m_sigma*iti->m_sigma;

            j++;
            }
        i++;
        }

    for (unsigned int i = 0; i < handles.size(); ++i)
        delete handles[i];

    #ifdef ENABLE_MPI
    if (m_pdata->getDomainDecomposition())
        {
        MPI_Allreduce(MPI_IN_PLACE,
                   &sigmasq[0],
                   ncv*ncv,
                   MPI_HOOMD_SCALAR,
                   MPI_SUM,
                   m_exec_conf->getMPICommunicator());
        MPI_Allreduce(MPI_IN_PLACE,
                   &h_M.data[0],
                   ncv*ncv,
                   MPI_HOOMD_SCALAR,
                   MPI_SUM,
                   m_exec_conf->getMPICommunicator());

        }
    #endif

    if (is_root)
        {
        // invert sigma matrix
        ArrayHandle<Scalar> h_sigma_inv(m_sigma_inv, access_location::host, access_mode::overwrite);

        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> m(ncv,ncv);

        for (unsigned int i = 0; i < ncv; ++i)
            for (unsigned int j = 0 ; j < ncv; ++j)
                m(i,j) = sqrt(sigmasq[i*ncv+j]);

        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> inv = m.inverse();

        for (unsigned int i = 0; i < ncv; ++i)
            for (unsigned int j = 0 ; j < ncv; ++j)
                h_sigma_inv.data[i*ncv+j] = inv(i,j);

        }

    delete[] sigmasq;

    if (m_prof)
        m_prof->pop();
    }

Scalar IntegratorMetaDynamics::sigmaDeterminant()
    {
    #ifdef ENABLE_MPI
    if (m_pdata->getDomainDecomposition() && m_exec_conf->getRank())
        return Scalar(0.0);
    #endif

    ArrayHandle<Scalar> h_sigma_inv(m_sigma_inv, access_location::host, access_mode::overwrite);

    unsigned int ncv = m_variables.size();
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> m(ncv,ncv);

    for (unsigned int i = 0; i < ncv; ++i)
        for (unsigned int j = 0 ; j < ncv; ++j)
            m(i,j) = h_sigma_inv.data[i*ncv+j];

    return m.determinant();
    }

void export_IntegratorMetaDynamics(py::module& m)
    {
    py::class_<IntegratorMetaDynamics, std::shared_ptr<IntegratorMetaDynamics> > integrator_metad(m, "IntegratorMetaDynamics", py::base<IntegratorTwoStep>());

    integrator_metad.def(py::init< std::shared_ptr<SystemDefinition>,
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
        .def("setMode", &IntegratorMetaDynamics::setMode)
        .def("setStride", &IntegratorMetaDynamics::setStride)
        .def("setAdaptive", &IntegratorMetaDynamics::setAdaptive)
        .def("setSigmaG", &IntegratorMetaDynamics::setSigmaG)
        .def("resetHistogram", &IntegratorMetaDynamics::resetHistogram)
        .def("setMultipleWalkers", &IntegratorMetaDynamics::setMultipleWalkers)
        ;

    py::enum_<IntegratorMetaDynamics::Enum>(integrator_metad,"mode")
        .value("standard", IntegratorMetaDynamics::mode_standard)
        .value("well_tempered", IntegratorMetaDynamics::mode_well_tempered)
        .export_values();
    ;
    }
