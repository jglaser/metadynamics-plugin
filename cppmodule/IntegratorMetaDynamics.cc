#include "IntegratorMetaDynamics.h"

#include <stdio.h>
#include <iomanip>

using namespace std;

#include <boost/python.hpp>
#include <boost/filesystem.hpp>

using namespace boost::python;
using namespace boost::filesystem;


IntegratorMetaDynamics::IntegratorMetaDynamics(boost::shared_ptr<SystemDefinition> sysdef,
            Scalar deltaT,
            Scalar W,
            Scalar T_shift,
            unsigned int stride,
            const std::string& filename,
            bool overwrite)
    : IntegratorTwoStep(sysdef, deltaT),
      m_W(W),
      m_T_shift(T_shift),
      m_stride(stride),
      m_num_update_steps(0),
      m_curr_bias_potential(0.0),
      m_file_initialized(false),
      m_filename(filename),
      m_overwrite(overwrite),
      m_is_appending(false),
      m_delimiter("\t")
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
    // initial update of the potential
    updateBiasPotential(timestep);

    Integrator::prepRun(timestep);
    }

void IntegratorMetaDynamics::update(unsigned int timestep)
    {
    // issue a warning if no integration methods are set
    if (!m_gave_warning && m_methods.size() == 0)
        {
        cout << "***Warning! No integration methods are set, continuing anyways." << endl;
        m_gave_warning = true;
        }
    
    // ensure that prepRun() has been called
    assert(m_prepared);
    
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

    if (! m_file_initialized && m_filename != "")
        {
        openOutputFile();
        if (! m_is_appending)
            writeFileHeader();
        m_file_initialized = true;
        }
    
    // collect values of collective variables
    std::vector< Scalar> current_val;
    std::vector<CollectiveVariableItem>::iterator it;
    for (it = m_variables.begin(); it != m_variables.end(); ++it)
        {
        unsigned int cv_index = it - m_variables.begin();
        Scalar val = it->m_cv->getCurrentValue(timestep);

        // append to history
        m_cv_values[cv_index].push_back(val);

        current_val.push_back(val);
        }

    if (m_prof)
        m_prof->push("Metadynamics");

    // update biasing weights by summing up partial derivivatives of Gaussians deposited every m_stride steps
    m_curr_bias_potential = 0.0;
    std::vector<double> bias(m_variables.size(), 0.0); 

    for (unsigned int step = 0; step < m_num_update_steps; step += m_stride)
        {
        double gauss_exp = 0.0;
        // calculate Gaussian contribution from t'=step
        std::vector<Scalar>::iterator val_it;
        for (val_it = current_val.begin(); val_it != current_val.end(); ++val_it)
            {
            Scalar val = *val_it;
            unsigned int cv_index = val_it - current_val.begin();
            Scalar sigma = m_variables[cv_index].m_sigma;
            double delta = val - m_cv_values[cv_index][step];
            gauss_exp += delta*delta/2.0/sigma/sigma;
            }
        double gauss = exp(-gauss_exp);

        // calculate partial derivatives
        std::vector<CollectiveVariableItem>::iterator cv_item;
        Scalar scal = exp(-m_bias_potential[step/m_stride]/m_T_shift);
        for (cv_item = m_variables.begin(); cv_item != m_variables.end(); ++cv_item)
            {
            unsigned int cv_index = cv_item - m_variables.begin();
            Scalar val = current_val[cv_index];
            Scalar sigma = m_variables[cv_index].m_sigma;
            bias[cv_index] -= m_W*scal/sigma/sigma*(val - m_cv_values[cv_index][step])*gauss;
            }

        m_curr_bias_potential += m_W*scal*gauss;
        }

    // write hills information
    if (m_file_initialized && (m_num_update_steps % m_stride == 0))
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
            if (cv != current_val.end()) m_file << m_delimiter;
            }

        m_file << endl;
        }
   
    if (m_num_update_steps % m_stride == 0)
        m_bias_potential.push_back(m_curr_bias_potential);

    // update current bias potential derivative for every collective variable
    std::vector<CollectiveVariableItem>::iterator cv_item;
    for (cv_item = m_variables.begin(); cv_item != m_variables.end(); ++cv_item)
        {
        unsigned int cv_index = cv_item - m_variables.begin();
        cv_item->m_cv->setBiasFactor(bias[cv_index]);
        }

    // increment number of updated steps
    m_num_update_steps++;

    if (m_prof)
        m_prof->pop();
    }

void export_IntegratorMetaDynamics()
    {
    class_<IntegratorMetaDynamics, boost::shared_ptr<IntegratorMetaDynamics>, bases<IntegratorTwoStep>, boost::noncopyable>
    ("IntegratorMetaDynamics", init< boost::shared_ptr<SystemDefinition>,
                          Scalar,
                          Scalar,
                          Scalar,
                          unsigned int,
                          const std::string&,
                          bool>())
    .def("registerCollectiveVariable", &IntegratorMetaDynamics::registerCollectiveVariable)
    .def("removeAllVariables", &IntegratorMetaDynamics::removeAllVariables)
    ;
    }
