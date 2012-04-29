#include "IntegratorMetaDynamics.h"

#include <boost/python.hpp>
using namespace boost::python;

IntegratorMetaDynamics::IntegratorMetaDynamics(boost::shared_ptr<SystemDefinition> sysdef,
            Scalar deltaT,
            Scalar W,
            Scalar T_shift,
            unsigned int stride)
    : IntegratorTwoStep(sysdef, deltaT),
      m_W(W),
      m_T_shift(T_shift),
      m_stride(stride),
      m_num_update_steps(0),
      m_curr_bias_potential(0.0)
    {
    assert(m_T_shift>0);
    assert(m_W > 0);
    m_log_names.push_back("bias_potential");
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
    updateBiasPotential(timestep);

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
                          unsigned int>())
    .def("registerCollectiveVariable", &IntegratorMetaDynamics::registerCollectiveVariable)
    .def("removeAllVariables", &IntegratorMetaDynamics::removeAllVariables)
    ;
    }
