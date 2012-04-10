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
      m_curr_bias_potential(0.0),
      m_reweighting_factor(1.0),
      m_Vdot_avg_sum(0.0)
    {
    assert(m_T_shift>0);
    assert(m_W > 0);
    m_log_names.push_back("bias_potential");
    m_log_names.push_back("reweighting_factor");
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

#ifdef ENABLE_MPI
    if (m_comm)
        {
        // perform all necessary communication steps
        m_comm->communicate(timestep+1);
        }
#endif

    // compute all collective variables
    std::vector< CollectiveVariableItem >::iterator cv_item;
    for (cv_item = m_variables.begin(); cv_item != m_variables.end(); ++cv_item)
        cv_item->m_cv->compute(timestep);

    // update bias potential
    updateBiasPotential(timestep);

    // compute the net force on all particles
#ifdef ENABLE_CUDA
    if (exec_conf->exec_mode == ExecutionConfiguration::GPU)
        computeNetForceGPU(timestep+1);
    else
#endif
        computeNetForce(timestep+1);

    // add forces from collective variables
    addCVForces();

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

    if (m_prof)
        m_prof->push("Metadynamics");

    // collect values of collective variables
    std::vector< Scalar> current_val;
    std::vector<CollectiveVariableItem>::iterator it;
    for (it = m_variables.begin(); it != m_variables.end(); ++it)
        {
        unsigned int cv_index = it - m_variables.begin();
        Scalar val = it->m_cv->calcEnergySum();

        // append to history
        m_cv_values[cv_index].push_back(val);

        current_val.push_back(val);
        }


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
        cv_item->m_curr_bias = bias[cv_index];
        }

    // append a weight of unity 
    m_log_weight.push_back(0.0);

    if (m_num_update_steps % m_stride == 0)
        { 
        // every m_stride_steps
        double avg = 0.0; 
        // Resample Vdot values and calculate average in current ensemble
        std::vector<double> weight_exp(m_num_update_steps);
        double sum_of_weights = 0.0;
        for (unsigned int step = 0; step < m_num_update_steps; step++)
            {
            double gauss_exp  = 0.0;
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

            // calculate weighted average
            avg += m_W * exp(-gauss_exp + m_log_weight[step]);
            sum_of_weights += exp(m_log_weight[step]);
            weight_exp[step] = - m_W*gauss;
            }

        if (m_num_update_steps)
            avg /= m_num_update_steps;
            
//        std::cout << " avg == " << avg << " sum_of_weights == " << sum_of_weights << std::endl;

        // evolve weighting factors
        for (unsigned int step = 0; step < m_num_update_steps; step++)
            {
            weight_exp[step] += avg;
            m_log_weight[step] += weight_exp[step];
            }
        
        m_Vdot_avg_sum += avg;
        }

    // update reweighting factor
    m_reweighting_factor = exp(m_curr_bias_potential - m_Vdot_avg_sum);

    // increment number of updated steps
    m_num_update_steps++;

    if (m_prof)
        m_prof->pop();
    }

void IntegratorMetaDynamics::addCVForces()
    {
    if (m_prof)
        {
        m_prof->push("Integrate");
        m_prof->push("Collective variable forces");
        }
    
        {
        // access the net force and virial arrays
        const GPUArray<Scalar4>& net_force  = m_pdata->getNetForce();
        const GPUArray<Scalar>&  net_virial = m_pdata->getNetVirial();
        const GPUArray<Scalar4>& net_torque = m_pdata->getNetTorqueArray();
        ArrayHandle<Scalar4> h_net_force(net_force, access_location::host, access_mode::readwrite);
        ArrayHandle<Scalar> h_net_virial(net_virial, access_location::host, access_mode::readwrite);
        ArrayHandle<Scalar4> h_net_torque(net_torque, access_location::host, access_mode::readwrite);
        
        // now, add up the net forces
        unsigned int nparticles = m_pdata->getN();
//        unsigned int net_virial_pitch = net_virial.getPitch();
        assert(nparticles <= net_force.getNumElements());
        assert(6*nparticles <= net_virial.getNumElements());
        assert(nparticles <= net_torque.getNumElements());

        std::vector<CollectiveVariableItem>::iterator cv_item; 
        for (cv_item = m_variables.begin(); cv_item != m_variables.end(); ++cv_item)
            {
            boost::shared_ptr<ForceCompute> fc = cv_item->m_cv;
            GPUArray<Scalar4>& h_force_array = fc->getForceArray();
            GPUArray<Scalar>& h_virial_array = fc->getVirialArray();
            GPUArray<Scalar4>& h_torque_array = fc->getTorqueArray();

            ArrayHandle<Scalar4> h_force(h_force_array,access_location::host,access_mode::read);
            ArrayHandle<Scalar> h_virial(h_virial_array,access_location::host,access_mode::read);
            ArrayHandle<Scalar4> h_torque(h_torque_array,access_location::host,access_mode::read);

//            unsigned int virial_pitch = h_virial_array.getPitch();
            for (unsigned int j = 0; j < nparticles; j++)
                {
                Scalar bias = cv_item->m_curr_bias;
                h_net_force.data[j].x += bias*h_force.data[j].x;
                h_net_force.data[j].y += bias*h_force.data[j].y;
                h_net_force.data[j].z += bias*h_force.data[j].z;
/*
                h_net_force.data[j].w += h_force.data[j].w;
                
                h_net_torque.data[j].x += h_torque.data[j].x;
                h_net_torque.data[j].y += h_torque.data[j].y;
                h_net_torque.data[j].z += h_torque.data[j].z;
                h_net_torque.data[j].w += h_torque.data[j].w;

                for (unsigned int k = 0; k < 6; k++)
                    h_net_virial.data[k*net_virial_pitch+j] += h_virial.data[k*virial_pitch+j];
*/
                }

            }
        }
    
    if (m_prof)
        {
        m_prof->pop();
        m_prof->pop();
        }
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
