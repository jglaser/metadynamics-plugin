#include "PathCollectiveVariable.h"

PathCollectiveVariable::PathCollectiveVariable(boost::shared_ptr<SystemDefinition> sysdef,
                                               Enum dir,
                                               const unsigned int num_frames,
                                               const Scalar scale,
                                               const std::string name)
    : CollectiveVariable(sysdef, name + ((dir == parallel) ? "_par" : "_trans")),
      m_direction(dir),
      m_num_frames(num_frames),
      m_scale(scale),
      m_cv_last_updated(0),
      m_cv(0.0),
      m_denom(0.0),
      m_numerator(0.0)
    {
    if (m_num_frames < 2)
        {
        m_exec_conf->msg->error() << "metadynamics.cv.path: Minimum number of frames is two."
                                  << std::endl << std::endl;
        throw std::runtime_error("Error setting up collective variable.");
        }
    
    m_log_name = "cv_path_" + getName();
    }

Scalar PathCollectiveVariable::getCurrentValue(unsigned int timestep)
    {
    if (m_cv_last_updated == timestep && !m_cv_last_updated) return m_cv;
    m_cv_last_updated = timestep;

    std::vector<Scalar> cv_val;

    std::vector<PathComponent>::iterator cv_it;
    for (cv_it = m_path_components.begin(); cv_it != m_path_components.end(); ++cv_it)
        {
        cv_val.push_back(cv_it->m_cv->getCurrentValue(timestep));
        }

    // calculate Euclidian norms of the distance to the path
    m_norm=std::vector<Scalar>();
    m_denom=Scalar(0.0);
    std::vector<Scalar>::iterator it;
    for (unsigned int frame = 0; frame < m_num_frames; ++frame)
        {
        Scalar n(0.0);
        for (it = cv_val.begin(); it != cv_val.end(); ++it)
            {
            Scalar del = *it - m_frames[it - cv_val.begin()][frame];
            n += del*del;
            }
        m_norm.push_back(n);
        m_denom += exp(-n*m_scale);
        }
  
    Scalar res;
    m_numerator=Scalar(0.0);
    if (m_direction == parallel)
        {
        // calculate coordinate along the path
        for (it = m_norm.begin(); it != m_norm.end(); ++it)
            {
            Scalar norm = *it;
            m_numerator += Scalar(it-m_norm.begin())*exp(-norm*m_scale);
            }
        res = m_numerator/m_denom/(Scalar(m_num_frames-1));
        }
    else if (m_direction == transverse)
        {
        res = -Scalar(1.0)/m_scale*log(m_denom);
        }
    else
        {
        // we should never get here
        m_exec_conf->msg->error() << "metadynamics.cv.path: Invalid path direction" << std::endl << std::endl;
        throw std::runtime_error("Error evaluating collective variable.");
        }

    m_cv = res;
    
    return res;
    }

void PathCollectiveVariable::computeBiasForces(unsigned int timestep)
    {
    // Compute bias factors for all path component CVs
    for (unsigned int frame = 0; frame < m_num_frames; ++frame)
        {
        std::vector<PathComponent>::iterator cv_it;
        Scalar norm = m_norm[frame];

        for (cv_it = m_path_components.begin(); cv_it != m_path_components.end(); ++cv_it)
            {
            Scalar cur_bias(0.0);
            if (m_direction == parallel)
                {
                cur_bias = -Scalar(frame)*Scalar(2.0)*sqrt(norm)*m_scale/m_denom;
                cur_bias += m_numerator/m_denom/m_denom*Scalar(2.0)*sqrt(norm)*m_scale;
                cur_bias /= Scalar(m_num_frames-1);
                }
            else if (m_direction == transverse)
                {
                cur_bias = Scalar(1.0)/m_denom*Scalar(2.0)*sqrt(norm);
                }

            std::cout << m_bias << " " << m_bias*cur_bias << std::endl;
            cv_it->m_cv->setBiasFactor(m_bias*cur_bias);
            }
        }
    }

void export_PathCollectiveVariable()
    {
    scope in_path_cv = 
    class_<PathCollectiveVariable, boost::shared_ptr<PathCollectiveVariable>, bases<CollectiveVariable>, boost::noncopyable >
        ("PathCollectiveVariable", init< boost::shared_ptr<SystemDefinition>,
                                         PathCollectiveVariable::Enum,
                                         const unsigned int,
                                         const Scalar,
                                         const std::string>())
        .def("registerPathComponent", &PathCollectiveVariable::registerPathComponent)
        .def("removeAllPathComponents", &PathCollectiveVariable::removeAllPathComponents);

    enum_<PathCollectiveVariable::Enum>("direction")
    .value("parallel", PathCollectiveVariable::parallel)
    .value("transverse", PathCollectiveVariable::transverse);
    ;
    } 

