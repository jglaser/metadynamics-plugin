#include "AspectRatio.h"

AspectRatio::AspectRatio(boost::shared_ptr<SystemDefinition> sysdef, const unsigned int dir1, const unsigned int dir2)
    : CollectiveVariable(sysdef, "cv_aspect_ratio"), m_dir1(dir1), m_dir2(dir2)
    {
    if (dir1==dir2 || dir1 >= 3 || dir2 >= 3)
        {
        m_exec_conf->msg->error() << "metadynamics.aspect_ratio: Invalid directions given." << std::endl;
        throw std::runtime_error("Error setting up metadynamics.aspect_ratio");
        }

    // reset force and virial
    ArrayHandle<Scalar4> h_force(m_force, access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar> h_virial(m_virial, access_location::host, access_mode::overwrite);

    memset(h_force.data, 0, sizeof(Scalar4)*m_force.getNumElements());
    memset(h_virial.data, 0, sizeof(Scalar)*m_virial.getNumElements());

    m_log_name = m_cv_name;
    }

Scalar AspectRatio::getCurrentValue(unsigned int timestep)
    {
    Scalar3 L = m_pdata->getGlobalBox().getL();

    Scalar length1(0.0), length2(0.0);

    switch(m_dir1)
        {
        case 0:
            length1 = L.x;
            break;
        case 1:
            length1 = L.y;
            break;
        case 2:
            length1 = L.z;
            break;
        }

    switch(m_dir2)
        {
        case 0:
            length1 = L.x;
            break;
        case 1:
            length2 = L.y;
            break;
        case 2:
            length2 = L.z;
            break;
        }

    return length1/length2;
    }

void AspectRatio::computeBiasForces(unsigned int timestep)
    {
    Scalar3 L = m_pdata->getGlobalBox().getL();

    Scalar d_l_x(0.0), d_l_y(0.0), d_l_z(0.0);

    switch(m_dir1)
        {
        case 0:
            switch(m_dir2)
                {
                case 1:
                    d_l_x = Scalar(1.0)/L.y;
                    d_l_y = -L.x/L.y/L.y;
                    d_l_z = Scalar(0.0);
                    break;
                case 2:
                    d_l_x = Scalar(1.0)/L.z;
                    d_l_y = Scalar(0.0);
                    d_l_z = -L.x/L.z/L.z;
                    break;
                }
                break;
        case 1:
            switch(m_dir2)
                {
                case 0:
                    d_l_x = -L.y/L.x/L.x;
                    d_l_y = Scalar(1.0)/L.x;
                    d_l_z = Scalar(0.0);
                    break;
                case 2:
                    d_l_x = Scalar(0.0);
                    d_l_y = Scalar(1.0)/L.z;
                    d_l_z = -L.y/L.z/L.z;
                    break;
                }
                break;
         case 2:
            switch(m_dir2)
                {
                case 0:
                    d_l_x = -L.z/L.x/L.x;
                    d_l_y = Scalar(0.0);
                    d_l_z = Scalar(1.0)/L.x;
                    break;
                case 1:
                    d_l_x = Scalar(0.0);
                    d_l_y = -L.z/L.y/L.y;
                    d_l_z = Scalar(1.0)/L.y;
                    break;
                }
            break;
        }

    m_external_virial[0] = - m_bias*d_l_x * L.x;
    m_external_virial[1] = - m_bias*Scalar(1.0/2.0)*(d_l_x * L.y + d_l_y * L.x);
    m_external_virial[2] = - m_bias*Scalar(1.0/2.0)*(d_l_x * L.z + d_l_z * L.x);
    m_external_virial[3] = - m_bias*d_l_y * L.y;
    m_external_virial[4] = - m_bias*Scalar(1.0/2.0)*(d_l_y * L.z + d_l_z * L.y);
    m_external_virial[5] = - m_bias*d_l_z * L.z;
    }

void export_AspectRatio()
    {
    class_<AspectRatio, boost::shared_ptr<AspectRatio>, bases<CollectiveVariable>, boost::noncopyable >
        ("AspectRatio", init< boost::shared_ptr<SystemDefinition>,
                                         const unsigned int,
                                         const unsigned int >() );
        ;
    }

