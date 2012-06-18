#include "LamellarOrderParameter.h"

#include <boost/python.hpp>

#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

using namespace boost::python;

LamellarOrderParameter::LamellarOrderParameter(boost::shared_ptr<SystemDefinition> sysdef,
                               const std::vector<Scalar>& mode,
                               const std::vector<int3>& lattice_vectors,
                               const std::vector<Scalar>& phases,
                               const std::string& suffix)
    : CollectiveVariable(sysdef, "cv_lamellar"), m_lattice_vectors(lattice_vectors), m_mode(mode), m_sum(0.0)
    {
    if (mode.size() != m_pdata->getNTypes())
        {
        m_exec_conf->msg->error() << "cv.lamellar: Number of mode parameters has to equal the number of particle types!" << std::endl;
        throw runtime_error("Error initializing cv.lamellar");
        }

    // allocate array of wave vectors
    GPUArray<Scalar3> wave_vectors(m_lattice_vectors.size(), m_exec_conf);
    m_wave_vectors.swap(wave_vectors);

    GPUArray<Scalar2> fourier_modes(m_lattice_vectors.size(), m_exec_conf);
    m_fourier_modes.swap(fourier_modes);

    GPUArray<Scalar> phase_array(m_lattice_vectors.size(), m_exec_conf);
    m_phases.swap(phase_array);

    // Copy over phase shifts
    ArrayHandle<Scalar> h_phases(m_phases, access_location::host, access_mode::overwrite);
    std::copy(phases.begin(), phases.end(), h_phases.data);

    m_cv_name += suffix;
    m_log_name = m_cv_name;
    }

void LamellarOrderParameter::computeForces(unsigned int timestep)
    {
    if (m_prof)
        m_prof->push("cv lamellar");

    ArrayHandle<Scalar4> h_postype(m_pdata->getPositions(), access_location::host, access_mode::read);

    ArrayHandle<Scalar4> h_force(m_force, access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar> h_virial(m_virial, access_location::host, access_mode::overwrite);

    memset(h_virial.data, 0, sizeof(Scalar)*6*m_virial.getPitch());

    calculateWaveVectors();

    calculateFourierModes();

    ArrayHandle<Scalar2> h_fourier_modes(m_fourier_modes, access_location::host, access_mode::read);

    ArrayHandle<Scalar3> h_wave_vectors(m_wave_vectors, access_location::host, access_mode::read);

    ArrayHandle<Scalar> h_phases(m_phases, access_location::host, access_mode::read);

    Scalar3 L = m_pdata->getGlobalBox().getL();
    Scalar V = L.x*L.y*L.z;

    for (unsigned int idx = 0; idx < m_pdata->getN(); idx++)
        {
        Scalar4 postype = h_postype.data[idx];

        Scalar3 pos = make_scalar3(postype.x, postype.y, postype.z);
        unsigned int type = __scalar_as_int(postype.w);
        Scalar mode = m_mode[type];

        Scalar4 force_energy = make_scalar4(0,0,0,0);

        for (unsigned int k = 0; k < m_wave_vectors.getNumElements(); k++)
            {
            Scalar3 q = h_wave_vectors.data[k];
            Scalar dotproduct = dot(pos,q);

            Scalar f; 
            f = mode*sin(dotproduct + h_phases.data[k]);

            force_energy.x += q.x*f;
            force_energy.y += q.y*f;
            force_energy.z += q.z*f;
            }

        force_energy.x *= m_bias;
        force_energy.y *= m_bias;
        force_energy.z *= m_bias;

        force_energy.x /= V;
        force_energy.y /= V;
        force_energy.z /= V;

        h_force.data[idx] = force_energy;
        }

    // Calculate value of collective variable (sum of real parts of fourier modes)
    m_sum = 0.0;
    for (unsigned k = 0; k < m_fourier_modes.getNumElements(); k++)
        {
        Scalar2 fourier_mode = h_fourier_modes.data[k];
        m_sum += fourier_mode.x;
        }

    m_sum /= V;

    if (m_prof)
        m_prof->pop();
    }

//! Calculate wave vectors
void LamellarOrderParameter::calculateWaveVectors()
    {
    ArrayHandle<Scalar3> h_wave_vectors(m_wave_vectors, access_location::host, access_mode::overwrite);

    const BoxDim &box = m_pdata->getGlobalBox();
    const Scalar3 L = box.getL();

    for (unsigned int k = 0; k < m_lattice_vectors.size(); k++)
        h_wave_vectors.data[k] = 2*M_PI*make_scalar3(m_lattice_vectors[k].x/L.x,
                                              m_lattice_vectors[k].y/L.y,
                                              m_lattice_vectors[k].z/L.z);
    }

//! Returns a list of fourier modes (for all wave vectors)
void LamellarOrderParameter::calculateFourierModes()
    {
    ArrayHandle<Scalar2> h_fourier_modes(m_fourier_modes, access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar3> h_wave_vectors(m_wave_vectors, access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_phases(m_phases, access_location::host, access_mode::read);

    ArrayHandle<Scalar4> h_postype(m_pdata->getPositions(), access_location::host, access_mode::read);

    for (unsigned int k = 0; k < m_wave_vectors.getNumElements(); k++)
        {
        h_fourier_modes.data[k] = make_scalar2(0.0,0.0);
        Scalar3 q = h_wave_vectors.data[k];
        
        for (unsigned int idx = 0; idx < m_pdata->getN(); idx++)
            {
            Scalar4 postype = h_postype.data[idx];

            Scalar3 pos = make_scalar3(postype.x, postype.y, postype.z);
            unsigned int type = __scalar_as_int(postype.w);
            Scalar mode = m_mode[type];
            Scalar dotproduct = dot(q,pos);
            h_fourier_modes.data[k].x += mode * cos(dotproduct + h_phases.data[k]);
            h_fourier_modes.data[k].y += mode * sin(dotproduct + h_phases.data[k]);
            }
        }
    }

Scalar LamellarOrderParameter::getLogValue(const std::string& quantity, unsigned int timestep)
    {
    if (quantity == m_log_name)
        {
        this->compute(timestep);
        return m_sum;
        }
    else
        {
        this->m_exec_conf->msg->error() << "cv.lamellar: " << quantity << " is not a valid log quantity"
                  << std::endl;
        throw std::runtime_error("Error getting log value");
        }
    }

void export_LamellarOrderParameter()
    {
    class_<LamellarOrderParameter, boost::shared_ptr<LamellarOrderParameter>, bases<CollectiveVariable>, boost::noncopyable >
        ("LamellarOrderParameter", init< boost::shared_ptr<SystemDefinition>,
                                         const std::vector<Scalar>&,
                                         const std::vector<int3>,
                                         const std::vector<Scalar>,
                                         const std::string&>());

    class_<std::vector<int3> >("std_vector_int3")
        .def(vector_indexing_suite< std::vector<int3> > ())
        ;
    }
