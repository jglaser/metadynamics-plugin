#include "LamellarOrderParameter.h"

#include <boost/python.hpp>

#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

using namespace boost::python;

LamellarOrderParameter::LamellarOrderParameter(boost::shared_ptr<SystemDefinition> sysdef,
                               const std::vector<Scalar>& mode,
                               const std::vector<int3>& lattice_vectors,
                               bool generate_symmetries,
                               const std::string& suffix)
    : CollectiveVariable(sysdef, "cv_lamellar"), m_sum(0.0)
    {
    if (mode.size() != m_pdata->getNTypes())
        {
        m_exec_conf->msg->error() << "cv.lamellar: Number of mode parameters has to equal the number of particle types!" << std::endl;
        throw runtime_error("Error initializing cv.lamellar");
        }

    m_mode = mode;
    if (generate_symmetries)
        m_lattice_vectors = applyCubicSymmetries(lattice_vectors);
    else
        m_lattice_vectors = lattice_vectors;

    // allocate array of wave vectors
    GPUArray<Scalar3> wave_vectors(m_lattice_vectors.size(), m_exec_conf);
    m_wave_vectors.swap(wave_vectors);

    GPUArray<Scalar2> fourier_modes(m_lattice_vectors.size(), m_exec_conf);
    m_fourier_modes.swap(fourier_modes);

    m_cv_name += suffix;
    m_log_name = m_cv_name;
    }

/*! Apply all rotations and mirror symmetries of the cubic lattice to the list
    of input wave vectors
 */
const std::vector<int3> LamellarOrderParameter::applyCubicSymmetries(const std::vector<int3>& lattice_vectors)
    {
    std::vector<int3> out;

    for (unsigned int n = 0; n < lattice_vectors.size(); n++)
        {
        int cur_vec[3] = {lattice_vectors[n].x, lattice_vectors[n].y, lattice_vectors[n].z};
        std::vector<int3> symmetries;

        // apply permutations
        for (unsigned int i = 0; i < 3; i++)
            for (unsigned int j = 0; j < 2; j++)
                {
                if (j==i) j++;

                unsigned int k = 0;

                if (k == i || k==j) k++;
                if (k == i || k==j) k++;
                
                symmetries.push_back( make_int3(cur_vec[i], cur_vec[j], cur_vec[k]));
                }


        // apply mirror symmetries
        unsigned int size = symmetries.size();
        for (unsigned int i = 0; i < 2; i++)
            for (unsigned int j = 0; j < 2; j++)
                for (unsigned int k = 0; k < 2; k++)
                    for (unsigned int l= 0; l < size; l++)
                        symmetries.push_back( make_int3( symmetries[l].x * ((i==0) ? 1 : -1),
                                                         symmetries[l].y * ((l==0) ? 1 : -1),
                                                         symmetries[l].z * ((k==0) ? 1 : -1)));
    
        // remove duplicate wave vectors
        for (unsigned int i = 0; i < symmetries.size(); i++)
            {
            bool found = false;

            for (unsigned  j = 0; j < i; j++)
                if (symmetries[i] == symmetries[j])
                    found = true;

            if (! found)
                {
                out.push_back(symmetries[i]);
                m_exec_conf->msg->notice(6) << "cv.lamellar: Adding wave vector (" << symmetries[i].x << ","
                                            << symmetries[i].y << "," << symmetries[i].z << ")" << std::endl;
                }
            }
        }
    return out;
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
    unsigned int size = m_wave_vectors.getNumElements();

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

            Scalar2 exponential;
            exponential.x = mode*cos(dotproduct);
            exponential.y = -mode*sin(dotproduct);

            Scalar2 fourier_mode = h_fourier_modes.data[k];
            Scalar im = -(exponential.x*fourier_mode.y + exponential.y*fourier_mode.x);

            force_energy.x += Scalar(2.0)*q.x*im;
            force_energy.y += Scalar(2.0)*q.y*im;
            force_energy.z += Scalar(2.0)*q.z*im;
            }

        force_energy.x *= m_bias;
        force_energy.y *= m_bias;
        force_energy.z *= m_bias;

        force_energy.x /= size *V;
        force_energy.y /= size *V;
        force_energy.z /= size *V;

        h_force.data[idx] = force_energy;
        }

    // Calculate value of collective variable (avg of structure factors)    
    m_sum = 0.0;
    for (unsigned k = 0; k < m_fourier_modes.getNumElements(); k++)
        {
        Scalar2 fourier_mode = h_fourier_modes.data[k];
        m_sum += fourier_mode.x * fourier_mode.x + fourier_mode.y * fourier_mode.y;
        }

    m_sum /= V * m_fourier_modes.getNumElements();

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
            h_fourier_modes.data[k].x += mode * cos(dotproduct);
            h_fourier_modes.data[k].y += mode * sin(dotproduct);
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
                                         bool,
                                         const std::string&>());

    class_<std::vector<int3> >("std_vector_int3")
        .def(vector_indexing_suite< std::vector<int3> > ())
        ;
    }
