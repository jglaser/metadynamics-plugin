/* \file LamellarOrderParameter.cc
 * \brief Implements the LamellarOrderParameter class
 */
#include "LamellarOrderParameter.h"

namespace py = pybind11;

LamellarOrderParameter::LamellarOrderParameter(std::shared_ptr<SystemDefinition> sysdef,
                               const std::vector<Scalar>& mode,
                               const std::vector<int3>& lattice_vectors,
                               const std::string& suffix)
    : CollectiveVariable(sysdef, "cv_lamellar"), m_mode(mode), m_cv(0.0)
    {
    if (mode.size() != m_pdata->getNTypes())
        {
        m_exec_conf->msg->error() << "cv.lamellar: Number of mode parameters has to equal the number of particle types!" << std::endl;
        throw std::runtime_error("Error initializing cv.lamellar");
        }

    // allocate array of wave vectors
    GPUArray<int3> lattice_vectors_gpuarray(lattice_vectors.size(), m_exec_conf);
    m_lattice_vectors.swap(lattice_vectors_gpuarray);

    GPUArray<Scalar2> fourier_modes(lattice_vectors.size(), m_exec_conf);
    m_fourier_modes.swap(fourier_modes);

    m_cv_name += suffix;
    m_log_name = m_cv_name;

    // this collective variable does not contribute to the virial
    ArrayHandle<Scalar> h_virial(m_virial, access_location::host, access_mode::overwrite);
    memset(h_virial.data, 0, sizeof(Scalar)*6*m_virial.getPitch());

    m_cv_last_updated = 0;

    // copy over lattice vectors
    ArrayHandle<int3> h_lattice_vectors(m_lattice_vectors, access_location::host, access_mode::overwrite);
    for (unsigned int k = 0; k < lattice_vectors.size(); k++)
        h_lattice_vectors.data[k] = lattice_vectors[k];
    }

void LamellarOrderParameter::computeCV(unsigned int timestep)
    {
    if (m_prof)
        m_prof->push("Lamellar");

    calculateFourierModes();

    ArrayHandle<Scalar2> h_fourier_modes(m_fourier_modes, access_location::host, access_mode::readwrite);

    unsigned int N = m_pdata->getNGlobal();

#ifdef ENABLE_MPI
    // reduce Fourier modes on on all processors
    if (m_pdata->getDomainDecomposition())
        MPI_Allreduce(MPI_IN_PLACE,h_fourier_modes.data,m_fourier_modes.getNumElements(), MPI_HOOMD_SCALAR, MPI_SUM, m_exec_conf->getMPICommunicator());
#endif

    // Calculate value of collective variable (sum of real parts of fourier modes)
    Scalar sum = 0.0;
    for (unsigned k = 0; k < m_fourier_modes.getNumElements(); k++)
        {
        Scalar2 fourier_mode = h_fourier_modes.data[k];
        sum += fourier_mode.x;
        }
    sum /= (Scalar) N;

    m_cv = sum;

    if (m_prof)
        m_prof->pop();

    m_cv_last_updated = timestep;
    }


void LamellarOrderParameter::computeBiasForces(unsigned int timestep)
    {
    if (m_prof)
        m_prof->push("Lamellar");

    if (m_cv_last_updated < timestep || timestep == 0)
        computeCV(timestep);

    ArrayHandle<Scalar4> h_postype(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_force(m_force, access_location::host, access_mode::overwrite);
    ArrayHandle<int3> h_lattice_vectors(m_lattice_vectors, access_location::host, access_mode::read);

    unsigned int N = m_pdata->getNGlobal();

    Scalar denom = (Scalar)N;

    // compute reciprocal lattice vectors
    const BoxDim& global_box = m_pdata->getGlobalBox();
    Scalar3 a1 = global_box.getLatticeVector(0);
    Scalar3 a2 = global_box.getLatticeVector(1);
    Scalar3 a3 = global_box.getLatticeVector(2);

    Scalar V_box = global_box.getVolume();
    Scalar3 b1 = Scalar(2.0*M_PI)*make_scalar3(a2.y*a3.z-a2.z*a3.y, a2.z*a3.x-a2.x*a3.z, a2.x*a3.y-a2.y*a3.x)/V_box;
    Scalar3 b2 = Scalar(2.0*M_PI)*make_scalar3(a3.y*a1.z-a3.z*a1.y, a3.z*a1.x-a3.x*a1.z, a3.x*a1.y-a3.y*a1.x)/V_box;
    Scalar3 b3 = Scalar(2.0*M_PI)*make_scalar3(a1.y*a2.z-a1.z*a2.y, a1.z*a2.x-a1.x*a2.z, a1.x*a2.y-a1.y*a2.x)/V_box;

    for (unsigned int idx = 0; idx < m_pdata->getN(); idx++)
        {
        Scalar4 postype = h_postype.data[idx];

        Scalar3 pos = make_scalar3(postype.x, postype.y, postype.z);
        unsigned int type = __scalar_as_int(postype.w);
        Scalar mode = m_mode[type];

        Scalar4 force_energy = make_scalar4(0,0,0,0);

        for (unsigned int k = 0; k < m_lattice_vectors.getNumElements(); k++)
            {
            Scalar3 q = b1*(Scalar)h_lattice_vectors.data[k].x + b2*(Scalar)h_lattice_vectors.data[k].y + b3*(Scalar)h_lattice_vectors.data[k].z;
            Scalar dotproduct = dot(pos,q);

            Scalar f;
            f = Scalar(2.0)*mode*sin(dotproduct);

            force_energy.x += q.x*f;
            force_energy.y += q.y*f;
            force_energy.z += q.z*f;
            }

        force_energy.x *= m_bias;
        force_energy.y *= m_bias;
        force_energy.z *= m_bias;

        force_energy.x /= denom;
        force_energy.y /= denom;
        force_energy.z /= denom;

        h_force.data[idx] = force_energy;
        }

    if (m_prof)
        m_prof->pop();
    }

//! Returns a list of fourier modes (for all wave vectors)
void LamellarOrderParameter::calculateFourierModes()
    {
    ArrayHandle<Scalar2> h_fourier_modes(m_fourier_modes, access_location::host, access_mode::overwrite);
    ArrayHandle<int3> h_lattice_vectors(m_lattice_vectors, access_location::host, access_mode::read);

    ArrayHandle<Scalar4> h_postype(m_pdata->getPositions(), access_location::host, access_mode::read);

    // compute reciprocal lattice vectors
    const BoxDim& global_box = m_pdata->getGlobalBox();
    Scalar3 a1 = global_box.getLatticeVector(0);
    Scalar3 a2 = global_box.getLatticeVector(1);
    Scalar3 a3 = global_box.getLatticeVector(2);

    Scalar V_box = global_box.getVolume();
    Scalar3 b1 = Scalar(2.0*M_PI)*make_scalar3(a2.y*a3.z-a2.z*a3.y, a2.z*a3.x-a2.x*a3.z, a2.x*a3.y-a2.y*a3.x)/V_box;
    Scalar3 b2 = Scalar(2.0*M_PI)*make_scalar3(a3.y*a1.z-a3.z*a1.y, a3.z*a1.x-a3.x*a1.z, a3.x*a1.y-a3.y*a1.x)/V_box;
    Scalar3 b3 = Scalar(2.0*M_PI)*make_scalar3(a1.y*a2.z-a1.z*a2.y, a1.z*a2.x-a1.x*a2.z, a1.x*a2.y-a1.y*a2.x)/V_box;


    for (unsigned int k = 0; k < m_lattice_vectors.getNumElements(); k++)
        {
        h_fourier_modes.data[k] = make_scalar2(0.0,0.0);
        Scalar3 q = b1*(Scalar)h_lattice_vectors.data[k].x + b2*(Scalar)h_lattice_vectors.data[k].y + b3*(Scalar)h_lattice_vectors.data[k].z;

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
        computeCV(timestep);
        return m_cv;
        }

    // nothing found, turn to base class
    return CollectiveVariable::getLogValue(quantity, timestep);
    }

void export_LamellarOrderParameter(py::module& m)
    {
    py::class_<LamellarOrderParameter, std::shared_ptr<LamellarOrderParameter> >(m,"LamellarOrderParameter",py::base<CollectiveVariable>())
        .def(py::init<std::shared_ptr<SystemDefinition>,
                     const std::vector<Scalar>&,
                     const std::vector<int3>,
                     const std::string&>());
    }
