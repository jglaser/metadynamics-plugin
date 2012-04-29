#include <hoomd/hoomd.h>

#ifndef __LAMELLAR_ORDER_PARAMETER_H__
#define __LAMELLAR_ORDER_PARAMETER_H__

#include "CollectiveVariable.h"

// need to declare these classes with __host__ __device__ qualifiers when building in nvcc
// HOSTDEVICE is __host__ __device__ when included in nvcc and blank when included into the host compiler
#ifdef NVCC
#define HOSTDEVICE __host__ __device__
#else
#define HOSTDEVICE
#endif

class LamellarOrderParameter : public CollectiveVariable
    {
    public:
        LamellarOrderParameter(boost::shared_ptr<SystemDefinition> sysdef,
                               const std::vector<Scalar>& mode,
                               const std::vector<int3>& lattice_vectors,
                               bool generate_symmetries);
        virtual ~LamellarOrderParameter() {}

        virtual void computeForces(unsigned int timestep);

        std::vector<std::string> getProvidedLogQuantities()
            {
            std::vector<std::string> list;
            list.push_back(m_log_name);
            return list;
            }

        Scalar getLogValue(const std::string& quantity, unsigned int timestep);

        Scalar getCurrentValue(unsigned int timestep)
            {
            this->compute(timestep);
            return m_sum;
            } 

    protected:
        std::string m_log_name;
        std::vector<int3> m_lattice_vectors;
        std::vector<Scalar> m_mode;

        Scalar m_sum; 

        GPUArray<Scalar3> m_wave_vectors;
        GPUArray<Scalar2> m_fourier_modes; //!< Fourier modes

        const std::vector<int3> applyCubicSymmetries(const std::vector<int3>& lattice_vectors);

        void calculateWaveVectors();

    private:
        void calculateFourierModes();

    };

// ------------ Vector math functions --------------------------
//! Comparison operator needed for export of std::vector<int3>
HOSTDEVICE inline bool operator== (const int3 &a, const int3 &b)
    {
    return (a.x == b.x &&
            a.y == b.y &&
            a.z == b.z);
    }


void export_LamellarOrderParameter();

#endif // __LAMELLAR_ORDER_PARAMETER_H__
