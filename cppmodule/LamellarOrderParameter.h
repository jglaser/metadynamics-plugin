#ifndef __LAMELLAR_ORDER_PARAMETER_H__
#define __LAMELLAR_ORDER_PARAMETER_H__

/*! \file LamellarOrderParameter.h
    \brief Declares the LamellarOrderParameter class
 */
#include <hoomd/hoomd.h>

#include <string.h>

#include "CollectiveVariable.h"

// need to declare these classes with __host__ __device__ qualifiers when building in nvcc
// HOSTDEVICE is __host__ __device__ when included in nvcc and blank when included into the host compiler
#ifdef NVCC
#define HOSTDEVICE __host__ __device__
#else
#define HOSTDEVICE
#endif

//! Collective variable for studying phase transitions in block copolymer systems
/*! This class implements a collective variable based on the Fourier modes of
   concentration fluctuations.

   The value of the collective variable \f$ s \f$ is given by
   \f$ s = \sum_{i = 1}^n \sum_{j = 1}^N a(type_j \cos(\mathbf{q}_i\mathbf{r_j} + \phi_i)) \f$,
   where \f$n\f$ is the number of modes supplied,
   \f$ \mathbf{q}_i = 2 \pi (n_{i,x}/L_x, n_{i,y}/L_y, n_{i,z}/L_z) \f$ is the 
   wave vector associated with mode \f$i\f$, \f$\phi_i\f$ its phase shift,
   and $a(type_j)$ is the mode coefficient for a particle of type \f$type\f$.

   The force is calculated as minus the derivative of \f$s\f$ with respect
   to particle positions \f$\mathbf{r}_j\f$, multiplied by the bias factor.
*/
class LamellarOrderParameter : public CollectiveVariable
    {
    public:
        /*! Constructs the collective variable
            \param sysdef The system definition
            \param mode The per-type coefficients of the Fourier mode
            \param lattice_vectors The Miller indices of the mode vector
            \param phases The phases for every mode
            \param suffix The suffix appended to the log name for this quantity
         */
        LamellarOrderParameter(boost::shared_ptr<SystemDefinition> sysdef,
                               const std::vector<Scalar>& mode,
                               const std::vector<int3>& lattice_vectors,
                               const std::vector<Scalar>& phases,
                               const std::string& suffix = ""
                               );
        virtual ~LamellarOrderParameter() {}

        /*! Compute the forces for this collective variable.
            The force that is written to the force arrays must be
            multiplied by the bias factor.

            \param timestep The current value of the time step
         */
        virtual void computeForces(unsigned int timestep);

        /*! Returns the names of provided log quantities.
         */
        std::vector<std::string> getProvidedLogQuantities()
            {
            std::vector<std::string> list;
            list.push_back(m_log_name);
            return list;
            }

        /*! Returns the value of a specific log quantity.
         * \param quantity The name of the quantity to return the value of
         * \param timestep The current value of the time step
         */
        Scalar getLogValue(const std::string& quantity, unsigned int timestep);

        /*! Returns the current value of the collective variable
         * \param timestep The current value of the time step
         */
        Scalar getCurrentValue(unsigned int timestep)
            {
            this->compute(timestep);
            return m_sum;
            } 

    protected:
        std::string m_log_name;               //!< The log name for this collective variable
        std::vector<int3> m_lattice_vectors;  //!< Stores the list of miller indices
        std::vector<Scalar> m_mode;           //!< Stores the per-type mode coefficients

        Scalar m_sum;                         //!< The current value of the collective variable

        GPUArray<Scalar3> m_wave_vectors;     //!< GPUArray of wave vectors
        GPUArray<Scalar2> m_fourier_modes;    //!< Fourier modes
        GPUArray<Scalar> m_phases;            //!< Phase shifts

        //! Helper function to update the wave vectors
        void calculateWaveVectors();

    private:
        //! Helper function to calculate the Fourier modes
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

//! Export LamellarOrderParameter to python
void export_LamellarOrderParameter();

#endif // __LAMELLAR_ORDER_PARAMETER_H__
