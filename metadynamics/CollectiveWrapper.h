#ifndef __COLLECTIVE_WRAPPER_H__
#define __COLLECTIVE_WRAPPER_H__

#include "CollectiveVariable.h"

#include <hoomd/GlobalArray.h>
#include <hoomd/Autotuner.h>

/*! Wrapper to convert a regular ForceCompute into a CollectiveVariable */

class CollectiveWrapper : public CollectiveVariable
    {
    public:
        /*! Constructs the collective variable
            \param sysdef The system definition
            \param name The name of this collective variable
         */
        CollectiveWrapper(std::shared_ptr<SystemDefinition> sysdef, std::shared_ptr<ForceCompute> fc, const std::string& name);
        virtual ~CollectiveWrapper() {}

        /*! Returns the current value of the collective variable
         *  \param timestep The currnt value of the timestep
         */
        virtual Scalar getCurrentValue(unsigned int timestep)
            {
            computeCV(timestep);
            return m_energy;
            }

        //! Set autotuner parameters
        /*! \param enable Enable/disable autotuning
            \param period period (approximate) in time steps when returning occurs
        */
        virtual void setAutotunerParams(bool enable, unsigned int period)
            {
            CollectiveVariable::setAutotunerParams(enable, period);
            m_tuner_reduce->setPeriod(period);
            m_tuner_reduce->setEnabled(enable);
            m_tuner_scale->setPeriod(period);
            m_tuner_scale->setEnabled(enable);
            }

    protected:
        std::shared_ptr<ForceCompute> m_fc; //!< The parent force compute
        Scalar m_energy;                    //!< The potential energy

        #ifdef ENABLE_HIP
        GlobalArray<Scalar> m_sum;     //!< for reading back potential energy from GPU
        std::unique_ptr<Autotuner> m_tuner_scale; //!< Autotuner for scaling forces
        std::unique_ptr<Autotuner> m_tuner_reduce; //!< Autotuner for collective variable reduction
        #endif

        virtual void computeCV(unsigned int timestep);

        /*! Compute the biased forces for this collective variable.
            The force that is written to the force arrays must be
            multiplied by the bias factor.

            \param timestep The current value of the time step
         */
        virtual void computeBiasForces(unsigned int timestep);

        //! Compute bias force on GPU
        void computeBiasForcesGPU(unsigned int timestep);

        //! Compute collective variable on GPU
        void computeCVGPU(unsigned int timestep);
    };

//! Export the CollectiveVariable class to python
void export_CollectiveWrapper(pybind11::module& m);

#endif // __COLLECTIVE_WRAPPER_H__
