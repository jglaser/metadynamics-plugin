#ifndef __STEINHARDT_QL_H__
#define __STEINHARDT_QL_H__

/*! \file SteinhardtQl.h
    \brief Declares the SteinhardtQl abstract class
 */

#include "CollectiveVariable.h"

#include <hoomd/md/NeighborList.h>

class SteinhardtQl : public CollectiveVariable
    {
    public:
        SteinhardtQl(std::shared_ptr<SystemDefinition> sysdef, Scalar rcut, Scalar ron,
            unsigned int lmax, std::shared_ptr<NeighborList> nlist, unsigned int type,
            const std::vector<Scalar>& Ql_ref,
            const std::string& log_suffix="");
        virtual ~SteinhardtQl() {}

        /*! Returns the current value of the collective variable
         *  \param timestep The currnt value of the timestep
         */
        virtual Scalar getCurrentValue(unsigned int timestep)
            {
            this->computeCV(timestep);
            return m_value;
            }

        /*! Returns the names of provided log quantities.
         */
        virtual std::vector<std::string> getProvidedLogQuantities()
            {
            std::vector<std::string> list = CollectiveVariable::getProvidedLogQuantities();
            list.push_back("cv_steinhardt");
            for (unsigned int l = 1; l <= m_lmax; l++)
                {
                list.push_back("steinhardt_Q"+std::to_string(l));
                }
            return list;
            }

        /*! Returns the value of a specific log quantity.
         * \param quantity The name of the quantity to return the value of
         * \param timestep The current value of the time step
         */
        Scalar getLogValue(const std::string& quantity, unsigned int timestep)
            {
            if (quantity == "cv_steinhardt")
                {
                this->computeCV(timestep);
                return m_value;
                }
            for (unsigned int l = 1; l <= m_lmax; ++l)
                {
                if (quantity == "steinhardt_Q"+std::to_string(l))
                    {
                    this->computeCV(timestep);
                    return m_Ql[l-1];
                    }
                }

            // nothing found?
            m_exec_conf->msg->error() << "cv.steinhardt: Invalid log quantity " << quantity << std::endl;
            throw std::runtime_error("Error querying log quantity");
            }

        // compute the collective variable
        void computeCV(unsigned int teimstep);

    protected:
        /*! Compute the biased forces for this collective variable.
            The force that is written to the force arrays must be
            multiplied by the bias factor.

            \param timestep The current value of the time step
         */
        virtual void computeBiasForces(unsigned int timestep);

    private:
        Scalar m_rcutsq;       //! Cutoff
        Scalar m_ronsq;        //!< Onset of smoothing
        unsigned int m_lmax;   //!< Maxiumum l of spherical harmonics
        std::shared_ptr<NeighborList> m_nlist; //!< The neighbor list
        unsigned int m_type;   //!< Particle type to compute order parameter for

        unsigned int m_cv_last_updated; //!< Last updated timestep
        bool m_have_computed;           //!< True if we have computed the CV at least once

        std::vector<Scalar> m_Ql; //!< List of compute Ql, up to lmax
        std::vector<Scalar> m_Ql_ref; //!< List of reference Ql
        std::string m_prof_name;  //!< Name for profiling
        Scalar m_value;          //!< Value of the collective variable
    };

//! Export the SteinhardtQl class to python
void export_SteinhardtQl(pybind11::module& m);

#endif // __STEINHARDT_QL_H__
