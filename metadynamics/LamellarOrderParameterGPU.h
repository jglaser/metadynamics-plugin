/*! \file LamellarOrderParameterGPU.h
 *  \brief Defines the LamellarOrderParameterGPU class
 */

#ifndef __LAMELLAR_ORDER_PARAMETER_GPU_H__
#define __LAMELLAR_ORDER_PARAMETER_GPU_H__

#include "LamellarOrderParameter.h"

#ifdef ENABLE_HIP

//! Class to calculate the lamellar order parameter on the GPU
class LamellarOrderParameterGPU : public LamellarOrderParameter
    {
    public:
        LamellarOrderParameterGPU(std::shared_ptr<SystemDefinition> sysdef,
                               const std::vector<Scalar>& mode,
                               const std::vector<int3>& lattice_vectors,
                               const std::string& suffix = "");

        virtual ~LamellarOrderParameterGPU() {}

        virtual void computeBiasForces(unsigned int timestep);

    protected:
        // calculates current CV value
        virtual void computeCV(unsigned int timestep);
    private:
        GPUArray<Scalar> m_gpu_mode;       //!< Factors multiplying per-type densities to obtain scalar quantity
        unsigned int m_block_size;          //!< Block size for fourier mode calculation
        GPUArray<Scalar2> m_fourier_mode_scratch; //!< Scratch memory for fourier mode calculation
    };

void export_LamellarOrderParameterGPU(pybind11::module& m);
#endif
#endif // __LAMELLAR_ORDER_PARAMETER_GPU_H__
