#include "LamellarOrderParameter.h"

#ifdef ENABLE_CUDA

class LamellarOrderParameterGPU : public LamellarOrderParameter
    {
    public:
        LamellarOrderParameterGPU(boost::shared_ptr<SystemDefinition> sysdef,
                               const std::vector<Scalar>& mode,
                               const std::vector<int3>& lattice_vectors,
                               bool generate_symmetries);

        virtual ~LamellarOrderParameterGPU() {}

        virtual void computeForces(unsigned int timestep);

    private:
        GPUArray<Scalar> m_gpu_mode;       //!< Factors multiplying per-type densities to obtain scalar quantity
    };

void export_LamellarOrderParameterGPU();
#endif
