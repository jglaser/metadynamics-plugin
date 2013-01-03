#include <hoomd/hoomd.h>

#include "OrderParameterMesh.h"

#ifndef __ORDER_PARAMETER_MESH_GPU_H__
#define __ORDER_PARAMETER_MESH_GPU_H__

/*! Order parameter evaluated using the particle mesh method
 */
class OrderParameterMeshGPU : public OrderParameterMesh
    {
    public:
        //! Constructor
        OrderParameterMeshGPU(boost::shared_ptr<SystemDefinition> sysdef,
                           const unsigned int nx,
                           const unsigned int ny,
                           const unsigned int nz,
                           const Scalar qstar,
                           const std::vector<Scalar> mode);
        virtual ~OrderParameterMeshGPU();

    protected:
        //! Helper function to setup FFT and allocate the mesh arrays
        virtual void initializeFFT();

        //! Helper function to assign particle coordinates to mesh
        virtual void assignParticles();

        //! Helper function to update the mesh arrays
        virtual void updateMeshes();

        //! Helper function to interpolate the forces
        virtual void interpolateForces();

        //! Helper function to calculate value of collective variable
        virtual Scalar computeCV();

    private:
        cufftHandle m_cufft_plan;          //!< The FFT plan

        GPUArray<cufftComplex> m_mesh;             //!< The particle density mesh
        GPUArray<cufftComplex> m_fourier_mesh;     //!< The fourier transformed mesh
        GPUArray<cufftComplex> m_fourier_mesh_G;   //!< Fourier transformed mesh times the influence function
        GPUArray<cufftComplex> m_fourier_mesh_x;   //!< The fourier transformed force mesh, x component
        GPUArray<cufftComplex> m_fourier_mesh_y;   //!< The fourier transformed force mesh, y component
        GPUArray<cufftComplex> m_fourier_mesh_z;   //!< The fourier transformed force mesh, z component
        GPUArray<cufftComplex> m_force_mesh_x;        //!< The force mesh, x component
        GPUArray<cufftComplex> m_force_mesh_y;        //!< The force mesh, y component
        GPUArray<cufftComplex> m_force_mesh_z;        //!< The force mesh, z component
        GPUArray<Scalar4> m_force_mesh;             //!< The force mesh

        GPUFlags<Scalar> m_sum;                    //!< Sum over fourier mesh values
        GPUArray<Scalar> m_sum_partial;            //!< Partial sums over fourier mesh values
        unsigned int m_block_size;                 //!< Block size for fourier mesh reduction
    };

void export_OrderParameterMeshGPU();

#endif
