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

        void testFFT();
    protected:
        //! Helper function to setup FFT and allocate the mesh arrays
        virtual void initializeFFT();

        //! Helper function to assign particle coordinates to mesh
        virtual void assignParticles();

        //! Helper function to update the mesh arrays
        virtual void updateMeshes();

        //! Helper function to interpolate the forces
        virtual void interpolateForces();

        //! Compute the optimal influence function
        virtual void computeInfluenceFunction();

        //! Helper function to calculate value of collective variable
        virtual Scalar computeCV();

    private:
        cufftHandle m_cufft_plan;          //!< The FFT plan
        cufftHandle m_cufft_plan_force;    //!< The FFT plan for the force mesh

        GPUArray<cufftReal> m_mesh;                //!< The particle density mesh
        GPUArray<cufftComplex> m_fourier_mesh;     //!< The fourier transformed mesh
        GPUArray<cufftComplex> m_fourier_mesh_G;   //!< Fourier transformed mesh times the influence function
        GPUArray<cufftComplex> m_fourier_mesh_force; //!< The fourier transformed force mesh
        GPUArray<cufftReal> m_ifourier_mesh_force;//!< The inverse-fourier transformed force mesh
        GPUArray<Scalar4> m_force_mesh;             //!< The force mesh

        GPUArray<Scalar4> m_particle_bins;         //!< Cell list for particle positions and modes
        GPUArray<unsigned int> m_n_cell;           //!< Number of particles per cell
        unsigned int m_cell_size;                  //!< Current max. number of particles per cell
        GPUFlags<unsigned int> m_cell_overflowed;  //!< Flag set to 1 if a cell overflows

        GPUFlags<Scalar> m_sum;                    //!< Sum over fourier mesh values
        GPUArray<Scalar> m_sum_partial;            //!< Partial sums over fourier mesh values
        unsigned int m_block_size;                 //!< Block size for fourier mesh reduction
        unsigned m_num_fourier_cells;              //!< Number of complex values in fourier mesh
    };

void export_OrderParameterMeshGPU();

#endif
