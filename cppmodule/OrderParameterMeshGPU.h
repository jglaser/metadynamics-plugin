#include <hoomd/hoomd.h>

#include "OrderParameterMesh.h"

#ifndef __ORDER_PARAMETER_MESH_GPU_H__
#define __ORDER_PARAMETER_MESH_GPU_H__

#ifdef ENABLE_CUDA
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

        //! Compute the optimal influence function
        virtual void computeInfluenceFunction();

        //! Helper function to calculate value of collective variable
        virtual Scalar computeCV();

        //! Helper function to compute the virial
        virtual void computeVirial();

    private:
        cufftHandle m_cufft_plan;          //!< The FFT plan
        bool m_local_fft;                  //!< True if we are only doing local FFTs (not distributed)

        #ifdef ENABLE_MPI
        typedef CommunicatorMeshGPU<cufftComplex, gpu_communicate_complex_mesh_map> CommunicatorMeshGPUComplex;
        typedef CommunicatorMeshGPU<Scalar4, gpu_communicate_scalar4_mesh_map> CommunicatorMeshGPUScalar4;

        boost::shared_ptr<DistributedFFTGPU> m_gpu_dfft;  //!< Distributed FFT for forward and inverse transforms
        boost::shared_ptr<CommunicatorMeshGPUComplex> m_gpu_mesh_comm_forward; //!< Communicator for density map
        boost::shared_ptr<CommunicatorMeshGPUScalar4> m_gpu_mesh_comm_inverse; //!< Communicator for force mesh
        #endif

        GPUArray<cufftComplex> m_mesh;                 //!< The particle density mesh
        GPUArray<cufftComplex> m_fourier_mesh;         //!< The fourier transformed mesh
        GPUArray<cufftComplex> m_fourier_mesh_G;       //!< Fourier transformed mesh times the influence function
        GPUArray<cufftComplex> m_fourier_mesh_force_x; //!< Force mesh in Fourier space, x component
        GPUArray<cufftComplex> m_fourier_mesh_force_y; //!< Force mesh in Fourier space, y component
        GPUArray<cufftComplex> m_fourier_mesh_force_z; //!< Force mesh in Fourier space, z component
        GPUArray<cufftComplex> m_force_mesh_x;      //!< The inverse-fourier transformed force mesh, x component
        GPUArray<cufftComplex> m_force_mesh_y;      //!< The inverse-fourier transformed force mesh, y component
        GPUArray<cufftComplex> m_force_mesh_z;      //!< The inverse-fourier transformed force mesh, z component
        GPUArray<Scalar4> m_force_mesh;             //!< Storage for force vectors

        GPUArray<Scalar4> m_particle_bins;         //!< Cell list for particle positions and modes
        GPUArray<unsigned int> m_n_cell;           //!< Number of particles per cell
        unsigned int m_cell_size;                  //!< Current max. number of particles per cell
        GPUFlags<unsigned int> m_cell_overflowed;  //!< Flag set to 1 if a cell overflows

        GPUFlags<Scalar> m_sum;                    //!< Sum over fourier mesh values
        GPUArray<Scalar> m_sum_partial;            //!< Partial sums over fourier mesh values
        GPUArray<Scalar> m_sum_virial_partial;     //!< Partial sums over virial mesh values
        GPUArray<Scalar> m_sum_virial;             //!< Final sum over virial mesh values
        unsigned int m_block_size;                 //!< Block size for fourier mesh reduction

        GPUArray<int4> m_bin_adj;                 //!< Particle bin adjacency list
        Index2D m_bin_adj_indexer;                 //!< Indexes elements in the bin adjacency list
        GPUArray<unsigned int> m_n_bin_adj;        //!< Number of adjacent bins for a cell
        
        // initialize the bin adjaceny indexer
        void initializeBinAdj();
    };

//! Define plus operator for complex data type (only need to compile by CommunicatorMesh base class)
inline cufftComplex operator + (cufftComplex& lhs, cufftComplex& rhs)
    {
    cufftComplex res;
    res.x = lhs.x + rhs.x;
    res.y = lhs.y + rhs.y;
    return res;
    }

//! Define plus operator for Scalar4 data type (only need to compile by CommunicatorMesh base class)
inline Scalar4 operator + (Scalar4& lhs, Scalar4& rhs)
    {
    Scalar4 res;
    res.x = lhs.x + rhs.x;
    res.y = lhs.y + rhs.y;
    res.z = lhs.z + rhs.z;
    res.w = lhs.w + rhs.w;
    return res;
    }


void export_OrderParameterMeshGPU();

#endif // ENABLE_CUDA
#endif // __ORDER_PARAMETER_MESH_GPU_H__
