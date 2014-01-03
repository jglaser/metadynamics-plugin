#include <hoomd/hoomd.h>

#include "OrderParameterMesh.h"

#ifndef __ORDER_PARAMETER_MESH_GPU_H__
#define __ORDER_PARAMETER_MESH_GPU_H__

#ifdef ENABLE_CUDA

//#define USE_HOST_DFFT

#ifdef ENABLE_MPI
#ifndef USE_HOST_DFFT
#include <dfft_cuda.h>
#else
#include <dfft_host.h>
#endif
#endif

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
                           const std::vector<Scalar> mode,
                           const std::vector<int3> zero_modes = std::vector<int3>());
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

        //! Compute maximum q vector
        virtual void computeQmax(unsigned int timestep);

    private:
        cufftHandle m_cufft_plan;          //!< The FFT plan
        bool m_local_fft;                  //!< True if we are only doing local FFTs (not distributed)

        #ifdef ENABLE_MPII
        typedef CommunicatorMeshGPU<cufftComplex, gpu_communicate_complex_mesh_map> CommunicatorMeshGPUComplex;
        boost::shared_ptr<CommunicatorMeshGPUComplex> m_gpu_mesh_comm; //!< Communicator for force mesh

        dfft_plan m_dfft_plan_forward;     //!< Forward distributed FFT
        dfft_plan m_dfft_plan_inverse;     //!< Forward distributed FFT
        #endif

        GPUArray<cufftComplex> m_mesh;                 //!< The particle density mesh
        GPUArray<cufftComplex> m_fourier_mesh;         //!< The fourier transformed mesh
        GPUArray<cufftComplex> m_fourier_mesh_G;       //!< Fourier transformed mesh times the influence function
        GPUArray<cufftComplex> m_inv_fourier_mesh;     //!< The inverse-fourier transformed force mesh

        uint3 m_n_ghost_bins;                      //!< Number of ghost bins in every direction
        Index2D m_bin_idx;                         //!< Total number of bins
        GPUArray<Scalar4> m_particle_bins;         //!< Cell list for particle positions and modes
        GPUArray<Scalar> m_mesh_scratch;           //!< Mesh with scratch space for density reduction
        Index2D m_scratch_idx;                     //!< Indexer for scratch space
        GPUArray<unsigned int> m_n_cell;           //!< Number of particles per cell
        unsigned int m_cell_size;                  //!< Current max. number of particles per cell
        GPUFlags<unsigned int> m_cell_overflowed;  //!< Flag set to 1 if a cell overflows

        GPUFlags<Scalar> m_sum;                    //!< Sum over fourier mesh values
        GPUArray<Scalar> m_sum_partial;            //!< Partial sums over fourier mesh values
        GPUArray<Scalar> m_sum_virial_partial;     //!< Partial sums over virial mesh values
        GPUArray<Scalar> m_sum_virial;             //!< Final sum over virial mesh values
        unsigned int m_block_size;                 //!< Block size for fourier mesh reduction

        GPUFlags<Scalar4> m_gpu_q_max;             //!< Return value for maximum Fourier mode reduction
        GPUArray<Scalar4> m_max_partial;           //!< Scratch space for reduction of maximum Fourier amplitude
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
