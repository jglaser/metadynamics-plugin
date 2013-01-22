#ifndef __DAFTGPU_H__
#define __DAFTGPU_H__

//! Class that implements the Daresbury Advanced Fourier Transform on the GPU, based on CUFFT
/* see http://www.hector.ac.uk/cse/distributedcse/reports/DL_POLY03/DL_POLY03_domain/index.html
 *
 * Forward transform is Sande-Tukey followed by local FFT, output in bit-reversed order
 * Inverse transform is FFT of bit-reversed input, followed by Cooley-Tukey
 */

#include <hoomd/hoomd.h>

class DAFTGPU
    {
    public:
        DAFTGPU(boost::shared_ptr<const ExecutionConfiguration> exec_conf,
             boost::shared_ptr<DomainDecomposition> decomposition,
             unsigned int nx,
             unsigned int ny,
             unsigned int nz);
        virtual ~DAFTGPU();

        void FFT3D(const GPUArray<cufftComplex>& in, const GPUArray<cufftComplex>& out, bool inverse);

    private:
        boost::shared_ptr<const ExecutionConfiguration> m_exec_conf;  //!< The execution configuration
        boost::shared_ptr<DomainDecomposition> m_decomposition; //!< Domain decomposition information
        unsigned int m_nx;                                      //!< Local mesh dimensions along x-axis
        unsigned int m_ny;                                      //!< Local mesh dimensions along y-axis
        unsigned int m_nz;                                      //!< Local mesh dimensions along z-axis
        GPUArray<cufftComplex> m_stage_buf;                     //!< Staging buffer for received data
        GPUArray<cufftComplex> m_combine_buf;                   //!< Buffer in which local and received data are combined
        GPUArray<cufftComplex> m_work_buf;                      //!< Buffer containing the input data to local FFT
        cufftHandle m_cufft_plan_x;    //!< FFT plan for x direction
        cufftHandle m_cufft_plan_y;    //!< FFT plan for y direction
        cufftHandle m_cufft_plan_z;    //!< FFT plan for x direction
    };

#endif
