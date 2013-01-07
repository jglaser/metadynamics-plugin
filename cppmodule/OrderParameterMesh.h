#include <hoomd/hoomd.h>

#ifndef __ORDER_PARAMETER_MESH_H__
#define __ORDER_PARAMETER_MESH_H__

#include "CollectiveVariable.h"
/*! Order parameter evaluated using the particle mesh method
 */
class OrderParameterMesh : public CollectiveVariable
    {
    public:
        //! Constructor
        OrderParameterMesh(boost::shared_ptr<SystemDefinition> sysdef,
                           const unsigned int nx,
                           const unsigned int ny,
                           const unsigned int nz,
                           const Scalar qstar,
                           const std::vector<Scalar> mode);
        virtual ~OrderParameterMesh();

        void computeForces(unsigned int timestep);

        Scalar getCurrentValue(unsigned int timestep);

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

    protected:
        Scalar3 m_mesh_size;                //!< The dimensions of a single cell along every coordinate
        uint3 m_mesh_points;                //!< Number of sub-divisions along one coordinate
        Index3D m_mesh_index;               //!< Indexer for the particle mesh 
        unsigned int m_radius;              //!< Radius of particle smearing (in units of mesh size)
        GPUArray<Scalar> m_mode;            //!< Per-type scalar multiplying density ("charges")
        GPUArray<Scalar> m_inf_f;           //!< Fourier representation of the influence function (real part)
        GPUArray<Scalar3> m_k;              //!< Mesh of k values
        Scalar m_qstarsq;                   //!< Short wave length cut-off squared for density harmonics
        bool m_is_first_step;               //!< True if we have not yet computed the influence function
        unsigned int m_cv_last_updated;     //!< Timestep of last update of collective variable
        Scalar m_E_self;                    //!< The self energy

        //! Helper function to setup the mesh indices
        void setupMesh();

        //! Helper function to setup FFT and allocate the mesh arrays
        virtual void initializeFFT();

        //! Compute the optimal influence function
        void computeInfluenceFunction();

        //! The CIC (cloud in cell) charge assignment function (Fourier transform, real part)
        Scalar assignCICFourier(Scalar k);

        //! Helper function to assign particle coordinates to mesh
        virtual void assignParticles();

        //! Helper function to update the mesh arrays
        virtual void updateMeshes();

        //! Helper function to interpolate the forces
        virtual void interpolateForces();

        //! Helper function to calculate value of collective variable
        virtual Scalar computeCV();

    private:
        kiss_fftnd_cfg m_kiss_fft;         //!< The FFT configuration
        kiss_fftnd_cfg m_kiss_ifft_x;      //!< Inverse FFT configuration, x component of force
        kiss_fftnd_cfg m_kiss_ifft_y;      //!< Inverse FFT configuration, y component of force
        kiss_fftnd_cfg m_kiss_ifft_z;      //!< Inverse FFT configuration, z component of force

        GPUArray<kiss_fft_cpx> m_mesh;             //!< The particle density mesh
        GPUArray<kiss_fft_cpx> m_fourier_mesh;     //!< The fourier transformed mesh
        GPUArray<kiss_fft_cpx> m_fourier_mesh_G;   //!< Fourier transformed mesh times the influence function
        GPUArray<kiss_fft_cpx> m_fourier_mesh_x;   //!< The fourier transformed force mesh, x component
        GPUArray<kiss_fft_cpx> m_fourier_mesh_y;   //!< The fourier transformed force mesh, y component
        GPUArray<kiss_fft_cpx> m_fourier_mesh_z;   //!< The fourier transformed force mesh, z component
        GPUArray<kiss_fft_cpx> m_force_mesh_x;     //!< The force mesh, x component
        GPUArray<kiss_fft_cpx> m_force_mesh_y;     //!< The force mesh, y component
        GPUArray<kiss_fft_cpx> m_force_mesh_z;     //!< The force mesh, z component

        boost::signals::connection m_boxchange_connection; //!< Connection to ParticleData box change signal

        std::string m_log_name;                    //!< Name of the log quantity
    };

void export_OrderParameterMesh();

#endif
