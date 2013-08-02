#include <hoomd/hoomd.h>

#ifndef __ORDER_PARAMETER_MESH_H__
#define __ORDER_PARAMETER_MESH_H__

#include "CollectiveVariable.h"

#include <boost/signals.hpp>
#include <boost/bind.hpp>

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
                           const std::vector<Scalar> mode,
                           const std::vector<int3> zero_modes = std::vector<int3>());
        virtual ~OrderParameterMesh();

        Scalar getCurrentValue(unsigned int timestep);

        /*! Returns the names of provided log quantities.
         */
        std::vector<std::string> getProvidedLogQuantities()
            {
            return m_log_names;
            }

        /*! Returns the value of a specific log quantity.
         * \param quantity The name of the quantity to return the value of
         * \param timestep The current value of the time step
         */
        Scalar getLogValue(const std::string& quantity, unsigned int timestep);

#ifdef ENABLE_MPI
        //! Set the communicator to use
        /*! \param comm MPI communication class
         */
        virtual void setCommunicator(boost::shared_ptr<Communicator> comm)
            {
            CollectiveVariable::setCommunicator(comm);

            m_ghost_layer_connection = comm->subscribeGhostLayer(boost::bind(&OrderParameterMesh::getGhostLayerWidth, this));
            }

        //! Get the ghost layer width
        /*! \returns the requested value of the ghost layer width
         */
        Scalar getGhostLayerWidth()
            {
            return m_ghost_layer_width;
            }
#endif

    protected:
        /*! Compute the biased forces for this collective variable.
            The force that is written to the force arrays must be
            multiplied by the bias factor.

            \param timestep The current value of the time step
         */
        void computeBiasForces(unsigned int timestep);

        Scalar3 m_mesh_size;                //!< The dimensions of a single cell along every coordinate
        uint3 m_mesh_points;                //!< Number of sub-divisions along one coordinate
        uint3 m_n_ghost_cells;              //!< Number of ghost cells along every axis
        Index3D m_mesh_index;               //!< Indexer for the particle mesh 
        Index3D m_force_mesh_index;         //!< Indexer for the force mesh
        unsigned int m_radius;              //!< Stencil radius (in units of mesh size)
        unsigned int m_n_inner_cells;       //!< Number of inner mesh points (without ghost cells)
        GPUArray<Scalar> m_mode;            //!< Per-type scalar multiplying density ("charges")
        GPUArray<Scalar> m_inf_f;           //!< Fourier representation of the influence function (real part)
        GPUArray<Scalar3> m_k;              //!< Mesh of k values
        Scalar m_qstarsq;                   //!< Short wave length cut-off squared for density harmonics
        bool m_is_first_step;               //!< True if we have not yet computed the influence function
        unsigned int m_cv_last_updated;     //!< Timestep of last update of collective variable
        bool m_box_changed;                 //!< True if box has changed since last compute
	    Scalar m_cv;			            //!< Current value of collective variable

        GPUArray<Scalar> m_virial_mesh;     //!< k-space mesh of virial tensor values

        unsigned int m_q_max_last_computed;        //!< Last time step at which q max was computed
        Scalar3 m_q_max;                           //!< Current wave vector with maximum amplitude
        Scalar m_sq_max;                           //!< Maximum structure factor

        GPUArray<int3> m_zero_modes;        //!< Fourier modes that should be zeroed

        //! Helper function to be called when box changes
        void setBoxChange()
            {
            m_box_changed = true;
            }

        //! Helper function to setup the mesh indices
        void setupMesh();

        //! Helper function to setup FFT and allocate the mesh arrays
        virtual void initializeFFT();

        //! Compute the optimal influence function
        virtual void computeInfluenceFunction();
 
        //! The TSC (triangular-shaped cloud) charge assignment function
        Scalar assignTSC(Scalar x);

        //! Derivative of the TSC (triangular-shaped cloud) charge assignment function
        Scalar assignTSCderiv(Scalar x);

        //! Helper function to assign particle coordinates to mesh
        virtual void assignParticles();

        //! Helper function to update the mesh arrays
        virtual void updateMeshes();

        //! Helper function to interpolate the forces
        virtual void interpolateForces();

        //! Helper function to calculate value of collective variable
        virtual Scalar computeCV();

        //! Helper function to compute the virial
        virtual void computeVirial();

        //! Helper function to compute q vector with maximum amplitude
        virtual void computeQmax(unsigned int timestep);

    private:
        kiss_fftnd_cfg m_kiss_fft;         //!< The FFT configuration
        kiss_fftnd_cfg m_kiss_ifft;        //!< Inverse FFT configuration

        #ifdef ENABLE_MPI
        boost::shared_ptr<DistributedKISSFFT> m_kiss_dfft;  //!< Distributed FFT for forward transform
        boost::shared_ptr<DistributedKISSFFT> m_kiss_idfft;  //!< Distributed FFT for inverse transform
        boost::shared_ptr<CommunicatorMesh<kiss_fft_cpx> > m_mesh_comm; //!< Communicator for force mesh
        #endif

        bool m_kiss_fft_initialized;               //!< True if a local KISS FFT has been set up

        GPUArray<kiss_fft_cpx> m_mesh;             //!< The particle density mesh
        GPUArray<kiss_fft_cpx> m_fourier_mesh;     //!< The fourier transformed mesh
        GPUArray<kiss_fft_cpx> m_fourier_mesh_G;   //!< Fourier transformed mesh times the influence function
        GPUArray<kiss_fft_cpx> m_inv_fourier_mesh; //!< The inverse-Fourier transformed mesh

        boost::signals::connection m_boxchange_connection; //!< Connection to ParticleData box change signal
        boost::signals::connection m_ghost_layer_connection; //!< Requests a ghost layer width

        std::vector<string> m_log_names;           //!< Name of the log quantity

        Scalar m_ghost_layer_width;                //!< The minimum width of the Communicator ghost layer

        //! Compute virial on mesh
        void computeVirialMesh();

        //! Helper function to compute number of ghost cells 
        uint3 computeNumGhostCells();
    
        //! Helper function to compute width of ghost particl layer
        void computeParticleGhostLayerWidth();
    };

//! Define plus operator for complex data type (needed by CommunicatorMesh)
inline kiss_fft_cpx operator + (kiss_fft_cpx& lhs, kiss_fft_cpx& rhs)
    {
    kiss_fft_cpx res;
    res.r = lhs.r + rhs.r;
    res.i = lhs.i + rhs.i;
    return res;
    }

void export_OrderParameterMesh();

#endif
