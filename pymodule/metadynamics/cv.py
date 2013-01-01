## \package metadynamics.cv
# \brief Defines the collective variables used for metadynamics integration

from hoomd_script import globals
from hoomd_script import util
from hoomd_script import data
from hoomd_script.force import _force

from hoomd_plugins.metadynamics import _metadynamics
import hoomd

## \internal
# \brief Base class for collective variables
#
# A collective_variable in python reflects a CollectiveVariable in C++.
# It is, in particular, a specialization of a force, since collective
# variables generate forces acting on the particles during the simulation.
class _collective_variable(_force):
    ## \internal
    # \brief Constructs a collective variable
    #
    # This mainly sets some parameters of the collective variable
    #
    # \param sigma Standard deviation of Gaussians added for this collective variable - only relevant in "well-tempered" or "standard" metadynamics
    # \param name Name of the collective variable
    def __init__(self, sigma, name=None):
        _force.__init__(self, name)

        self.sigma = sigma

        # default grid parameters
        self.cv_min = 0.0
        self.cv_max = 0.0
        self.num_points = 0
        
        self.use_grid = False

        self.ftm_min = 0.0
        self.ftm_max = 0.0

        self.ftm_parameters_set = False

    ## \var sigma
    # \internal

    ## \var cv_min
    # \internal

    ## \var cv_max
    # \internal

    ## \var num_points
    # \internal

    ## \var use_grid
    # \internal

    ## \var ftm_min
    # \internal

    ## \var ftm_max
    # \internal

    ## \var ftm_num_points
    # \internal

    ## Sets grid mode for this collective variable
    # \param cv_min Minimum of the collective variable (smallest grid value)
    # \param cv_max Maximum of the collective variable (largest grid value)
    # \param num_points Dimension of the grid for this collective variable 
    def enable_grid(self,cv_min, cv_max, num_points):
        util.print_status_line()

        self.cv_min = cv_min
        self.cv_max = cv_max
        self.num_points = num_points

        self.use_grid = True

    ## Sets parameters for the histogram of flux-tempered metadynamics
    # \param ftm_min Minimum of the collective variable (smallest grid value)
    # \param ftm_max Maximum of the collective variable (largest grid value)
    # \param num_points Dimension of the grid for this collective variable 
    def enable_histograms(self,ftm_min, ftm_max):
        util.print_status_line()

        self.ftm_min = ftm_min
        self.ftm_max = ftm_max

        self.ftm_parameters_set = True

    ## Set parameters for this collective variable
    # \param sigma The standard deviation
    def set_params(self, sigma=None):
        util.print_status_line()

        if sigma is not None:
            self.sigma = sigma

## \brief Lamellar order parameter as a collective variable to study phase transitions in block copolymer systems
#
# This collective variable is based on the Fourier modes of concentration
# or composition fluctuations.
#
# The value of the collective variable \f$ s \f$ is given by
# \f[ s = V^{-1} \sum_{i = 1}^n \sum_{j = 1}^N a(type_j) \cos(\mathbf{q}_i\mathbf{r_j} + \phi_i), \f]
# where
# - \f$n\f$ is the number of modes supplied
# - \f$N\f$ is the number of particles
# - \f$V\f$ is the system volume
# - \f$ \mathbf{q}_i = 2 \pi (\frac{n_{i,x}}{L_x}, \frac{n_{i,y}}{L_y}, \frac{n_{i,z}}{L_z}) \f$ is the 
# wave vector associated with mode \f$ i \f$,
# - \f$ \phi_i \f$ its phase shift,
# - \f$a(type)\f$ is the mode coefficient for a particle of type \f$type\f$.
#
# The force on particle i is calculated as
# \f$\vec f_i = - \frac{\partial V}{\partial s} \vec\nabla_i s\f$.
#
# <h2>Example:</h2>
#
# In a diblock copolymer melt constructed from monomers of types A and B, use
# \code
# metadynamics.cv.lamellar(sigma=.05,mode=dict(A=1.0, B=-1.0), lattice_vectors=[(0,0,3)],phi=[0.0])
# \endcode
# to define the order parameter for a Fourier mode along the (003) direction,
# with mode coefficients of +1 and -1 for A and B monomers. 
# The width of Gaussians deposited is sigma=0.05. The phase shift is zero.
#
# <h2> Logging:</h2>
# The log name of this collective variable used by the command \b analyze.log
# is \b cv_lamellar.
class lamellar(_collective_variable):
    ## Construct a lamellar order parameter
    # \param sigma Standard deviation of deposited Gaussians
    # \param mode Per-type list (dictionary) of mode coefficients
    # \param lattice_vectors List of reciprocal lattice vectors (Miller indices) for every mode
    # \param name Name given to this collective variable
    def __init__(self, sigma, mode, lattice_vectors, name=None):
        util.print_status_line()

        if name is not None:
            name = "_" + name
            suffix = name
        else:
            suffix = "" 

        _collective_variable.__init__(self, sigma, name)

        if len(lattice_vectors) == 0:
                globals.msg.error("cv.lamellar: List of supplied latice vectors is empty.\n")
                raise RuntimeEror('Error creating collective variable.')
     
        if type(mode) != type(dict()):
                globals.msg.error("cv.lamellar: Mode amplitudes specified incorrectly.\n")
                raise RuntimeEror('Error creating collective variable.')

        cpp_mode = hoomd.std_vector_float()
        for i in range(0, globals.system_definition.getParticleData().getNTypes()):
            t = globals.system_definition.getParticleData().getNameByType(i)

            if t not in mode.keys():
                globals.msg.error("cv.lamellar: Missing mode amplitude for particle type " + t + ".\n")
                raise RuntimeEror('Error creating collective variable.')
            cpp_mode.append(mode[t])

        cpp_lattice_vectors = _metadynamics.std_vector_int3()
        for l in lattice_vectors:
            if len(l) != 3:
                globals.msg.error("cv.lamellar: List of input lattice vectors not a list of triples.\n")
                raise RuntimeError('Error creating collective variable.')
            cpp_lattice_vectors.append(hoomd.make_int3(l[0], l[1], l[2]))

        if not globals.exec_conf.isCUDAEnabled():
            self.cpp_force = _metadynamics.LamellarOrderParameter(globals.system_definition, cpp_mode, cpp_lattice_vectors, suffix)
        else:
            self.cpp_force = _metadynamics.LamellarOrderParameterGPU(globals.system_definition, cpp_mode, cpp_lattice_vectors, suffix)

        globals.system.addCompute(self.cpp_force, self.force_name)

    ## \var cpp_force
    # \internal 

    ## \internal
    def update_coeffs(self):
        pass

## \brief Aspect ratio of the tetragonal simulation box as collective variable
#
class aspect_ratio(_collective_variable):
    ## Construct a lamellar order parameter
    # The aspect ratio is defined as the ratio between box lengths in 
    # direction 1 and 2
    #
    # \param dir1 Cartesian index of first direction 
    # \param dir2 Cartesian index of second direction
    # \param sigma Standard deviation of deposited Gaussians
    def __init__(self, sigma, dir1, dir2, name=""):
        util.print_status_line()

        _collective_variable.__init__(self, sigma, name)

        self.cpp_force = _metadynamics.AspectRatio(globals.system_definition, int(dir1), int(dir2))

        globals.system.addCompute(self.cpp_force, self.force_name)

    ## \var cpp_force
    # \internal 

    ## \internal
    def update_coeffs(self):
        pass

class mesh(_collective_variable):
    ## Construct a lamellar order parameter
    # \param sigma Standard deviation of deposited Gaussians
    # \param qstar Short-wavelength cutoff
    # \param mode Per-type list (dictionary) of mode coefficients
    # \param nx Number of mesh points along first axis
    # \param ny Number of mesh points along second axis
    # \param nz Number of mesh points along third axis
    # \param name Name given to this collective variable
    def __init__(self, sigma, qstar, mode, nx, ny=None, nz=None, name=None):
        util.print_status_line()

        if name is not None:
            name = "_" + name
            suffix = name
        else:
            suffix = "" 

        if ny is None:
            ny = nx

        if nz is None:
            nz = nx

        _collective_variable.__init__(self, sigma, name)

        if type(mode) != type(dict()):
                globals.msg.error("cv.mesh: Mode amplitudes specified incorrectly.\n")
                raise RuntimeEror('Error creating collective variable.')

        cpp_mode = hoomd.std_vector_float()
        for i in range(0, globals.system_definition.getParticleData().getNTypes()):
            t = globals.system_definition.getParticleData().getNameByType(i)

            if t not in mode.keys():
                globals.msg.error("cv.mesh: Missing mode amplitude for particle type " + t + ".\n")
                raise RuntimeEror('Error creating collective variable.')
            cpp_mode.append(mode[t])

#        if not globals.exec_conf.isCUDAEnabled():
        self.cpp_force = _metadynamics.OrderParameterMesh(globals.system_definition, nx,ny,nz,qstar, cpp_mode)
#        else:

        globals.system.addCompute(self.cpp_force, self.force_name)

    ## \var cpp_force
    # \internal 

    ## \internal
    def update_coeffs(self):
        pass


