## \package metadynamics.cv
# \brief Defines the collective variables used for metadynamics integration

from hoomd.metadynamics import  _metadynamics
import hoomd
from hoomd import _hoomd
from hoomd import md
from hoomd.md import nlist as nl
from hoomd.md import _md

## \internal
# \brief Base class for collective variables
#
# A collective_variable in python reflects a CollectiveVariable in C++.
# It is, in particular, a specialization of a force, since collective
# variables generate forces acting on the particles during the simulation.
class _collective_variable(md.force._force):
    ## \internal
    # \brief Constructs a collective variable
    #
    # This mainly sets some parameters of the collective variable
    #
    # \param sigma Standard deviation of Gaussians added for this collective variable - only relevant for "well-tempered" or "standard" metadynamics
    # \param name Name of the collective variable
    def __init__(self, sigma, name=None):
        # register as ForceCompute
        md.force._force.__init__(self, name)

        self.sigma = sigma

        # default grid parameters
        self.cv_min = 0.0
        self.cv_max = 0.0
        self.num_points = 0

        self.grid_set = False

        self.ftm_min = 0.0
        self.ftm_max = 0.0

        self.ftm_parameters_set = False

        self.umbrella = False
        self.reweight = False

    ## \var sigma
    # \internal

    ## \var cv_min
    # \internal

    ## \var cv_max
    # \internal

    ## \var num_points
    # \internal

    ## \var grid_set
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
    def set_grid(self,cv_min, cv_max, num_points):
        hoomd.util.print_status_line()

        self.cv_min = cv_min
        self.cv_max = cv_max
        self.num_points = int(num_points)

        self.grid_set = True

    ## Sets parameters for the histogram of flux-tempered metadynamics
    # \param ftm_min Minimum of the collective variable (smallest grid value)
    # \param ftm_max Maximum of the collective variable (largest grid value)
    # \param num_points Dimension of the grid for this collective variable 
    def enable_histograms(self,ftm_min, ftm_max):
        hoomd.util.print_status_line()

        self.ftm_min = ftm_min
        self.ftm_max = ftm_max

        self.ftm_parameters_set = True

    ## Set parameters for this collective variable
    # \param sigma The standard deviation
    # \param umbrella The umbrella mode, if this is an umbrella potential
    # \param kappa Umbrella potential stiffness
    # \param cv0 Umbrella potential minimum position
    # \param umbrella If True, do not add Gaussians in this collective variable
    # \param width_flat Width of flat region of umbrella potential
    # \param scale Prefactor multiplying umbrella potential
    # \param reweight True if CV should be included in reweighting
    def set_params(self, sigma=None, kappa=None, cv0=None, umbrella=None, width_flat=None, scale=None, reweight=None):
        hoomd.util.print_status_line()

        if sigma is not None:
            self.sigma = sigma

        if umbrella is not None:
            if umbrella=="no_umbrella":
                cpp_umbrella = self.cpp_force.umbrella.no_umbrella 
                self.reweight=False
                self.umbrella=False
            elif umbrella=="linear":
                cpp_umbrella = self.cpp_force.umbrella.linear
                self.reweight=True
                self.umbrella=True
            elif umbrella=="harmonic":
                cpp_umbrella = self.cpp_force.umbrella.harmonic
                self.reweight=True
                self.umbrella=True
            elif umbrella=="wall":
                cpp_umbrella = self.cpp_force.umbrella.wall
                self.reweight=True
                self.umbrella=True
            elif umbrella=="gaussian":
                cpp_umbrella = self.cpp_force.umbrella.gaussian
                self.reweight=True
                self.umbrella=True
            else:
                hoomd.context.msg.error("cv: Invalid umbrella mode specified.")
                raise RuntimeError("Error setting parameters of collective variable.");

            self.cpp_force.setUmbrella(cpp_umbrella)

        if kappa is not None:
            self.cpp_force.setKappa(kappa)

        if width_flat is not None:
            self.cpp_force.setWidthFlat(width_flat)

        if cv0 is not None:
            self.cpp_force.setMinimum(cv0)

        if scale is not None:
            self.cpp_force.setScale(scale)

        if reweight is not None:
            self.reweight = reweight

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
    def __init__(self, mode, lattice_vectors, name=None,sigma=1.0):
        hoomd.util.print_status_line()

        if name is not None:
            name = "_" + name
            suffix = name
        else:
            suffix = "" 

        _collective_variable.__init__(self, sigma, name)

        if len(lattice_vectors) == 0:
                hoomd.context.msg.error("cv.lamellar: List of supplied latice vectors is empty.\n")
                raise RuntimeEror('Error creating collective variable.')
     
        if type(mode) != type(dict()):
                hoomd.context.msg.error("cv.lamellar: Mode amplitudes specified incorrectly.\n")
                raise RuntimeEror('Error creating collective variable.')

        cpp_mode = _hoomd.std_vector_scalar()
        for i in range(0, hoomd.context.current.system_definition.getParticleData().getNTypes()):
            t = hoomd.context.current.system_definition.getParticleData().getNameByType(i)

            if t not in mode.keys():
                hoomd.context.msg.error("cv.lamellar: Missing mode amplitude for particle type " + t + ".\n")
                raise RuntimeEror('Error creating collective variable.')
            cpp_mode.append(mode[t])

        cpp_lattice_vectors = _metadynamics.std_vector_int3()
        for l in lattice_vectors:
            if len(l) != 3:
                hoomd.context.msg.error("cv.lamellar: List of input lattice vectors not a list of triples.\n")
                raise RuntimeError('Error creating collective variable.')
            cpp_lattice_vectors.append(hoomd.make_int3(l[0], l[1], l[2]))

        if not hoomd.context.exec_conf.isCUDAEnabled():
            self.cpp_force = _metadynamics.LamellarOrderParameter(hoomd.context.current.system_definition, cpp_mode, cpp_lattice_vectors, suffix)
        else:
            self.cpp_force = _metadynamics.LamellarOrderParameterGPU(hoomd.context.current.system_definition, cpp_mode, cpp_lattice_vectors, suffix)

        hoomd.context.current.system.addCompute(self.cpp_force, self.force_name)

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
    def __init__(self, dir1, dir2, name="",sigma=1.0):
        hoomd.util.print_status_line()

        _collective_variable.__init__(self, sigma, name)

        self.cpp_force = _metadynamics.AspectRatio(hoomd.context.current.system_definition, int(dir1), int(dir2))

        # add to System
        hoomd.context.current.system.addCompute(self.cpp_force, self.force_name)

    ## \var cpp_force
    # \internal 

    ## \internal
    def update_coeffs(self):
        pass

def _table_eval(r, rmin, rmax, V, F, width):
    dr = (rmax - rmin) / float(width-1);
    i = int(round((r - rmin)/dr))
    return (V[i], F[i])

class mesh(_collective_variable):
    ## Construct a lamellar order parameter
    # \param sigma Standard deviation of deposited Gaussians
    # \param mode Per-type list (dictionary) of mode coefficients
    # \param nx Number of mesh points along first axis
    # \param ny Number of mesh points along second axis
    # \param nz Number of mesh points along third axis
    # \param name Name given to this collective variable
    # \param zero_modes Indices of modes that should be zeroed
    def __init__(self, mode, nx, ny=None, nz=None, name=None,sigma=1.0,zero_modes=None):
        hoomd.util.print_status_line()

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
                hoomd.context.msg.error("cv.mesh: Mode amplitudes specified incorrectly.\n")
                raise RuntimeEror('Error creating collective variable.')

        cpp_mode = _hoomd.std_vector_scalar()
        for i in range(0, hoomd.context.current.system_definition.getParticleData().getNTypes()):
            t = hoomd.context.current.system_definition.getParticleData().getNameByType(i)

            if t not in mode.keys():
                hoomd.context.msg.error("cv.mesh: Missing mode amplitude for particle type " + t + ".\n")
                raise RuntimeEror('Error creating collective variable.')
            cpp_mode.append(mode[t])

        cpp_zero_modes = _metadynamics.std_vector_int3()
        if zero_modes is not None:
            for l in zero_modes:
                if len(l) != 3:
                    hoomd.context.msg.error("cv.lamellar: List of modes to zero not a list of triples.\n")
                    raise RuntimeError('Error creating collective variable.')
                cpp_zero_modes.append(hoomd.make_int3(l[0], l[1], l[2]))

        if not hoomd.context.exec_conf.isCUDAEnabled():
            self.cpp_force = _metadynamics.OrderParameterMesh(hoomd.context.current.system_definition, nx,ny,nz, cpp_mode, cpp_zero_modes)
        else:
            self.cpp_force = _metadynamics.OrderParameterMeshGPU(hoomd.context.current.system_definition, nx,ny,nz, cpp_mode, cpp_zero_modes)

        hoomd.context.current.system.addCompute(self.cpp_force, self.force_name)

    ## \var cpp_force
    # \internal

    # Set parameters for the collective variable
    # \param sq_pow Power of S(q), minus one, in the mode sum
    def set_params(self, sq_pow = None, use_table=None, **args):
        hoomd.util.print_status_line()

        if sq_pow is not None:
            self.cpp_force.setSqPower(sq_pow)

        if use_table is not None:
            self.cpp_force.setUseTable(use_table)

        # call base class method
        hoomd.util.quiet_status()
        _collective_variable.set_params(self,**args)
        hoomd.util.quiet_status()

    # Set the table to be used for the convolution kernel
    # \param func The function the returns the convolution kernel and its derivative
    # \param kmin Minimum k
    # \param kmax Maximum k
    # \param width Number of interpolation points
    # \param coeff Additional parameters to the function, as a dict (optional)
    def set_kernel(self, func, kmin, kmax, width, coeff = dict()):
        # allocate arrays to store kernel and derivative
        Ktable = _hoomd.std_vector_scalar()
        dKtable = _hoomd.std_vector_scalar()

        # calculate dr
        dk = (kmax - kmin) / float(width-1);

        # evaluate the function
        for i in range(0, width):
            k = kmin + dk * i;
            (K,dK) = func(k, kmin, kmax, **coeff)

            Ktable.append(K)
            dKtable.append(dK)

        # pass table to C++ collective variable
        self.cpp_force.setTable(Ktable, dKtable, kmin, kmax)

    ## \internal
    def update_coeffs(self):
        pass

## Potential Energy (Well-Tempered Ensemble)
#
# Use the potential energy as a collective variable
class potential_energy(_collective_variable):
    ## Construct a well-tempered ensemble
    # \param sigma Standard deviation of deposited Gaussians
    def __init__(self, sigma=1.0):
        hoomd.util.print_status_line()

        name = 'cv_potential_energy'

        _collective_variable.__init__(self, sigma, name)

        # disable as regular ForceCompute
        self.enabled = False

        self.cpp_force = _metadynamics.WellTemperedEnsemble(hoomd.context.current.system_definition, name)

        hoomd.context.current.system.addCompute(self.cpp_force, name)

    ## \var cpp_force
    # \internal 

    ## \internal
    def update_coeffs(self):
        pass

## Force Wraper
#
# Use an arbitrary force as collective variable
class wrap(_collective_variable):
    ## Construct the collective variable
    # \param force The handle to the force we are wrapping
    # \param sigma Standard deviation of deposited Gaussians
    def __init__(self, force, sigma=1.0):
        hoomd.util.print_status_line()

        if not isinstance(force, md.force._force):
            hoomd.context.msg.error("cv.wrap needs a md._force instance as argument.")

        name = 'cv_'+force.name

        _collective_variable.__init__(self, sigma, name)

        self.cpp_force = _metadynamics.CollectiveWrapper(hoomd.context.current.system_definition, force.cpp_force, name)

        if force.enabled or force.log:
            hoomd.context.current.system.addCompute(self.cpp_force, name)
        self.log = force.log

    def disable(self, log=False):
        self.disable(log)
        force.disable(log)

    def enable(self):
        self.enable()
        force.enable()

    ## \internal
    def update_coeffs(self):
        pass

## Steinhardt Ql
#
class steinhardt(_collective_variable):
    ## Construct the collective variable
    # \param r_cut Cut-off for neighbor search
    # \param r_on Onset of smoothing
    # \param lmax Maximum Ql to compute
    # \param Ql_ref List of reference Ql values (of length lmax+1)
    # \param nlist Neighbor list object
    # \param type Type of particles to compute order parameter for
    # \param name Name of Ql instance (optional)
    # \param sigma Standard deviation of deposited Gaussians
    def __init__(self, r_cut, r_on, lmax, Ql_ref, nlist, type, name=None, sigma=1.0):
        hoomd.util.print_status_line()

        suffix = ""
        if name is not None:
            suffix = "_" + name

        _collective_variable.__init__(self, sigma, name)

        self.type = type

        # subscribe to neighbor list rcut
        self.nlist = nlist
        self.r_cut = r_cut
        self.nlist.subscribe(lambda: self.get_rcut())
        self.nlist.update_rcut()

        if hoomd.context.exec_conf.isCUDAEnabled():
            self.nlist.cpp_nlist.setStorageMode(_md.NeighborList.storageMode.full);

        type_list = []
        for i in range(0, hoomd.context.current.system_definition.getParticleData().getNTypes()):
            type_list.append(hoomd.context.current.system_definition.getParticleData().getNameByType(i))

        if type not in type_list:
            hoomd.context.msg.error("cv.steinhardt: Invalid particle type.");
            raise RuntimeError('Error creating collective variable.')

        cpp_Ql_ref = _hoomd.std_vector_scalar()
        for Ql in list(Ql_ref):
            cpp_Ql_ref.append(Ql)

        self.cpp_force = _metadynamics.SteinhardtQl(hoomd.context.current.system_definition, float(r_cut), float(r_on), int(lmax), nlist.cpp_nlist, type_list.index(type), cpp_Ql_ref, suffix)
        hoomd.context.current.system.addCompute(self.cpp_force, self.force_name)

    def get_rcut(self):
        # go through the list of only the active particle types in the sim
        ntypes = hoomd.context.current.system_definition.getParticleData().getNTypes();
        type_list = [];
        for i in range(0,ntypes):
            type_list.append(hoomd.context.current.system_definition.getParticleData().getNameByType(i));

        my_typeid = type_list.index(self.type)
        # update the rcut by pair type
        r_cut_dict = nl.rcut();
        for i in range(0,ntypes):
            for j in range(i,ntypes):
                # interaction only for one particle type pair
                if i == my_typeid and j == my_typeid:
                    # get the r_cut value
                    r_cut_dict.set_pair(type_list[i],type_list[j], self.r_cut);
                else:
                    r_cut_dict.set_pair(type_list[i],type_list[j], -1.0);
        return r_cut_dict;

    ## \internal
    def update_coeffs(self):
        pass


