## \package metadynamics.integrate
# \brief Commands to integrate the equation of motion using metadynamics
#
# This package implements a metadynamics integration mode using an 
# adaptive bias potential. Metadynamics is described in detail in
# Barducci et al., Metadynamics, Wiley Interdiscipl. Rev.: Comput. Mol. Sci. 5, pp. 826-843 (2011).
#
# Metadynamics integration (\b integrate.mode_metadynamics) can be combined
# with any standard integration methods, such as NVT, NVE etc. supported
# by HOOMD-Blue.
#
# In addition to integration methods, metadynamics also requires at least
# one collective variable (\b cv) to be defined, the values of which
# will be sampled to updateei the biasing potential. The forces from
# the bias potential are added to the particles during the simulation.
#
# Some of the features of this package are loosely inspired by the
# PLUMED plugin for Metadynamics, http://www.plumed-code.org/.
#
# This package supports well-tempered metadynamics with multiple collective
# variables, on- and off-grid bias potentials, and saving of and restarting
# from grid information. It is also possible to simply equilibrate the system
# in the presence of a previously generated bias potential,
# without updating the latter, to sample a histogram of values of the
# collective variable (i.e. for error control)
#
# In the following, we give an example for using this package.
# This sets up metadynamics with Gaussians of height unity (in energy units),
# which are deposited every 5000 steps, and a well-tempered metadynamics
# temperature shift of 7 (in temperature units). The collective variable is a
# lamellar order parameter. At the end of the simulation, the bias potential
# is saved into a file.
#
# \code
# all = group.all
# meta = metadynamics.integrate.mode_metadynamics(dt=0.005, W=1,stride=5000, deltaT=dT)
# integrate.nvt(group=all, T=1, tau=0.5)
# # set up a collective variable on a grid
# lamellar = metadynamics.cv.lamellar(sigma=0.05, mode=dict(A=1.0, B=-1.0), lattice_vectors=[(0,0,3)], phi=[0.0])
# lamellar.enable_grid(cv_min=-2.0, cv_max=2.0, num_points=400)
# # Run the metadynamics simulation for 10^5 time steps
# run(1e5)
# # dump bias potential
# meta.dump_grid("grid.dat")
# \endcode
#
# If the saved bias potential should be used to continue the simulation from,
# this can be accomplished by
# \code
# meta = metadynamics.integrate.mode_metadynamics(dt=0.005, W=1,
# integrate.nvt(group=all, T=1, tau=0.5)
# #set up a collective variable on a grid
# lamellar = metadynamics.cv.lamellar(sigma=0.05, mode=dict(A=1.0, B=-1.0), lattice_vectors=[(0,0,3)], phi=[0.0])
# lamellar.enable_grid(cv_min=-2.0, cv_max=2.0, num_points=400)
# # restart from saved bias potential
# meta.restart_from_grid("grid.dat")
# run(1e5)
# \endcode
#

import _metadynamics

from hoomd_script.integrate import _integrator
from hoomd_script.force import _force
from hoomd_script import util
from hoomd_script import globals

import hoomd

import cv

## Enables integration using metadynamics
#
# The command integrate.mode_metadynamics sets up MD integration which
# continuously samples the collective variables and uses their values to
# update the bias potential. Collective variables have to be defined
# for metadynamics to work. Also, integration methods need to be defined
# in the same way as for integrate.mode_standard.
#
# Currently, the only collective variable available is 
# - cv.lamellar
#
# While metadynamics in principle works independently of the integration
# method, it has thus far been tested with
# - integrate.nvt
# only.
#
# By default, integrate.mode_metadynamics uses the well-tempered variant
# of metadynamics, which uses a shift temperature deltaT and
# converges to a well-defined bias potential after a typical time for
# convergence that depends entirely on the system simulated and on the value
# of deltaT. By contrast, standard metadynamics does not converge to a limiting
# potential, and thus the free energy landscape is 'overfilled'.
# Standard metadynamics corresponds to deltaT=\f$\infty\f$.
# To approximate standard metadynamics, very large values of the temperature
# shift can therefore be used. 
#
# \note The collective variables need to be defined before the first 
# call to run(). They cannot be changed after that (i.e. after run() has
# been called at least once), since the integrator maintains a history
# of the collective variables also in between multiple calls to the run()
# command. The only way to reset metadynamics integration is to use
# another integrate.mode_metadynamics instance instead of the original one.
#
# Two modes of operation are supported:
# 1) Resummation of Gaussians every time step
# 2) Evaluation of the bias potential on a grid
# 
# In the first mode, the integration will slow down with increasing
# simulation time, as the number of Gaussians increase.
# In the second mode, a current grid of values of the
# collective variables is maintained and updated whenever a new 
# Gaussian is deposited. This avoids the slowing down, and the second mode
# is thus preferrable for long simulations. In this mode, however, a
# reasonable of grid points has to be chosen for accuracy (typically on the 
# order of 200-500, depending on the collective variable and the system
# under study).
#
# It is possible to output the grid after the simulation, and to restart
# from the grid file. It is also possible to restart from the grid file
# and turn off the deposition of new Gaussians, e.g. to equilibrate
# the system in the bias potential landscape and measure the histogram of
# the collective variable, to correct for errors.
#
# \note Grid mode has to be specified for all collective variables
# simultaneously, or for none.
class mode_metadynamics(_integrator):
    ## Specifies the metadynamics integration mode
    # \param dt Each time step of the simulation run() will advance the real time of the system forward by \a dt (in time units) 
    # \param W Height of Gaussians (in energy units) deposited 
    # \param stride Interval (number of time steps) between depositions of Gaussians
    # \param deltaT Temperature shift (in temperature units) for well-tempered metadynamics
    # \param filename (optional) Name of the log file to write hills information to
    # \param overwrite (optional) True if the hills file should be overwritten
    # \param add_hills (optional) True if Gaussians should be deposited during the simulation
    def __init__(self, dt, W, stride, deltaT, filename="", overwrite=False, add_hills=True):
        util.print_status_line();
    
        # initialize base class
        _integrator.__init__(self);
        
        # initialize the reflected c++ class
        self.cpp_integrator = _metadynamics.IntegratorMetaDynamics(globals.system_definition, dt, W, deltaT, stride, add_hills, filename, overwrite);

        self.supports_methods = True;

        globals.system.setIntegrator(self.cpp_integrator);

        self.cv_names = [];

    ## \internal
    # \brief Registers the collective variables with the C++ integration class
    def update_forces(self):
        if self.cpp_integrator.isInitialized():
            notfound = False;
            num_cv = 0
            for f in globals.forces:
                if f.enabled and isinstance(f, cv._collective_variable):
                    if f.name not in self.cv_names:
                        notfound = True
                    num_cv += 1;

            if (len(self.cv_names) != num_cv) or notfound:
                globals.msg.error("integrate.mode_metadynamics: Set of collective variables has changed since last run. This is unsupported.\n")
                raise RuntimeError('Error setting up Metadynamics.');
        else:
            self.cpp_integrator.removeAllVariables()

            use_grid = False;
            for f in globals.forces:
                if f.enabled and isinstance(f, cv._collective_variable):
                    self.cpp_integrator.registerCollectiveVariable(f.cpp_force, f.sigma, f.cv_min, f.cv_max, f.num_points)

                    if f.use_grid is True:
                        if len(self.cv_names) == 0:
                            use_grid = True
                        else:
                            if use_grid is False:
                                globals.msg.error("integrate.mode_metadynamics: Not all collective variables have been set up for grid mode.\n")
                                raise RuntimeError('Error setting up Metadynamics.');
                                
                    self.cv_names.append(f.name)

            if len(self.cv_names) == 0:
                globals.msg.warning("integrate.mode_metadynamics: No collective variables defined. Continuing with simulation anyway.\n")

            self.cpp_integrator.setGrid(use_grid)

        _integrator.update_forces(self)

    ## Dump information about the bias potential
    # If a grid has been previously defined for the collective variables,
    # this method dumps the values of the bias potential evaluated on the grid
    # points to a file, for later restart or analysis. This method can
    # be used after the simulation has been run.
    #
    # \param filename The file to dump the information to
    def dump_grid(self, filename):
        util.print_status_line();

        self.cpp_integrator.dumpGrid(filename)

    ## Restart from a previously saved grid file
    # This command may be used before starting the simulation with the 
    # run() command. Upon start of the simulation, the supplied grid file
    # is then read in and used to initialize the bias potential.
    # 
    # \param filename The file to read, which has been previously generated by dump_grid
    def restart_from_grid(self, filename):
        util.print_status_line();

        self.cpp_integrator.restartFromGridFile(filename)

    ## Set parameters of the integration
    # \param add_hills True if new Gaussians should be added during the simulation
    def set_params(self, add_hills=True):
        util.print_status_line();
       
        self.cpp_integrator.setAddHills(add_hills)

