"""Commands to integrate the equation of motion using metadynamics.

This package implements a metadynamics integration mode using an
adaptive bias potential.

Metadynamics integration (integrate.mode_metadynamics) can be combined
with any standard integration methods, such as NVT, NVE etc. supported
by HOOMD-Blue.

In addition to integration methods, metadynamics also requires at least
one :mod:`collective variable <metadynamics.cv>` to be defined,
the values of which will be sampled to update the bias potential. The
forces generated from the bias potential are added to the particles during
the simulation.

This package supports well-tempered metadynamics with multiple collective
variables, on- and off-grid bias potentials, and saving of and restarting
from grid information. It is also possible to simply equilibrate the system
in the presence of a previously generated bias potential,
without updating the latter, to sample a histogram of values of the
collective variable (i.e. for error control).
"""
import hoomd
from hoomd import md

from hoomd.metadynamics import _metadynamics
from hoomd.metadynamics import cv

class mode_metadynamics(md.integrate._integrator):
    """Enables integration using metadynamics, a free energy technique.

    The command **integrate.mode_metadynamics** sets up MD integration with
    an arbitrary integration method (such as NVT), to continuously sample
    the collective variables and use their values to
    update the bias potential, from which forces are calculated.

    Some features of this package are loosely inspired by the
    PLUMED plugin for Metadynamics, http://www.plumed-code.org/.

    The metadynamics algorithm is reviewed in detail in
    [Barducci et al., Metadynamics, Wiley Interdiscipl. Rev.: Comput. Mol. Sci. 5, pp. 826-843 (2011)](http://dx.doi.org/10.1002/wcms.31)

    Explictly, the metadynamics biasing potential :math:`V(s,t)` at time :math:`t`
    takes the form

    .. math::

       V(\mathbf{s}, t) = \sum\limits_{t'=0, t_G, 2 t_G,\dots}^{t'<t}
                          W e^{-\frac{V[\mathbf{s}(\mathbf{r}(t')]}{\Delta T}}
                          \exp\left\{-\sum\limits_{i=1}^d
                          \frac{[s_i(\mathbf{r}) - s_i(\mathbf{r}(t'))]^2}{2\sigma_i^2}\right\},
    where

     * :math:`s` is the vector of collective variables,
     * :math:`t_G` is the stride (in time units); it should be chosen on the order
       of several :math:`\tau` , where :math:`\tau` is a typical internal relaxation
       time of the system,
     * :math:`W` is the height of Gaussians added during the simulation (in energy units),
     * :math:`\sigma_i` is the standard deviation of Gaussian added for collective
       variable :math:`i`.

    Before a metadynamics run, the collective variables need to be defined
    and the integration methods need to specified.

    While metadynamics in principle works independently of the integration
    method, it has thus far been tested with :class:`integrate.nvt` only.

    During a metadynamics run, the potential is updated every :math:`t_G/\Delta t`
    steps and forces derived from the potential are applied to the particles
    every step.

    The result of a metadynamics run is either a hills file (which contains
    the positions and heights of Gaussians that are added together to form
    the bias potential), or a bias potential evaluated on a grid.
    The negative of the bias potential can be used to calculate the free energy
    surface (FES).

    By default, **integrate.mode_metadynamics** uses the *well-tempered* variant
    of metadynamics, where a shift temperature :math:`\Delta T` is defined, which
    converges to a well-defined bias potential after a typical time for
    convergence that depends entirely on the system simulated and on the value
    of :math:`\Delta T`. The latter quantity should be chosen such that
    :math:`(\Delta T + T) k_B T` equals the typical height of free energy barriers
    in the system.

    By contrast, *standard* metadynamics does not converge
    to a limiting potential, and thus the free energy landscape is 'overfilled'.
    Standard metadynamics corresponds to :math:`\Delta T = \infty`.
    If the goal is to approximate standard metadynamics, large values (e.g.
    :math:`\Delta T = 100`) of the temperature shift can therefore be used.

    .. note::

        The collective variables need to be defined before the first
        call to **run()**. They cannot be changed after that (i.e. after **run()** has
        been called at least once), since the integrator maintains a history
        of the collective variables also in between multiple calls to the **run()**
        command. The only way to reset metadynamics integration is to use
        another **integrate.mode_metadynamics** instance instead of the original one.

    Two modes of operation are supported:

        1. Resummation of Gaussians every time step
        2. Evaluation of the bias potential on a grid

    In the first mode, the integration will slow down with increasing
    simulation time, as the number of Gaussians increases with time.

    In the second mode, a current grid of values of the
    collective variables is maintained and updated whenever a new
    Gaussian is deposited. This avoids the slowing down, and this mode
    is thus preferrable for long simulations. However, a
    reasonable set of grid points has to be chosen for accuracy (typically on
    the order of 200-500, depending on the collective variable and the system
    under study).

    It is possible to output the grid after the simulation, and to restart
    from the grid file. It is also possible to restart from the grid file
    and turn off the deposition of new Gaussians, e.g., to equilibrate
    the system in the bias potential landscape and measure the histogram of
    the collective variable, to correct for errors.

    .. note::

        Grid mode is automatically enabled when it is specified for all
        collective variables simultaneously. Otherwise, it has to be disabled
        for all collective variables at the same time.

    In the following, we give an example for using metadynamics in a diblock
    copolymer system.

    This sets up metadynamics with Gaussians of height :math:`W =1`
    (in energy units), which are deposited every :math:`t_G/\Delta t=5000`
    steps :math:`\Delta t = 0.005` in time units), with a well-tempered
    metadynamics temperature shift :math:`\Delta T = 7` (in temperature units).
    The collective variable is a lamellar order parameter.
    At the end of the simulation, the bias potential
    is saved into a file.

    .. code::

        all = group.all
        meta = metadynamics.integrate.mode_metadynamics
            dt=0.005, mode="well_tempered", W=1,stride=5000, deltaT=dT)

        # Use the NVT integration method
        integrate.nvt(group=all, T=1, tau=0.5)

        # set up a collective variable on a grid
        lamellar = metadynamics.cv.lamellar(
            sigma=0.05, mode=dict(A=1.0, B=-1.0),
            lattice_vectors=[(0,0,3)], phi=[0.0])
        lamellar.enable_grid(cv_min=-2.0, cv_max=2.0, num_points=400)

        # Run the metadynamics simulation for 10^5 time steps
        run(1e5)

        # dump bias potential
        meta.dump_grid("grid.dat")

    If the saved bias potential should be used to continue the simulation from,
    this can be accomplished by the following piece of code:

    .. code::

        meta = metadynamics.integrate.mode_metadynamics(dt=0.005, W=1)
        integrate.nvt(group=all, T=1, tau=0.5)

        # set up a collective variable on a grid
        lamellar = metadynamics.cv.lamellar(
            sigma=0.05, mode=dict(A=1.0, B=-1.0), lattice_vectors=[(0,0,3)], phi=[0.0])
        lamellar.enable_grid(cv_min=-2.0, cv_max=2.0, num_points=400)

        # restart from saved bias potential
        meta.restart_from_grid("grid.dat")
        run(1e5)

    :param dt:
        Each time step of the simulation run() will advance the real time of the system forward by \a dt (in time units)
    :param stride:
        Interval (number of time steps) between depositions of Gaussians
    :param mode:
        Metadynamics mode - "standard" (default) or "well_tempered"
    :param W:
        *(only in mode="standard" or "well_tempered")*
        Height of Gaussians (in energy units) deposited
    :param deltaT:
        *(only in mode="well_tempered")*
        Temperature shift (in temperature units) for well-tempered metadynamics
    :param T:
        *(only for adaptive Gaussians)*
        Temperature
    :param filename:
        *(optional)*
        Name of the log file to write hills information to
    :param overwrite:
        *(optional)*
        True if the hills file should be overwritten
    :param add_hills:
        *(optional)*
        True if Gaussians should be deposited during the simulation
    """

    def __init__(self, dt, stride, mode="standard", W=1.0,  deltaT=1.0, T=1.0, filename="", overwrite=False, add_hills=True):
        hoomd.util.print_status_line()

        # initialize base class
        md.integrate._integrator.__init__(self)

        if (mode == "standard"):
            cpp_mode = _metadynamics.IntegratorMetaDynamics.mode.standard
        elif (mode == "well_tempered"):
            cpp_mode = _metadynamics.IntegratorMetaDynamics.mode.well_tempered
        else:
            hoomd.context.msg.error("integrate.mode_metadynamics: Unsupported metadynamics mode.\n")
            raise RuntimeError('Error setting up Metadynamics.')

        # initialize the reflected c++ class
        self.cpp_integrator = _metadynamics.IntegratorMetaDynamics(
            hoomd.context.current.system_definition, dt, W, deltaT, T, stride, add_hills, filename, overwrite, cpp_mode)

        self.supports_methods = True

        hoomd.context.current.system.setIntegrator(self.cpp_integrator)

        self.cv_names = []

    def update_forces(self):
        """Registers the collective variables with the C++ integration class"""
        if self.cpp_integrator.isInitialized():
            notfound = False
            num_cv = 0
            for f in hoomd.context.current.forces:
                if isinstance(f, cv._collective_variable) and f.grid_set:
                    if f.name != self.cv_names[num_cv]:
                        notfound = True
                    num_cv += 1

            if (len(self.cv_names) != num_cv) or notfound:
                hoomd.context.msg.error(
                    "integrate.mode_metadynamics: Set of collective variables has changed since last run. This is unsupported.\n")
                raise RuntimeError('Error setting up Metadynamics.')

        # (re-) register collective variables with integrator
        self.cv_names = []
        self.cpp_integrator.removeAllVariables()

        for f in hoomd.context.current.forces:
            if isinstance(f, cv._collective_variable):

                # enable histograms if required
                if f.grid_set is True:
                    self.cpp_integrator.registerCollectiveVariable(
                        f.cpp_force, f.sigma, f.cv_min, f.cv_max, f.num_points)

                    self.cv_names.append(f.name)
                else:
                    if not f.umbrella:
                        hoomd.context.msg.warning(
                            "integrate.mode_metadynamics: Grid parameters not set. Ignoring CV " + f.name)

        if len(self.cv_names) == 0:
            hoomd.context.msg.warning(
                "integrate.mode_metadynamics: No collective variables defined. Continuing with simulation anyway.\n")

        if not self.cpp_integrator.isInitialized():
            self.cpp_integrator.setGrid(True)

        md.integrate._integrator.update_forces(self)

    def dump_grid(self, filename1, filename2="", period=0):
        """Dump information about the bias potential.

        If a grid has been previously defined for the collective variables,
        this method dumps the values of the bias potential evaluated on the grid
        points to a file, for later restart or analysis. This method can
        be used to dump the grid during the simulation or at any time before
        or after the simulation.

        :param filename1:
            First file to dump the grid to
        :param filename2:
            Second file to dump the grid to, if a period has been
            set. If this parameter is given, files are dumped
            in an alternating fashion.
        :param period:
            Number of timesteps between periodic dumps. If zero
            (default), file is written when the command is called.
        """
        hoomd.util.print_status_line()

        self.cpp_integrator.dumpGrid(filename1, filename2, int(period))

    def restart_from_grid(self, filename):
        """Restart from a previously saved grid file.

        This command may be used before starting the simulation with the
        run() command. Upon start of the simulation, the supplied grid file
        is then read in and used to initialize the bias potential.

        :param filename:
            The file to read, which has been previously generated by dump_grid
        """
        hoomd.util.print_status_line()

        self.cpp_integrator.restartFromGridFile(filename)

    def reset_histogram(self):
        """Reset the histogram.

        This command resets the histogram of values of the collective variable visited.
        """
        hoomd.util.print_status_line()

        self.cpp_integrator.resetHistogram()

    def set_params(self, add_hills=None, mode=None, stride=None, adaptive=None, sigma_g=None, multiple_walkers=None):
        """Set parameters of the integration.

        :param mode:
            The variant of metadynamics to be used
        :param add_hills:
            True if new Gaussians should be added during the simulation
        :param stride:
            The stride for bias potential updates
        :param adaptive:
            True if adaptive Gaussians should be used
        :param sigma_g:
            Estimated RMSD of particle positions for adapative Gaussian mode
        """
        hoomd.util.print_status_line()

        if add_hills is not None:
            self.cpp_integrator.setAddHills(add_hills)

        if mode is not None:
            if (mode == "standard"):
                cpp_mode = _metadynamics.IntegratorMetaDynamics.mode.standard
            elif (mode == "well_tempered"):
                cpp_mode = _metadynamics.IntegratorMetaDynamics.mode.well_tempered
            else:
                hoomd.context.msg.error("integrate.mode_metadynamics: Unsupported metadynamics mode.\n")
                raise RuntimeError('Error setting up Metadynamics.')

            self.cpp_integrator.setMode(cpp_mode)

        if stride is not None:
            self.cpp_integrator.setStride(int(stride))

        if adaptive is not None:
            self.cpp_integrator.setAdaptive(adaptive)

        if sigma_g is not None:
            self.cpp_integrator.setSigmaG(sigma_g)

        if multiple_walkers is not None:
            self.cpp_integrator.setMultipleWalkers(multiple_walkers)
