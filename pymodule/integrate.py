import _metadynamics

from hoomd_script.integrate import _integrator
from hoomd_script.force import _force
from hoomd_script import util
from hoomd_script import globals

import hoomd

import cv

class mode_metadynamics(_integrator):
    def __init__(self, dt, W, stride, deltaT, filename="", overwrite=False, add_hills=True):
        util.print_status_line();
    
        # initialize base class
        _integrator.__init__(self);
        
        # initialize the reflected c++ class
        self.cpp_integrator = _metadynamics.IntegratorMetaDynamics(globals.system_definition, dt, W, deltaT, stride, add_hills, filename, overwrite);

        self.supports_methods = True;

        globals.system.setIntegrator(self.cpp_integrator);

        self.cv_names = [];

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

    def dump_grid(self, filename):
        util.print_status_line();

        self.cpp_integrator.dumpGrid(filename)

    def restart_from_grid(self, filename):
        util.print_status_line();

        self.cpp_integrator.restartFromGridFile(filename)

    def set_params(self, add_hills=False):
        util.print_status_line();
       
        self.cpp_integrator.setAddHills(add_hills)
