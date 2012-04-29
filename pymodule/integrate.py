import _metadynamics

from hoomd_script.integrate import _integrator
from hoomd_script.force import _force
from hoomd_script import util
from hoomd_script import globals

import hoomd

import cv

class mode_metadynamics(_integrator):
    def __init__(self, dt, W, stride, deltaT):
        util.print_status_line();
    
        # initialize base class
        _integrator.__init__(self);
        
        # initialize the reflected c++ class
        self.cpp_integrator = _metadynamics.IntegratorMetaDynamics(globals.system_definition, dt, W, deltaT, stride);

        self.supports_methods = True;

        globals.system.setIntegrator(self.cpp_integrator);

        self.cv_list = [];

    def update_forces(self):
        self.cpp_integrator.removeAllVariables()

        have_cv = False
        for f in globals.forces:
            if isinstance(f, cv._collective_variable):
                self.cpp_integrator.registerCollectiveVariable(f.cpp_force, f.sigma)
                have_cv=True

        if not have_cv:
            globals.msg.warning("integrate.mode_metadynamics: No collective variables defined. Continuing with simulation anyway.\n")

        _integrator.update_forces(self)
