import _metadynamics

# Next, since we are extending an updater, we need to bring in the base class updater and some other parts from 
# hoomd_script
from hoomd_script.integrate import _integrator
from hoomd_script.force import _force
from hoomd_script import util
from hoomd_script import globals
import hoomd

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

    def register_cv(self, cv_name, sigma):
        found = False;
        # Find the collective variable in the list of forces
        for f in globals.forces:
            if f.name == '_'+cv_name:
                found = True;
                break;

        if not found:
            print >> sys.stderr, "\n***Error! Cannot find collective variable '" + cv_name + "' in list of forces.\n"
            raise RuntimeError('Error initializing collective variable')

        self.cpp_integrator.registerCollectiveVariable(f.cpp_force, sigma)
