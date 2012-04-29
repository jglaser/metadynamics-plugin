from hoomd_script import globals
from hoomd_script import util
from hoomd_script import data
from hoomd_script.force import _force

import _metadynamics
import hoomd

class lamellar(_force):

    def __init__(self, mode, lattice_vectors, generate_symmetries=False, name = ""):
        util.print_status_line()

        _force.__init__(self, name)

        if len(lattice_vectors) == 0:
                globals.msg.error("List of supplied latice vectors is empty.\n")
                raise RuntimeEror('Error creating collective variable.')
     
        
        if type(mode) != type(dict()):
                globals.msg.error("Mode amplitudes specified incorrectly.\n")
                raise RuntimeEror('Error creating collective variable.')

        cpp_mode = hoomd.std_vector_float()
        for i in range(0, globals.system_definition.getParticleData().getNTypes()):
            t = globals.system_definition.getParticleData().getNameByType(i)

            if t not in mode.keys():
                globals.msg.error("Missing mode amplitude for particle type " + t + ".\n")
                raise RuntimeEror('Error creating collective variable.')
            cpp_mode.append(mode[t])

        cpp_lattice_vectors = _metadynamics.std_vector_int3()
        for l in lattice_vectors:
            if len(l) != 3:
                globals.msg.error("List of input lattice vectors not a list of triples.\n")
                raise RuntimeEror('Error creating collective variable.')
            cpp_lattice_vectors.append(hoomd.make_int3(l[0], l[1], l[2]))

        if not globals.exec_conf.isCUDAEnabled():
            self.cpp_force = _metadynamics.LamellarOrderParameter(globals.system_definition, cpp_mode, cpp_lattice_vectors, generate_symmetries)
        else:
            self.cpp_force = _metadynamics.LamellarOrderParameterGPU(globals.system_definition, cpp_mode, cpp_lattice_vectors, generate_symmetries)

        globals.system.addCompute(self.cpp_force, self.force_name)
    
    def update_coeffs(self):
        pass
