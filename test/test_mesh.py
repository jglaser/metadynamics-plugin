from hoomd import *
from hoomd import md

import math

import numpy as np
context.initialize()

init.create_lattice(unitcell=lattice.sc(a=1.0), n=[10,10,10]);

nl = md.nlist.cell()
wca = md.pair.lj(r_cut=2**(1./6.),nlist=nl)
wca.pair_coeff.set('A','A',sigma=1,epsilon=1)
wca.set_params(mode='shift')

md.integrate.mode_standard(dt=0.001)

# thermalize
bd = md.integrate.brownian(group = group.all(),kT=1.0,seed=123)
bd.set_gamma('A',100)
run(1e4)
bd.disable()

nve = md.integrate.nve(group = group.all())

# test mesh order parameter
from hoomd import metadynamics
mesh = metadynamics.cv.mesh(nx=32,mode={'A': 1})
cv0 = 0.025
mesh.set_params(umbrella='harmonic',cv0=cv0,kappa=10000/cv0**2)

# energy (= kinetic energy + potential energy + umbrella energy) should be conserved with 4-5 sigfigs in single precision
# & cv_mesh should be close to target value (cv0)

log = analyze.log(quantities=['kinetic_energy','potential_energy','umbrella_energy_mesh','cv_mesh'],filename='test.log',period=100,overwrite=True)

run(1e5)
