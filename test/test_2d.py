# Call with --nrank=1 and multiple MPI ranks
# bias.restart_0.dat and bias.dat_2 should be identical up to rounding errors

from hoomd import *
from hoomd import md

import math

import numpy as np

with context.initialize():
    snap = data.make_snapshot(N=1,box=data.boxdim(L=10**(1./3.)))
    system = init.read_snapshot(snap)

    # test mesh order parameter
    from hoomd import metadynamics

    meta = metadynamics.integrate.mode_metadynamics(dt=0.005, mode='well_tempered', stride=1,deltaT=1,W=1)
    md.integrate.nve(group=group.all())

    density = metadynamics.cv.density(group=group.all(),sigma=0.25)
    density.set_grid(cv_min=0,cv_max=1,num_points=100)

    aspect = metadynamics.cv.aspect_ratio(sigma=0.1,dir1=0,dir2=1)
    aspect.set_grid(cv_min=0,cv_max=2,num_points=200)

    meta.dump_grid('bias.dat',period=1)

    meta.set_params(multiple_walkers=True)
    run(1)

    system.box = system.box.scale(s=0.125**(1./3.))
    run(1)

with context.initialize():
    snap = data.make_snapshot(N=1,box=data.boxdim(L=10**(1./3.)))
    system = init.read_snapshot(snap)

    # test mesh order parameter
    from hoomd import metadynamics

    meta = metadynamics.integrate.mode_metadynamics(dt=0.005, mode='well_tempered', stride=1,deltaT=1,W=1)
    md.integrate.nve(group=group.all())

    density = metadynamics.cv.density(group=group.all(),sigma=0.25)
    density.set_grid(cv_min=0,cv_max=1,num_points=100)

    aspect = metadynamics.cv.aspect_ratio(sigma=0.1,dir1=0,dir2=1)
    aspect.set_grid(cv_min=0,cv_max=2,num_points=200)

    meta.restart_from_grid('bias.dat_1')
    meta.dump_grid('bias_restart.dat',period=1)
    meta.set_params(multiple_walkers=True)
    system.box = system.box.scale(s=0.125**(1./3.))
    run(1)
