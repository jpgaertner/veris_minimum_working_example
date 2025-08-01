from time import perf_counter

t1 = perf_counter()

import jax
from jax.experimental import multihost_utils
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from jax import numpy as jnp
from jax.lax import fori_loop
import jaxdecomp
from functools import partial
from time import perf_counter

# initialize the xla backend in a distributed context.
# this assigns the available devices (cpus/ gpus) to the processes
jax.distributed.initialize()

# import veris and the state objects containing the variables and settings
from veris.area_mass import SeaIceMass, AreaWS
from veris.dynsolver import WindForcingXY, IceVelocities
from veris.dynamics_routines import SeaIceStrength
from veris.ocean_stress import OceanStressUV
from veris.advection import Advection
from veris.clean_up import clean_up_advection, ridging
from veris.fill_overlap import fill_overlap, fill_overlap_uv
from initialize_dyn_1024 import vs, sett

t2 = perf_counter()


##### set up sharding #####

# define the dimensions of the processor mesh
pdims = (2, 2)

# create the processor mesh, where each device is identified by an index in the mesh
mesh = jax.make_mesh(pdims, axis_names=('x', 'y'))

# define the sharding. this specifies the data being partitioned
# across the devices in the 'x' and 'y' dimension of the processor mesh
sharding = NamedSharding(mesh, P('x', 'y'))

def apply_sharding_to_array(arr):
    # takes a jax array, and returns the same array but with sharding information added
    # to it, which allows jax to know how to partition the array across multiple devices.
    # this does not split the array across different devices. the array is split across
    # devices when a jax.jit function is called on the sharded array
    return jax.device_put(arr, sharding)

# apply sharding to all of the variables of the my_state object.
# the inputu of jax.tree.map has to be a jax array or pytree-compatible:
# for any function that takes a jax array as input, you can input any
# pytree-compatible instance, and jax will automatically flatten it,
# perform the function on the leaf values with are jax arrays, and then unflatten it.
# (this is why StateObject needs the flatten/ unflatten methods)
vs = jax.tree.map(apply_sharding_to_array, vs)

t3 = perf_counter()


##### set up the halo exchange ######

# define a halo size of 2 grid cells on each side of the array in both x and y directions
halo_size = ((2, 2), (2, 2))

@partial(shard_map, mesh=mesh, in_specs=(P('x', 'y'), P()), out_specs=P('x', 'y'))
def add_halo(arr, halo_size):
    # adds halo to the array with the given halo size (default halo values are 0)
    return jnp.pad(arr, halo_size)

@partial(shard_map, mesh=mesh, in_specs=(P('x', 'y'), P()), out_specs=P('x', 'y'))
def apply_halo_values(arr, halo_size):
    # copies the values from the halo to the actual array,
    # then returns the array without the halo

    # this syntax assumes that ol_left = ol_right, ol_top = ol_bottom
    olx = int(halo_size[1][0])
    oly = int(halo_size[0][0])

    # apply halos in x direction
    arr = arr.at[olx:2*olx, :].set(arr[:olx, :])
    arr = arr.at[-2*olx:-olx, :].set(arr[-olx:, :])

    # apply halos in y direction
    arr = arr.at[:, oly:2*oly].set(arr[:, :oly])
    arr = arr.at[:, -2*oly:-oly].set(arr[:, -oly:])

    # cut off the halo in the returned array
    return arr[olx:-olx,oly:-oly]

@partial(shard_map, mesh=mesh, in_specs=(P('x', 'y'), P()), out_specs=P('x', 'y'))
def remove_halo(arr, halo_size):
    # this syntax assumes that ol_left = ol_right, ol_top = ol_bottom
    olx = int(halo_size[1][0])
    oly = int(halo_size[0][0])

    # cut off the halo
    return arr[olx:-olx,oly:-oly]

# add halos to all variables in vs
for attr in dir(vs):
    if not attr.startswith('__'): # ignore special attributes
        value = getattr(vs, attr)
        if not isinstance(value, type(vs.add)): # ignore methods
            setattr(vs, attr, fill_overlap(sett, add_halo(value, halo_size)))

t4 = perf_counter()


##### define functions acting on the state object #####

@partial(jax.jit)
def loop_body(i, arg_body):
    vs.uIce, vs.vIce = arg_body
    
    vs.uIce = vs.uIce + vs.uOcean
    vs.uIce = fill_overlap(sett, vs.uIce)

    arg_body = vs.uIce, vs.vIce
    return arg_body

@partial(jax.jit, static_argnames=['sett'])
# in jax.jit, the control flow cannot depend on input variables, instead it
# has to be determined at compile time. with  static_argnames=['sett'], sett
# and its attributes (like useAdaptiveEVP) become part of the compiled function,
# allowing for the control flow to be static (= determined at compile time)
def test_model(vs, sett):
    if sett.useAdaptiveEVP:

        arg_body = vs.uIce, vs.vIce
        arg_body = fori_loop(0, 1, loop_body, arg_body)
        vs.uIce, vs.vIce = arg_body
    return vs

# veris without thermodynamics
@partial(jax.jit, static_argnames=['sett'])
def dyn_model(vs, sett):

    # calculate sea ice mass centered around c-, u-, and v-points
    vs.SeaIceMassC, vs.SeaIceMassU, vs.SeaIceMassV = SeaIceMass(vs, sett)

    # calculate sea ice cover fraction centered around u- and v-points
    vs.AreaW, vs.AreaS = AreaWS(vs, sett)

    # calculate surface forcing due to wind
    vs.WindForcingX, vs.WindForcingY = WindForcingXY(vs, sett)

    # calculate ice strength
    vs.SeaIceStrength = SeaIceStrength(vs, sett)

    # calculate ice velocities
    vs.uIce, vs.vIce, vs.sigma1, vs.sigma2, vs.sigma12 = IceVelocities(vs, sett)

    # calculate stresses on ocean surface
    vs.OceanStressU, vs.OceanStressV = OceanStressUV(vs, sett)

    # calculate change in sea ice fields due to advection
    vs.hIceMean, vs.hSnowMean, vs.Area = Advection(vs, sett)

    # correct overshoots and other pathological cases after advection
    (
        vs.hIceMean,
        vs.hSnowMean,
        vs.Area,
        vs.TSurf,
        vs.os_hIceMean,
        vs.os_hSnowMean,
    ) = clean_up_advection(vs, sett)

    # cut off ice cover fraction at 1 after advection
    vs.Area = ridging(vs, sett)

    # fill overlaps
    vs.hIceMean = fill_overlap(sett, vs.hIceMean)
    vs.hSnowMean = fill_overlap(sett, vs.hSnowMean)
    vs.Area = fill_overlap(sett, vs.Area)

    return vs


##### actual computation #####

# the computations in this funtion are done in parallel. calling a jax-compiled
# function on a sharded array triggers parallel execution across multiple devices
#vs = test_model(vs, sett)

# the first function call compiles it, exclude this from the benchmark
for _ in range(3):
    dyn_model(vs, sett)

t5 = perf_counter()

n_timesteps = 50
for i in range(n_timesteps):
    vs = dyn_model(vs, sett)

t6 = perf_counter()


##### finalize and gather results #####

# remove_halo: remove the halo from each partitioning region
# apply_halo_values:
# copy values from the halo to the actual array and remove halo.
# this is not something you want in your model. use this only to
# verify that the halo exchange is working
# multihost_utils.process_allgather:
# gather results of all partitioning regions and combine them into a single array

# remove halos and gather results for all variables in vs
for attr in dir(vs):
    if not attr.startswith('__'): # ignore special attributes
        value = getattr(vs, attr)
        if not isinstance(value, type(vs.add)): # ignore methods
            setattr(vs, attr, multihost_utils.process_allgather(
                              #apply_halo_values(value, halo_size), tiled=True
                              remove_halo(value, halo_size), tiled=True
                              )
                   )

t7 = perf_counter()

print('time for imports and jax initialization: ', t2 - t1)
print('time for setting up sharding: ', t3 - t2)
print('time for setting up halos: ', t4 - t3)
print('time for compilation: ', t5 - t4)
print('time for model run: ', t6 - t5)
print('time for finalizing: ', t7 - t6)

nproc = pdims[0]*pdims[1]

jnp.save(f'results/file_{vs.hIceMean.shape[0]}_{nproc}_{n_timesteps}.npy', [vs.hIceMean, vs.hSnowMean, vs.Area, vs.uIce, vs.vIce, vs.uWind, vs.vWind, vs.uOcean, vs.vOcean])
