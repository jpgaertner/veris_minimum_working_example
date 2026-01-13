from time import perf_counter

t1 = perf_counter()

import jax
from jax.experimental import multihost_utils
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from jax import shard_map, numpy as jnp
from jax.lax import fori_loop
from functools import partial

import sys, os
sys.path.append(os.path.expanduser('~/veris_paper/veris_minimum_working_example'))
import initialize_mesh_sharding


##### set up sharding #####

# define dimensions of the processor mesh. this determines
# how devices are logically arranged for sharding
pdims = (1, 4)

# sets up distributed communication if running on multiple tasks
initialize_mesh_sharding.initialize(pdims)

mesh = initialize_mesh_sharding.mesh
sharding = initialize_mesh_sharding.sharding

# import veris and the state objects containing the variables and settings
from veris.area_mass import SeaIceMass, AreaWS
from veris.dynsolver import WindForcingXY, IceVelocities
from veris.dynamics_routines import SeaIceStrength
from veris.ocean_stress import OceanStressUV
from veris.advection import Advection
from veris.clean_up import clean_up_advection, ridging
from veris.fill_overlap import fill_overlap
from initialize_dyn_3072 import vs, sett

t2 = perf_counter()


def apply_sharding_to_array(arr):
    # assigns sharding metadata to a JAX array without moving data.
    # this does not split the array across different devices. the array is
    # split across devices when a jax.jit function is called on the sharded array
    return jax.device_put(arr, sharding)



# apply sharding to all fields in vs.
# the input of jax.tree.map has to be a jax array or pytree-compatible:
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

# add halos to all variables in vs
for attr in dir(vs):
    if not attr.startswith('__'): # ignore special attributes
        value = getattr(vs, attr)
        if not isinstance(value, type(vs.add)): # ignore methods
            setattr(vs, attr, fill_overlap(add_halo(value, halo_size)))

t4 = perf_counter()


##### define functions acting on the state object #####

# loop_body and test_model are minimal test functions to verify the
# JIT compilation and sharding. they are not used in the actual benchmark
@partial(jax.jit)
def loop_body(i, arg_body):
    vs.uIce, vs.vIce = arg_body
    
    vs.uIce = vs.uIce + vs.uOcean
    vs.uIce = fill_overlap(vs.uIce)

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

# dynamics-only veris (without thermodynamic growth/melting)
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
    vs.hIceMean = fill_overlap(vs.hIceMean)
    vs.hSnowMean = fill_overlap(vs.hSnowMean)
    vs.Area = fill_overlap(vs.Area)

    return vs


##### actual computation #####

# the computations in this funtion are done in parallel. calling a jax-compiled
# function on a sharded array triggers parallel execution across multiple devices
#vs = test_model(vs, sett)

# the first function call compiles it, exclude this from the benchmark
dump = dyn_model(vs, sett)

# force compilation to complete
jax.block_until_ready(dump)

t5 = perf_counter()

n_timesteps = 1440
for i in range(n_timesteps):
    vs = dyn_model(vs, sett)

# force all computations to complete before timing ends.
# JAX's lazy execution model defers the actual execution of JIT-compiled
# functions until their results are required or explicitly synchronized.
# without this synchronization, perf_counter() may return before the
# computation finishes, leading to an underestimation of runtime.
jax.block_until_ready(vs)

t6 = perf_counter()


##### finalize and gather results #####

# remove_halo: remove the halo from each partitioning region
# apply_halo_values:
# copy values from the halo to the actual array and remove halo.
# this is not something you want in your model. use this only to
# verify that the halo exchange is working
# multihost_utils.process_allgather:
# gather results of all partitioning regions and combine them into a single array

@partial(shard_map, mesh=mesh, in_specs=(P('x', 'y'), P()), out_specs=P('x', 'y'))
def apply_halo_values(arr, halo_size):
    # copies the values from the halo to the actual array,
    # then returns the array without the halo
    # ! this is just used for testing, not the actual simulations !

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

# remove halos and gather results for all variables in vs
for attr in dir(vs):
    if not attr.startswith('__'): # ignore special attributes
        value = getattr(vs, attr)
        if not isinstance(value, type(vs.add)): # ignore methods
            # remove halo from each shard (local operation)
            var_interior = remove_halo(value, halo_size)
            # allgather the interior
            var_global = multihost_utils.process_allgather(var_interior, tiled=True)
            setattr(vs, attr, var_global)

t7 = perf_counter()


##### print diagnostics #####

def print_separator(title=''):
    '''Print a formatted separator'''
    print(f'\n{"="*70}')
    print(f'  {title}')
    print(f'{"="*70}\n')

process_id = jax.process_index()
if process_id == 0:
    print_separator('configuration')
    for i, device in enumerate(jax.devices()):
        try:
            print(f'device {device.id}: {device.kind}')
        except:
            print(f'device {device.id}: {device.platform}')
    print('number of iterations in model run: ', n_timesteps)
    print('grid length: ', vs.hIceMean.shape[0])

print_separator(f'benchmark results on device {process_id}')
print('time for imports and jax initialization:', t2 - t1)
print('time for setting up sharding:', t3 - t2)
print('time for setting up halos:', t4 - t3)
print('time for compilation:', t5 - t4)
print('time for model run:', t6 - t5)
print('time for finalizing:', t7 - t6)

output_path = 'out_gpu/out/'
jnp.save(f'{output_path}file_{vs.hIceMean.shape[0]}_{pdims[0]}x{pdims[1]}_{n_timesteps}.npy', [vs.hIceMean, vs.hSnowMean, vs.Area, vs.uIce, vs.vIce, vs.uWind, vs.vWind, vs.uOcean, vs.vOcean])
