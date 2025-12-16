import jax
import os
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P


mesh = None
sharding = None


def initialize(pdims):
    '''this routine allows veris to run on either a single task with mutliple
    devices or on muttiple tasks with one device per task.
    
    JAX_COORDINATOR_ADDRESS is set in the job script only when running on mutliple tasks.
    jax.distributed.initialize() enables communication across tasks,
    but fails if only one task is used.
    
    the mesh and sharding are defined using the provided pdims, with axis names ('x', 'y').
    '''
    if 'JAX_COORDINATOR_ADDRESS' in os.environ:
        jax.distributed.initialize()

    global mesh, sharding
    mesh = jax.make_mesh(pdims, axis_names=('x','y'))
    sharding = NamedSharding(mesh, P('x','y'))
