import jax
import jax.numpy as jnp
from veris.variables import variables
from veris.settings import settings


##### create state objects containing all variables and settings #####

class StateObject:
    def __init__(self, dict_in):
        # this copies the dictionary, with each key of the dictionary being
        # an attribute of the class
        for key, var in dict_in.items():
            setattr(self, key, var)

    def add(self, dict_in):
        # allows adding more dictionaries after init
        for key, var in dict_in.items():
            setattr(self, key, var)

    def tree_flatten(self):
        # returns the leaf values (the values that are contained in
        # the keys of self) and their keys
        leaf_values = []
        leaf_keys = []
        # self.__dict__ returns a dictionary containing the variables of self,
        # sorted() of a dictionary returns the sorted keys
        for key in sorted(self.__dict__):
            val = getattr(self, key)
            leaf_values.append(val)
            leaf_keys.append(key)
        return leaf_values, leaf_keys

    @classmethod
    def tree_unflatten(cls, leaf_keys, leaf_values):
        # creates an instance of StateObject based on leaf_keys and leaf_values
        obj = cls({})
        for key, val in zip(leaf_keys, leaf_values):
            setattr(obj, key, val)
        return obj

# only after registering the flatten and unflatten methods of the StateObject
# class with the jax pytree system, jax knows how to flatten/ unflatten a
# StateObject instance, which is necessary for using jax's automatic differentiation
# and parallelization features on such an instance (in general, it is necessary
# for using a jax-compiled function on such an instance)
jax.tree_util.register_pytree_node(
    StateObject,
    StateObject.tree_flatten,
    StateObject.tree_unflatten
)

# initialize all variables to the right size and fill them with 0
nx, ny = 1, 1
zeros2d = jnp.zeros((nx,ny))
for key in variables:
    variables[key] = zeros2d

# creating two state objects like this allows for accessing the variables and
# settings in a way that is compatible with the Veros syntax (like vs.uIce, sett.useEVP)
vs = StateObject(variables)
sett = StateObject(settings)


##### set intial conditions, grid and masks #####

def set_inits(vs, sett):
    ones2d = jnp.ones((nx, ny))
    
    vs.hIceMean    = ones2d * 1.3
    vs.hSnowMean   = ones2d * 0.1
    vs.Area        = ones2d * 0.9
    vs.TSurf       = ones2d * 273.0
    vs.SeaIceLoad  = ones2d * (sett.rhoIce * vs.hIceMean
                               + sett.rhoSnow * vs.hSnowMean)
    vs.wSpeed      = ones2d * 2
    vs.ocSalt      = ones2d * 29
    vs.theta       = ones2d * sett.celsius2K - 1.66
    vs.Qnet        = ones2d * 173.03212617345582
    vs.Qsw         = ones2d * 0
    vs.SWdown      = ones2d * 0
    vs.LWdown      = ones2d * 80
    vs.ATemp       = ones2d * 253
    vs.precip      = ones2d * 0
    
    vs.maskInC     = ones2d
    vs.maskInU     = ones2d
    vs.maskInV     = ones2d
    vs.iceMask     = ones2d
    vs.iceMaskU    = ones2d
    vs.iceMaskV    = ones2d

    return vs
    
vs = set_inits(vs, sett)
