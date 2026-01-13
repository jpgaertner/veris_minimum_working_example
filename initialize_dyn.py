import jax
import jax.numpy as jnp
from veris.variables import variables
from veris.settings import settings


grid_len = 1024
nx, ny = grid_len, grid_len

use_circular_overlap = False
'''use this for running veris in a jupyter notebook
(run_dyn.ipynb). if you use this, you also have to
set use_sharding = False in veris.settings'''
if use_circular_overlap:
    olx, oly = 2, 2
else:
    olx, oly = 0, 0

accuracy = 'float64'


##### create state objects containing all variables and settings #####

class StateObject:
    def __init__(self, dict_in):
        # copies the dictionary dict_int and creates an instance of StateObject,
        # with each key of the dictionary being an attribute of the class
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

# register StateObject as a JAX pytree node so it can be used with jax.jit and
# other transformation that require tree flattening. This allows a StateObject
# instance to be passed into compiled functions and manipulated as a single unit
jax.tree_util.register_pytree_node(
    StateObject,
    StateObject.tree_flatten,
    StateObject.tree_unflatten
)

# initialize all variables to the right size and fill them with 0
zeros2d = jnp.zeros((grid_len+2*olx, grid_len+2*oly), dtype=accuracy)
for key in variables:
    variables[key] = zeros2d

# create state objects for variables and settings, enabling access via dot
# notation (e.g., vs.uIce, sett.useEVP); used in Veros-style syntax
vs = StateObject(variables)
sett = StateObject(settings)


##### create wind and ocean surface currents (and initial
    # sea ice thickness) similar to the Mehlmann 21 benchmark #####

Lx=512e3
Ly=Lx
dx = Lx/(nx-1)
dy = dx

f0=1.4e-4
gravity=9.81
Ho=1000

x = (jnp.arange(nx,dtype = accuracy)+0.5)*dx;
y = (jnp.arange(ny,dtype = accuracy)+0.5)*dy;
xx,yy = jnp.meshgrid(x,y);
xu,yu = jnp.meshgrid(x-.5*dx,y);
xv,yv = jnp.meshgrid(x,y-.5*dy);

# Flat bottom at z=-Ho
h=-Ho*jnp.ones((ny,nx),dtype = accuracy);
# channnel walls
h = h.at[:,-1].set(0)
h = h.at[-1,:].set(0)

variableWindField = True
shiftWindField = True
if variableWindField:
# variable wind field
    period = 16 # days
    if shiftWindField: period = 4 # days
    writeFreq = 3 # hours
    t = jnp.arange(0,period*24/writeFreq)/(24./writeFreq) # time in days
    vmax = 15.0; # maximale windgeschwindigkeit in m/s

    if shiftWindField:
        t = t + 4
        mx = Lx*.1 + Lx*.1*t
        my = Ly*.1 + Ly*.1*t
    else:
        tP=jnp.mod(t,period/2)
        tP[t>=period/2.]=period/2.-tP[t>=period/2.]
        tP = tP/(0.5*period)
        oLx=150.e3
        oLy=oLx
        mx = -oLx+(2*oLx+Lx)*tP
        my = -oLy+(2*oLy+Ly)*tP

    alpha0= jnp.pi/2. - jnp.pi/2./5. # 90 degrees is without convergence or divergence
    alpha = jnp.pi/2. - jnp.pi/2./5.*jnp.maximum(jnp.sign(jnp.roll(mx,-1)-mx),0.) \
            -jnp.pi/2./10.*jnp.maximum(jnp.sign(mx-jnp.roll(mx,-1)),0.)

    uwind = jnp.zeros((t.shape[0],xx.shape[0],xx.shape[1]))
    vwind = jnp.zeros((t.shape[0],yy.shape[0],yy.shape[1]))
    for k,myt in enumerate(t):
        wx =  jnp.cos(alpha[k])*(xx-mx[k]) + jnp.sin(alpha[k])*(yy-my[k])
        wy = -jnp.sin(alpha[k])*(xx-mx[k]) + jnp.cos(alpha[k])*(yy-my[k])
        r = jnp.sqrt((mx[k]-xx)*(mx[k]-xx)+(my[k]-yy)*(my[k]-yy))
        s = 1.0/50.e3*jnp.exp(-r/100.e3)
        if shiftWindField:
            w = jnp.tanh(myt*(8.0-myt)/2.)
        else:
            if myt<8:
                w = jnp.tanh(myt*(period/2.-myt)/2.)
            elif myt>=8 and myt<16:
                w = -jnp.tanh((myt-period/2.)*(period-myt)/2.)
        #    w = np.sin(2.0*np.pi*myt/period)

        # reset scaling factor w to one
        w = 1.
        uwind = uwind.at[k,:,:].set(-wx*s*w*vmax)
        vwind = vwind.at[k,:,:].set(-wy*s*w*vmax)

        spd=jnp.sqrt(uwind[k,:,:]**2+vwind[k,:,:]**2)
        div=uwind[k,1:-1,2:]-uwind[k,1:-1,:-2] \
             +vwind[k,2:,1:-1]-vwind[k,:-2,1:-1]

# ocean surface velocity
uo = +0.01*(2*yy-Ly)/Ly
vo = -0.01*(2*xx-Lx)/Lx

# initial ice thickness
hice = 0.3 + 0.005*jnp.sin(500*xx) + 0.005*jnp.sin(500*yy)
# symmetrize
hices = 0.5*(hice + hice.transpose())
# initial ice thickness for comparison with Mehlmann 21
hice = 0.3 + 0.005*(jnp.sin(60./1000.e3*xx) + jnp.sin(30./1000.e3*yy))
# symmetrize
hices = 0.5*(hice + hice.transpose())


##### set intial conditions, grid, masks, and local settings #####

ones2d = jnp.ones((grid_len+2*olx, grid_len+2*oly), dtype=accuracy)

def fill_overlap_loc(A):
    # fill halo regions using circular overlap
    if use_circular_overlap:
        A = A.at[:2, :].set(A[-4:-2, :])
        A = A.at[-2:, :].set(A[2:4, :])
        A = A.at[:, :2].set(A[:, -4:-2])
        A = A.at[:, -2:].set(A[:, 2:4])

    return A

def createfrom(inner_field):
    if use_circular_overlap:
        # this returns a field that is 2 grid cells in each direction larger than
        # inner_field, with these additional grid cells filled as circular overlaps
        field = ones2d
    
        field = field.at[2:-2,2:-2].set(inner_field)
        field = fill_overlap_loc(field)
        return field
    else:
        return inner_field

def set_inits(vs):
    # set initial conditions, grid and masks. this function
    # populates the vs object with all required fields

    #hIceMean  = hice
    hIceMean  = ones2d * 0.3
    hSnowMean = ones2d * 0.0
    Area      = ones2d * 1.0
    TSurf     = ones2d * 273.0
    
    uWind  = createfrom(uwind[15,:,:])
    vWind  = createfrom(vwind[15,:,:])
    
    maskInC = ones2d * 1
    maskInC = maskInC.at[-1-olx,:].set(0)
    maskInC = maskInC.at[:,-1-oly].set(0)
    maskInU = maskInC * jnp.roll(maskInC,1+2*olx,axis=1)
    maskInV = maskInC * jnp.roll(maskInC,1+2*oly,axis=0)

    maskInC = fill_overlap_loc(maskInC)
    maskInU = fill_overlap_loc(maskInU)
    maskInV = fill_overlap_loc(maskInV)
    
    uOcean = createfrom(uo) * maskInU
    vOcean = createfrom(vo) * maskInV

    R_low = createfrom(h)

    fcormin = 1.4604e-4
    fcormax = 1.4596e-4 + 8.0e-8 * grid_len
    y = jnp.linspace(fcormin, fcormax, grid_len)
    y = y[:,jnp.newaxis] * jnp.ones((grid_len,grid_len))
    fCori = createfrom(y)
    #fCori = createfrom(1.46e-4)
    #fCori = ones2d * 0

    iceMask, iceMaskU, iceMaskV  = maskInC, maskInU, maskInV

    deltaX = ones2d * dx
    dxC, dyC, dxG, dyG, dxU, dyU, dxV, dyV = [deltaX for _ in range(8)]
    recip_dxC, recip_dyC, recip_dxG, recip_dyG, \
    recip_dxU, recip_dyU, recip_dxV, recip_dyV = [1 / deltaX for _ in range(8)]

    rA  = dxU * dyV
    rAz = dxV * dyU
    rAu = dxC * dyG
    rAv = dxG * dyC
    recip_rA = 1 / rA
    recip_rAz = 1 / rAz
    recip_rAu = 1 / rAu
    recip_rAv = 1 / rAv

    fields = [
        hIceMean, hSnowMean, Area, TSurf, uOcean, vOcean, uWind, vWind, R_low,
        maskInC, maskInU, maskInV, iceMask, iceMaskU, iceMaskV, fCori,
        dxC, dyC, dxG, dyG, dxU, dyU, dxV, dyV,
        recip_dxC, recip_dyC, recip_dxG, recip_dyG, recip_dxU, recip_dyU, recip_dxV, recip_dyV,
        rA, rAz, rAu, rAv, recip_rA, recip_rAz, recip_rAu, recip_rAv
    ]

    fields = jnp.array(fields).transpose(0,2,1)

    (vs.hIceMean, vs.hSnowMean, vs.Area, vs.TSurf, vs.uOcean, vs.vOcean, vs.uWind, vs.vWind, vs.R_low,
    vs.maskInC, vs.maskInU, vs.maskInV, vs.iceMask, vs.iceMaskU, vs.iceMaskV, vs.fCori,
    vs.dxC, vs.dyC, vs.dxG, vs.dyG, vs.dxU, vs.dyU, vs.dxV, vs.dyV,
    vs.recip_dxC, vs.recip_dyC, vs.recip_dxG, vs.recip_dyG, vs.recip_dxU, vs.recip_dyU, vs.recip_dxV, vs.recip_dyV,
    vs.rA, vs.rAz, vs.rAu, vs.rAv, vs.recip_rA, vs.recip_rAz, vs.recip_rAu, vs.recip_rAv
    ) = fields

    return vs

# set initial conditions, grid and masks
vs = set_inits(vs)

# use these settings for the benchmark
deltat = 600
''' defaults in the mitgcm benchmark:
deltaX = ones2d * 8000
'evpAlpha'             : 123456.7,
'evpBeta'              : 123456.7,
'''
input_settings = {
            'deltatTherm'          : deltat,
            'recip_deltatTherm'    : 1 / deltat,
            'deltatDyn'            : deltat,
            'recip_deltatDyn'      : 1 / deltat,
            'useEVP'               : True,
            'useFreedrift'         : False,
            'useAdaptiveEVP'       : True,
            'useRelativeWind'      : False,
            'evpAlpha'             : 500,
            'evpBeta'              : 500,
            'nEVPsteps'            : 120,
        }
sett.add(input_settings)
