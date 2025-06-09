from veros.core.operators import update, at, numpy as npx
from veros.state import VerosState
from veros import veros_routine
from veris.variables import VARIABLES
from veris.settings import SETTINGS


grid_len = 128
nx, ny = grid_len, grid_len
olx, oly = 2, 2


##### create wind and ocean surface currents (and initial
    # sea ice thickness) similar to the Mehlmann benchmark #####

Lx=512e3
Ly=Lx
dx = Lx/(nx-1)
dy = dx

f0=1.4e-4
gravity=9.81
Ho=1000
accuracy='float64'
x = (npx.arange(nx,dtype = accuracy)+0.5)*dx;
y = (npx.arange(ny,dtype = accuracy)+0.5)*dy;
xx,yy = npx.meshgrid(x,y);
xu,yu = npx.meshgrid(x-.5*dx,y);
xv,yv = npx.meshgrid(x,y-.5*dy);

# Flat bottom at z=-Ho
h=-Ho*npx.ones((ny,nx),dtype = accuracy);
# channnel walls
h = update(h, at[:,-1], 0);
h = update(h, at[-1,:], 0);

variableWindField = True
shiftWindField = True
if variableWindField:
# variable wind field
    period = 16 # days
    if shiftWindField: period = 4 # days
    writeFreq = 3 # hours
    t = npx.arange(0,period*24/writeFreq)/(24./writeFreq) # time in days
    vmax = 15.0; # maximale windgeschwindigkeit in m/s

    if shiftWindField:
        t = t + 4
        mx = Lx*.1 + Lx*.1*t
        my = Ly*.1 + Ly*.1*t
    else:
        tP=npx.mod(t,period/2)
        tP[t>=period/2.]=period/2.-tP[t>=period/2.]
        tP = tP/(0.5*period)
        oLx=150.e3
        oLy=oLx
        mx = -oLx+(2*oLx+Lx)*tP
        my = -oLy+(2*oLy+Ly)*tP

    alpha0= npx.pi/2. - npx.pi/2./5. # 90 grad ist ohne Konvergenz oder Divergenz
    alpha = npx.pi/2. - npx.pi/2./5.*npx.maximum(npx.sign(npx.roll(mx,-1)-mx),0.) \
            -npx.pi/2./10.*npx.maximum(npx.sign(mx-npx.roll(mx,-1)),0.)

    uwind = npx.zeros((t.shape[0],xx.shape[0],xx.shape[1]))
    vwind = npx.zeros((t.shape[0],yy.shape[0],yy.shape[1]))
    for k,myt in enumerate(t):
        wx =  npx.cos(alpha[k])*(xx-mx[k]) + npx.sin(alpha[k])*(yy-my[k])
        wy = -npx.sin(alpha[k])*(xx-mx[k]) + npx.cos(alpha[k])*(yy-my[k])
        r = npx.sqrt((mx[k]-xx)*(mx[k]-xx)+(my[k]-yy)*(my[k]-yy))
        s = 1.0/50.e3*npx.exp(-r/100.e3)
        if shiftWindField:
            w = npx.tanh(myt*(8.0-myt)/2.)
        else:
            if myt<8:
                w = npx.tanh(myt*(period/2.-myt)/2.)
            elif myt>=8 and myt<16:
                w = -npx.tanh((myt-period/2.)*(period-myt)/2.)
        #    w = np.sin(2.0*np.pi*myt/period)

        # reset scaling factor w to one
        w = 1.
        uwind = update(uwind, at[k,:,:], -wx*s*w*vmax);
        vwind = update(vwind, at[k,:,:], -wy*s*w*vmax);

        spd=npx.sqrt(uwind[k,:,:]**2+vwind[k,:,:]**2)
        div=uwind[k,1:-1,2:]-uwind[k,1:-1,:-2] \
             +vwind[k,2:,1:-1]-vwind[k,:-2,1:-1]

# ocean
uo = +0.01*(2*yy-Ly)/Ly
vo = -0.01*(2*xx-Lx)/Lx

# initial thickness:
hice = 0.3 + 0.005*npx.sin(500*xx) + 0.005*npx.sin(500*yy)
# symmetrize
hices = 0.5*(hice + hice.transpose())
# initial thickness for comparison with:
hice = 0.3 + 0.005*(npx.sin(60./1000.e3*xx) + npx.sin(30./1000.e3*yy))


##### create fields with overlap, also define grid and masks #####

def fill_overlap(A):
        A = update(A, at[:2, :], A[-4:-2, :])
        A = update(A, at[-2:, :], A[2:4, :])
        A = update(A, at[:, :2], A[:, -4:-2])
        A = update(A, at[:, -2:], A[:, 2:4])

        return A

ones2d = npx.ones((grid_len+4,grid_len+4))

def create(val):
    return ones2d * val

def createfrom(field):
    field_full_size = ones2d * 1
    field_full_size = update(field_full_size, at[2:-2,2:-2], field)
    field_full_size = fill_overlap(field_full_size)
    return field_full_size


@veros_routine
def set_inits(state):
    vs = state.variables

    #vs.hIceMean  = createfrom(hice)
    hIceMean  = create(0.3)
    hSnowMean = create(0)
    Area      = create(1)
    TSurf     = create(273.0)
    
    uWind  = createfrom(uwind[15,:,:])
    vWind  = createfrom(vwind[15,:,:])
    
    maskInC = create(1)
    maskInC = update(maskInC, at[-3,:], 0)
    maskInC = update(maskInC, at[:,-3], 0)
    maskInU = maskInC * npx.roll(maskInC,5,axis=1)
    maskInV = maskInC * npx.roll(maskInC,5,axis=0)
    
    maskInC = fill_overlap(maskInC)
    maskInU = fill_overlap(maskInU)
    maskInV = fill_overlap(maskInV)
    
    uOcean = createfrom(uo) * maskInU
    vOcean = createfrom(vo) * maskInV
    
    R_low = createfrom(h)

    fcormax = 0.0001562
    fcormin = 0.00014604
    y = npx.linspace(fcormin, fcormax, grid_len)
    y = y[:,npx.newaxis] * npx.ones((grid_len,grid_len))
    fCori = createfrom(y)
    #vs.fCori = create(1.46e-4)
    fCori = create(0)

    iceMask, iceMaskU, iceMaskV  = maskInC, maskInU, maskInV

    deltaX = create(8000)
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

    fields = npx.array(fields).transpose(0,2,1)

    (vs.hIceMean, vs.hSnowMean, vs.Area, vs.TSurf, vs.uOcean, vs.vOcean, vs.uWind, vs.vWind, vs.R_low,
    vs.maskInC, vs.maskInU, vs.maskInV, vs.iceMask, vs.iceMaskU, vs.iceMaskV, vs.fCori,
    vs.dxC, vs.dyC, vs.dxG, vs.dyG, vs.dxU, vs.dyU, vs.dxV, vs.dyV,
    vs.recip_dxC, vs.recip_dyC, vs.recip_dxG, vs.recip_dyG, vs.recip_dxU, vs.recip_dyU, vs.recip_dxV, vs.recip_dyV,
    vs.rA, vs.rAz, vs.rAu, vs.rAv, vs.recip_rA, vs.recip_rAz, vs.recip_rAu, vs.recip_rAv
    ) = fields


dimensions = dict(xt=nx, yt=ny)
state = VerosState(VARIABLES, SETTINGS, dimensions)
state.initialize_variables()
set_inits(state)

deltat = 3600
input_settings = {
            'deltatTherm'       : deltat,
            'recip_deltatTherm' : 1 / deltat,
            'deltatDyn'         : deltat,
            'recip_deltatDyn'   : 1 / deltat,
            'gridcellWidth'     : 8000,
            'veros_fill'        : False,
            'useEVP'            : True,
            'useAdaptiveEVP'    : True,
            'useRelativeWind'   : False,
            'evpAlpha'          : 123456.7,
            'evpBeta'           : 123456.7,
            'nEVPsteps'         : 400
        }

with state.settings.unlock():
    state.settings.update(input_settings)