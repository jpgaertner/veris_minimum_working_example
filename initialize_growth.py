from veros.core.operators import numpy as npx
from veros.state import VerosState
from veros import veros_routine
from veris.variables import VARIABLES
from veris.settings import SETTINGS


nx, ny = 1, 1
olx, oly = 2, 2

ones2d = npx.ones((nx+2*olx, ny+2*oly))

@veros_routine
def set_inits(state):
    vs = state.variables
    
    vs.hIceMean    = ones2d * 1.3
    vs.hSnowMean   = ones2d * 0.1
    vs.Area        = ones2d * 0.9
    vs.TSurf       = ones2d * 273.0
    vs.SeaIceLoad  = ones2d * (SETTINGS['rhoIce'][0] * state.variables.hIceMean
                               + SETTINGS['rhoSnow'][0] * state.variables.hSnowMean)
    vs.wSpeed      = ones2d * 2
    vs.ocSalt      = ones2d * 29
    vs.theta       = ones2d * SETTINGS['celsius2K'][0] - 1.66
    vs.Qnet        = ones2d * 173.03212617345582
    vs.Qsw         = ones2d * 0
    vs.SWdown      = ones2d * 0
    vs.LWdown      = ones2d * 80
    vs.ATemp       = ones2d * 253
    vs.precip      = ones2d * 0
    
    vs.maskInC     = ones2d * 1
    vs.maskInU     = ones2d * 1
    vs.maskInV     = ones2d * 1
    vs.iceMask     = ones2d * 1
    vs.iceMaskU    = ones2d * 1
    vs.iceMaskV    = ones2d * 1
    
dimensions = dict(xt=nx, yt=ny)
state = VerosState(VARIABLES, SETTINGS, dimensions)
state.initialize_variables()
set_inits(state)
