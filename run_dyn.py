from time import perf_counter
import matplotlib.pyplot as plt
import cmocean.cm as cm
from veros import veros_routine
from veros import runtime_settings
runtime_settings.backend = 'numpy'
from veros.core.operators import numpy as npx

from veris.area_mass import SeaIceMass, AreaWS
from veris.dynsolver import WindForcingXY, IceVelocities
from veris.dynamics_routines import SeaIceStrength
from veris.ocean_stress import OceanStressUV
from veris.advection import Advection
from veris.clean_up import clean_up_advection, ridging
from veris.fill_overlap import fill_overlap
from initialize_dyn import state

@veros_routine
def dyn_model(state):
    vs = state.variables

    # calculate sea ice mass centered around c-, u-, and v-points
    vs.SeaIceMassC, vs.SeaIceMassU, vs.SeaIceMassV = SeaIceMass(state)

    # calculate sea ice cover fraction centered around u- and v-points
    vs.AreaW, vs.AreaS = AreaWS(state)

    # calculate surface forcing due to wind
    vs.WindForcingX, vs.WindForcingY = WindForcingXY(state)

    # calculate ice strength
    vs.SeaIceStrength = SeaIceStrength(state)

    # calculate ice velocities
    vs.uIce, vs.vIce, vs.sigma1, vs.sigma2, vs.sigma12 = IceVelocities(state)

    # calculate stresses on ocean surface
    vs.OceanStressU, vs.OceanStressV = OceanStressUV(state)

    # calculate change in sea ice fields due to advection
    vs.hIceMean, vs.hSnowMean, vs.Area = Advection(state)

    # correct overshoots and other pathological cases after advection
    (
        vs.hIceMean,
        vs.hSnowMean,
        vs.Area,
        vs.TSurf,
        vs.os_hIceMean,
        vs.os_hSnowMean,
    ) = clean_up_advection(state)

    # cut off ice cover fraction at 1 after advection
    vs.Area = ridging(state)

    # fill overlaps
    vs.hIceMean = fill_overlap(state,vs.hIceMean)
    vs.hSnowMean = fill_overlap(state,vs.hSnowMean)
    vs.Area = fill_overlap(state,vs.Area)

t0 = perf_counter()
n = 10
for n_timesteps in range(n):
    dyn_model(state)

t1 = perf_counter()

print(f'n_timesteps = {n}')
print(f'grid_len = {npx.shape(state.variables.hIceMean[2:-2,2:-2])[0]}')
print(f'time for model run:  {t1 - t0}')
