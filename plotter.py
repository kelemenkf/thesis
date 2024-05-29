import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.stats import norm

from multifractal.multifractal import Multifractal as mf
from multifractal.method_of_moments import MethodOfMoments as mom
from multifractal.simulator import Simulator as sim
from multifractal.data_handler import DataHandler as dh

def get_increments(t):
    model = sim.Simulator('mmar', T=512, dt_scale=t, loc=l, scale=s, diffusion=diffusion, drift=drift)
    return model.sim_mmar()[0]


def get_realizations(t, n):
    R = np.array([])
    while R.size < n:
        R = np.append(R, get_increments(t))
    return R

