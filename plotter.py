import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.stats import norm

from multifractal.multifractal import Multifractal as mf
from multifractal.method_of_moments import MethodOfMoments as mom
from multifractal.simulator import Simulator as sim
from multifractal.data_handler import DataHandler as dh


class Plotter():
    def __init__(self, T, t, n, loc, scale, diffusion, drift):
        self.T = T
        self.t = t
        self.n = n
        self.loc = loc
        self.scale = scale
        self.drift = drift
        self.diffusion = diffusion

    def get_increments(self):
        model = sim.Simulator('mmar', T=self.T, dt_scale=self.t, loc=self.l, scale=self.s, diffusion=self.diffusion, drift=self.drift)
        return model.sim_mmar()[0]


    def get_realizations(self):
        R = np.array([])
        while R.size < self.n:
            R = np.append(R, self.get_increments())
        return R


    def plot_returns_pdfs(self, n):
        res = self.get_realizations()
        bins = np.histogram(res, bins=math.ceil(np.sqrt(res.size)))
        std = np.std(res)
        mean = np.mean(res)
        x = np.linspace(mean - 4*std, mean + 4*std, self.n)
        y = norm.pdf(x, mean, std)
        plt.hist(res, bins[1], density=True)
        plt.plot(x, y, label=f'Normal Distribution\n$\mu={mean}$, $\sigma={std}$')
