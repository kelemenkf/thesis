import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from scipy.stats import norm

from repos.multifractal.multifractal import Multifractal
from repos.multifractal.method_of_moments import MethodOfMoments
from repos.multifractal.simulator import Simulator
from repos.multifractal.data_handler import DataHandler


class Plotter():
    def __init__(self, T, dt, n, diffusion, drift, loc=0, scale=1):
        self.T = T
        self.dt = dt
        self.n = n
        self.loc = loc
        self.scale = scale
        self.drift = drift
        self.diffusion = diffusion

    def get_increments(self):
        model = Simulator('mmar', T=self.T, dt_scale=self.dt, loc=self.loc, scale=self.scale, diffusion=self.diffusion, drift=self.drift)
        return model.sim_mmar()[0]


    def get_realizations(self):
        R = np.array([])
        while R.size < self.n:
            R = np.append(R, self.get_increments())
        return R


    def plot_returns_pdfs(self):
        res = self.get_realizations()
        bins = np.histogram(res, bins=math.ceil(np.sqrt(res.size)))
        std = np.std(res)
        mean = np.mean(res)
        x = np.linspace(mean - 4*std, mean + 4*std, self.n)
        y = norm.pdf(x, mean, std)
        plt.hist(res, bins[1], density=True)
        plt.plot(x, y, label=f'Normal Distribution\n$\mu={mean}$, $\sigma={std}$')


    def plot_scaling_function_bm(self):
        sample = Simulator(sim_type='bm', diffusion=self.diffusion, drift=self.drift)
        sample = sample.sim_bm(self.n)[0]
        sample = pd.DataFrame(sample, columns=['logreturn'])
        bm_handler = DataHandler(sample, obs_data=False)
        X, eps = bm_handler.get_data()
        bm_mom = MethodOfMoments('binomial', X=X, delta_t=eps, q=[0.1,5], gran=0.1)
        bm_mom.plot_tau_q()
