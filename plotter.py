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
    def __init__(self, T, dt, n, diffusion, drift, sim_type='mmar', loc=0, scale=1):
        self.T = T
        self.dt = dt
        self.n = n
        self.loc = loc
        self.scale = scale
        self.sim_type = sim_type
        self.drift = drift
        self.diffusion = diffusion

    def get_increments(self):
        model = Simulator(self.sim_type, T=self.T, dt_scale=self.dt, loc=self.loc, scale=self.scale, diffusion=self.diffusion, drift=self.drift)
        return model.sim_mmar()[0]


    def get_realizations(self):
        R = np.array([])
        while R.size < self.n:
            R = np.append(R, self.get_increments())
        return R


    def plot_returns_density(self):
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


    def plot_dist(self, increments=[1,7,30,180]):
        '''
        Plots the return distribution of a single realization of an mmar.
        '''
        fig, axes = plt.subplots(len(increments), 1, sharex='row', figsize=(20, 40))
        fig.subplots_adjust(hspace=0.5)

        for i in range(len(increments)):
            self.n = self.T // increments[i]
            self.dt_scale = increments[i]

            sim = Simulator(self.sim_type, T=self.T, dt_scale=self.dt, loc=self.loc, scale=self.scale, diffusion=self.diffusion, drift=self.drift)

            if self.sim_type in ['mmar_m', 'mmar']:
                y, _ = sim.sim_mmar()
            elif self.sim_type in ['bm', 'fbm']:
                y, _ = sim.sim_bm(self.n)
                y = np.diff(y)
            bins = np.histogram(y, bins=math.ceil(np.sqrt(y.size)))  

            axes[i].hist(y, bins[1], density=True)
            axes[i].set_title(f"Return distribution at scale of {increments[i]} days.")
            axes[i].set_xlabel("X(t)")
            axes[i].set_ylabel("Density")