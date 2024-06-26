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
    def __init__(self, T, dt, n, diffusion=1, drift=0, sim_type='mmar', loc=0, scale=1, H=0.5):
        self.T = T
        self.dt = dt
        self.n = n
        self.loc = loc
        self.scale = scale
        self.sim_type = sim_type
        self.drift = drift
        self.diffusion = diffusion
        self.H = H
        self.handler = self.get_handler()
        self.sim_data = self.simulated_data()


    def get_handler(self):
        '''
        Initiates a DataHandler for handling data obtained through simulation.
        '''
        sample = Simulator(sim_type='bm', diffusion=self.diffusion, drift=self.drift)
        sample = sample.sim_bm(self.n)[0]
        sample = pd.DataFrame(sample, columns=['logreturn'])
        bm_handler = DataHandler(sample, obs_data=False)
        return bm_handler


    def simulated_data(self):
        '''
        Returns the simulated data in a format required for multifractal analysis. 
        '''
        X, eps = self.handler.get_data()
        return X, eps


    def plot_returns(self, save, path, name):
        '''
        Plots with returns simulated through WBM.
        '''
        self.handler.plot_x_diff(save=save, path=path, name=name)

    
    def plot_cumulative_returns(self, save, path, name):
        '''
        Plots cumulative returns. 
        '''
        self.handler.plot_x(save=save, path=path, name=name)
    

    def plot_scaling_function_bm(self, save=False, path="", name=""):
        '''
        Plots the scaling function (tau(q)) of a brownian motion, with drift equal to self.drift, 
        and diffusion equal to self.diffusion. The arguments determine if and where to save the resulting plot. 
        '''
        bm_mom = MethodOfMoments('binomial', X=self.sim_data[0], delta_t=self.sim_data[1], q=[0.1,5], gran=0.1)
        bm_mom.plot_tau_q(save=save, path=path, name=name)


    def plot_partition_function_bm(self, save=False, path="", name=""):
        '''
        Plots the partition function of WBM data. 
        '''
        bm_mom = MethodOfMoments('binomial', X=self.sim_data[0], delta_t=self.sim_data[1], q=[0.1,5], gran=0.1)
        bm_mom.partition_plot(save=save, renorm=True, path=path, name=name)


    def plot_multifractal_spectrum_bm(self, save=False, path="", name=""):
        '''
        Plots the multifractal specturm of WBM data. 
        '''
        bm_mom = MethodOfMoments('binomial', X=self.sim_data[0], delta_t=self.sim_data[1], q=[0.1,5], gran=0.1)
        bm_mom.plot_fitted_f_alpha(save=save, path=path, name=name)


    def get_increments(self):
        '''
        Returns an array of increments of a time series defined by self.sim_type, with length
        self.T, and scale self.dt_scale. N will be T/dt_scale. This is useful if we want to 
        plot returns at arbitrary scales. By making N = dt_scale, we get a single realization at the desired scale. 
        '''
        model = Simulator(self.sim_type, T=self.T, dt_scale=self.dt, loc=self.loc, scale=self.scale, diffusion=self.diffusion, drift=self.drift, H=self.H)
        return model.sim_mmar()[0]


    def get_realizations(self):
        '''
        Returns self.n number of discrete realizations of a process specified by self.sim_type. 
        '''
        R = np.array([])
        while R.size < self.n:
            R = np.append(R, self.get_increments())
        return R


    def plot_returns_density(self):
        '''
        Plots the density function of returns produced by the MMAR, and a normal distribution with similar 
        mean and standard deviation as the simulated data. 
        '''
        res = self.get_realizations()
        print(res.shape)
        bins = np.histogram(res, bins=math.ceil(np.sqrt(res.size)))
        std = np.std(res)
        mean = np.mean(res)
        x = np.linspace(mean - 4*std, mean + 4*std, self.n)
        y = norm.pdf(x, mean, std)
        plt.hist(res, bins[1], density=True)
        plt.plot(x, y, label=f'Normal Distribution\n$\mu={mean}$, $\sigma={std}$')


    def plot_dist(self, increments=[1,7,30,180]):
        '''
        Plots the return distribution of a single realization of an mmar at different increments, 
        defined by the argument increments.
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

