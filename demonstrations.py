from repos.multifractal.simulator import Simulator


def plot_scaling(n=3, process='bm', H=0.5):
    '''
    Demonstrates that x(t) -> a^Hx(at)
    '''
    for t in [2**i for i in range(1, n+1)]:
        sim = Simulator(sim_type=process, T=1000*t, dt_scale=t, H=H)
        sim.plot_process()



def mean_squared_distance_relationship(n=3, process='bm', H=0.5):
    '''
    Demonstrates the mean squared distance relationship  ⟨x^2(t)⟩ ∼ t^2H
    '''
    MSDS = []
    for t in  [2**i for i in range(1, n+1)]:
        sim = Simulator(process, T=1000*t, dt_scale=t, H=H)
        msd = sim.msd(10000)
        MSDS.append(msd)
    return MSDS
