import matplotlib.pyplot as plt
import numpy as np

def plot_population_data(S, R, I, gt_changing_points, population_unit = 1e6):
    """
    Plots the population data for susceptible (S), recovered (R), and infected (I) individuals over time.

    Parameters:
    S (array-like): Array of susceptible population counts over time.
    R (array-like): Array of recovered population counts over time.
    I (array-like): Array of infected population counts over time.

    Returns:
    None
    """
    plt.plot(S/population_unit, label="S")
    plt.plot(I/population_unit, label="I")
    plt.plot(R/population_unit, label="R")

    i = 0
    for c_point in gt_changing_points[1:]:
        if i == 0:
            plt.axvline(x=c_point-1, color="black", linestyle="--", alpha=0.5, label="True Changing Points")
            i += 1
        else:
            plt.axvline(x=c_point-1, color="black", linestyle="--", alpha=0.5)       
    plt.xlim(0,len(S))
    plt.xlabel("Time [days]")
    plt.ylabel("Population [Milions of people]")
    plt.legend()
    plt.show()
    
def safe_log(x):
    x = np.array(x)
    mask = x > 1e-16
    output = np.zeros(x.shape)
    output[mask] = np.log(x[mask])
    output[~mask] = -10_000
    return output

def constraint_lhs(deltas, tau_l, tau_u):
    n_samples = deltas.shape[0]
    sum_tau = np.sum(deltas[:, tau_l:tau_u]) / n_samples
    return sum_tau

def possible_taus(tau_k, delta_tau, T):
    lower= tau_k
    upper= tau_k +delta_tau
    possible=[]
    for i in range(delta_tau+1):
        if upper<=T and lower>=0:
            possible.append((lower, upper))
            lower-=1
            upper-=1
    return possible
