import matplotlib.pyplot as plt
import numpy as np

def plot_population_data(S, R, I, gt_changing_points):
    """
    Plots the population data for susceptible (S), recovered (R), and infected (I) individuals over time.

    Parameters:
    S (array-like): Array of susceptible population counts over time.
    R (array-like): Array of recovered population counts over time.
    I (array-like): Array of infected population counts over time.

    Returns:
    None
    """
    plt.plot(S, label="S")
    plt.plot(I, label="I")
    plt.plot(R, label="R")
    for c_point in gt_changing_points[1:]:
        plt.axvline(x=c_point-1, color="black", linestyle="--", alpha=0.5)
        
    
    plt.xlim(0,100)
    plt.xlabel("Time")
    plt.ylabel("Population")
    plt.legend()
    plt.show()
    
def safe_log(x):
    x = np.array(x)
    mask = x > 1e-16
    output = np.zeros(x.shape)
    output[mask] = np.log(x[mask])
    output[~mask] = -10_000
    return output