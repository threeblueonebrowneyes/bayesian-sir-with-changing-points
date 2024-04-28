import numpy as np
from scipy.special import gamma as gamma_func
from utils import *

def add(delta, T):
    """
    Adds a new element to the `delta` array by randomly selecting an index from the candidates where `delta` is 0,
    and setting the value at that index to 1.

    Parameters:
    delta (numpy.ndarray): The input array.
    T (int): The length of the array.

    Returns:
    numpy.ndarray: The updated `delta` array with a new element added.
    """
    _delta = delta.copy()
    candidate_idx = np.arange(T)[(_delta == 0)]
    index = np.random.choice(candidate_idx)
    _delta[index] = 1
    return _delta


def delete(delta, T):
    """
    Deletes a random element from the given array 'delta' by setting its value to 0.

    Parameters:
    delta (numpy.ndarray): The input array from which an element will be deleted.
    T (int): The length of the array 'delta'.

    Returns:
    numpy.ndarray: The modified array 'delta' with a random element set to 0.
    """
    _delta = delta.copy()
    candidate_idx = np.arange(1, T)[_delta[1:] == 1]
    index = np.random.choice(candidate_idx)
    _delta[index] = 0
    return _delta


def swap(delta, T):
    """
    Swaps two consecutive elements in the `delta` array.

    Parameters:
    delta (numpy.ndarray): The input array of binary values.
    T (int): The length of the `delta` array.

    Returns:
    numpy.ndarray: The modified `delta` array with two consecutive elements swapped.
    """
    _delta = delta.copy()  
    candidate_idx = np.arange(1,T-1)[_delta[1: T - 1] - _delta[2:T] != 0]
    index = np.random.choice(candidate_idx)
    _delta[index] = 1 - _delta[index]
    _delta[index + 1] = 1 - _delta[index + 1]

    return _delta


def propose_delta(delta, T):
    """
    Proposes a new delta based on the current delta and a given threshold T.
    
    Parameters:
        delta (numpy.ndarray): The current delta array.
        T (int): The threshold value.
        
    Returns:
        numpy.ndarray: The proposed delta array.
    """
    
    delta_orig = delta.copy()
    _K = np.sum(delta_orig)
    
    if _K == 1:
        
        delta_ = add(delta_orig, T)
        
    elif _K == T:
        
        delta_ = delete(delta_orig, T)
        
    else:
        
        _idx = np.random.choice(np.arange(3))
        p_list = [add, delete, swap]
        delta_ = p_list[_idx](delta_orig, T)
        
    return delta_.copy()


def log_j_ratio(sum_candidate, sum_original, T):
    """
    Calculates the logarithm of the ratio of the transition probabilities for a Metropolis-Hastings algorithm.

    Parameters:
    sum_candidate (int): The sum of the candidate values.
    sum_original (int): The sum of the original values.
    T (int): The total number of values.

    Returns:
    float: The logarithm of the ratio of the transition probabilities.
    """
    if sum_original == sum_candidate:
        return 0  # np.log(1)
    elif (sum_candidate, sum_original) == (1, 2) or (sum_candidate, sum_original) == (T, T-1):
        return np.log(3./(T-1))
    elif (sum_candidate, sum_original) == (2, 1) or (sum_candidate, sum_original) == (T-1, T):
        return np.log((T-1)/3.)
    elif sum_candidate == (sum_original-1):
        return np.log((sum_original-1)/(T-sum_candidate))
    else:
        return np.log((T-sum_original)/(sum_candidate-1))
    
def log_likelihood(delta, beta, gamma):
    """
    Calculate the log-likelihood of the Bayesian SIR model.

    Parameters:
    delta (numpy.ndarray): Array of observed values.
    beta (numpy.ndarray): Array of beta values.
    gamma (numpy.ndarray): Array of gamma values.

    Returns:
    float: The log-likelihood value.
    """
    _delta = delta.copy()
    eta = np.cumsum(_delta)
    K = np.sum(_delta, dtype=int)

    total = 0
    for i in range(1, K + 1):
        indic = eta == i
        _n = np.sum(indic)
        _sum_gamma = np.sum(safe_log(gamma[indic]))
        _sum_beta = np.sum(beta[indic])
        total += 0.2 * safe_log(0.1) - 2 * safe_log(gamma_func(0.1))
        total += 2 * safe_log(gamma_func(0.1 + _n)) - _sum_gamma
        total += -(0.1 + _n) * (safe_log(0.1 + _sum_beta) +
                                safe_log(0.1 - _sum_gamma))

    return total

def accept_delta(delta_original, delta_candidate, beta, gamma, T, p):
    """
    Accepts or rejects a candidate delta based on the Metropolis-Hastings algorithm.

    Parameters:
    - delta_original (numpy.ndarray): The original delta array.
    - delta_candidate (numpy.ndarray): The candidate delta array.
    - beta (float): The beta parameter.
    - gamma (float): The gamma parameter.
    - T (int): The total number of time steps.
    - p (float): The probability of accepting the candidate delta.

    Returns:
    - numpy.ndarray: The accepted delta array.
    """

    exponent = np.sum(delta_candidate - delta_original)
    prior_ratio = exponent * np.log(p / (1 - p))
    
    sum_candidate = np.sum(delta_candidate)
    sum_original = np.sum(delta_original)
    
    j_ratio = log_j_ratio(sum_candidate, sum_original, T)
    
    likelihood_ratio = log_likelihood(delta_candidate, beta, gamma) - log_likelihood(delta_original, beta, gamma)

    mh_log = prior_ratio + likelihood_ratio + j_ratio

    prob = min(0, mh_log)

    random_num = np.log(np.random.random())

    if random_num < prob:
        return delta_candidate
    else:
        return delta_original
    
def update_b(delta, beta):
    """
    Update the values of b based on the given delta and beta arrays.

    Parameters:
    delta (numpy.ndarray): An array containing the delta values.
    beta (numpy.ndarray): An array containing the beta values.

    Returns:
    numpy.ndarray: An array containing the updated values of b.
    """
    K = np.sum(delta)
    b = np.zeros(K)
    eta = np.cumsum(delta)
    for k in range(1, K + 1):
        b[k - 1] = np.random.gamma(
            shape=0.1 + np.sum(eta == k),
            scale= 1.0 / (0.1 + np.sum(beta[eta == k])),
        )

    return b[eta - 1]


def update_r(delta, gamma):
    """
    Update the values of r based on the given delta and gamma.

    Parameters:
    delta (numpy.ndarray): An array of delta values.
    gamma (numpy.ndarray): An array of gamma values.

    Returns:
    numpy.ndarray: An array of updated r values.
    """
    K = np.sum(delta)
    r = np.zeros(K)
    eta = np.cumsum(delta)
    for k in range(1, K + 1):
        r[k - 1] = np.random.gamma(
            shape=0.1 + np.sum(eta == k),
            scale= 1.0 / (0.1 - np.sum(safe_log(gamma[eta == k]))),
        )

    return r[eta - 1]

def update_beta(b, T, S_0, P_0, S, P, delta_I):
    """
    Update the beta values for each time step in the Bayesian SIR model.

    Parameters:
    - b (array-like): Array of beta values for each time step.
    - T (int): Total number of time steps.
    - S_0 (float): Initial susceptible population.
    - P_0 (float): Initial total population.
    - S (array-like): Array of susceptible population values for each time step.
    - P (array-like): Array of total population values for each time step.
    - delta_I (array-like): Array of new infected cases for each time step.

    Returns:
    - _beta (array-like): Array of updated beta values for each time step.
    """
    _beta = np.zeros(T)
    for t in range(T):
        if t != 0:
            y = np.random.beta(
                a=S[t - 1] - delta_I[t] + b[t] / P[t - 1] + 1,
                b=delta_I[t] + 1,
            )
            _beta[t] = -safe_log(y) / P[t - 1]
        else:
            _beta[t] = y = np.random.beta(
                a=S_0 - delta_I[t] + b[t] / P_0 + 1,
                b=delta_I[t] + 1,
            )
            _beta[t] = -safe_log(y) / P_0

    return _beta



def update_gamma(r, T, I_0, I, delta_R):
    """
    Update the gamma values for each time step in the Bayesian SIR model.

    Parameters:
    - r: array-like, the recovery rate at each time step
    - T: int, the total number of time steps
    - I_0: int, the initial number of infected individuals
    - I: array-like, the number of infected individuals at each time step
    - delta_R: array-like, the number of newly recovered individuals at each time step

    Returns:
    - _gamma: array-like, the updated gamma values for each time step
    """
    _gamma = np.zeros(T)
    for t in range(T):
        if t != 0:
            _gamma[t] = np.random.beta(
                a=delta_R[t] + r[t],
                b=1 + I[t-1] - delta_R[t],
            )
        else:
            _gamma[t] = np.random.beta(
                a=delta_R[t] + r[t],
                b=1 + I_0 - delta_R[t],
            )

    return _gamma