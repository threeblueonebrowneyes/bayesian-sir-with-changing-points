import numpy as np
from scipy.special import gamma as gamma_func
from utils import *

def add(delta, T):
    _delta = delta.copy()
    candidate_idx = np.arange(T)[(_delta == 0)]
    index = np.random.choice(candidate_idx)
    _delta[index] = 1
    return _delta


def delete(delta, T):
    _delta = delta.copy()
    candidate_idx = np.arange(1,T)[_delta[1:] == 1]
    index = np.random.choice(candidate_idx)
    _delta[index] = 0
    return _delta


def swap(delta, T):
    _delta = delta.copy()  
    candidate_idx = np.arange(1,T-1)[_delta[1: T - 1] - _delta[2:T] != 0]
    index = np.random.choice(candidate_idx)
    _delta[index] = 1 - _delta[index]
    _delta[index + 1] = 1 - _delta[index + 1]

    return _delta


def propose_delta(delta, T):
    
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
    _gamma = np.zeros(T)
    for t in range(T):
        if t!=0:
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