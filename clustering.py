import numpy as np


def loss(eta, q):
    """
    Calculates the loss between two arrays.

    Parameters:
    eta (ndarray): The first array.
    q (ndarray): The second array.

    Returns:
    float: The loss between the two arrays.
    """
    return np.sum(np.abs((eta == eta[:, None]).astype(int) - q))

def clustering(delta_hat, eta_hat, q, T, continue_add, continue_swap):
    """
    Perform clustering algorithm to optimize delta_hat.

    Args:
        delta_hat (numpy.ndarray): Array of length T representing the initial delta values.
        eta_hat (numpy.ndarray): Array of length T representing the initial eta values.
        q (float): Target value for clustering.
        T (int): Number of elements in delta_hat.
        continue_add (bool): Flag indicating whether to continue adding elements.
        continue_swap (bool): Flag indicating whether to continue swapping elements.

    Returns:
        numpy.ndarray: Array of length T representing the optimized delta values.
    """

    iteration = 0
    current_loss = loss(eta_hat, q)

    while continue_add or continue_swap:
        iteration += 1

        all_loss = []
        candidate_indexes = range(1, T)
        for i in candidate_indexes:

            delta_candidate = delta_hat.copy()

            delta_candidate[i] = 1 - delta_candidate[i]

            eta_candidate = np.cumsum(delta_candidate)

            candidate_loss = loss(eta_candidate, q)

            all_loss.append(candidate_loss)

        if all_loss and (min(all_loss) < current_loss):

            index_min = int(candidate_indexes[np.argmin(all_loss)])
            delta_hat[index_min] = 1 - delta_hat[index_min]

            current_loss = min(all_loss)
            continue_add = True
        else:
            continue_add = False

        if np.sum(delta_hat, dtype=int) in range(2, T):
            all_loss = []
            candidate_indexes = (
                np.where((delta_hat[1 : T - 1] - delta_hat[2:T]) != 0)[0] + 1
            )
            for i in candidate_indexes:
                delta_candidate = np.copy(delta_hat)

                mask = np.array([0, 1]) + i
                delta_candidate[mask] = 1 - delta_candidate[mask]

                eta_candidate = np.cumsum(delta_candidate)

                candidate_loss = loss(eta_candidate, q)

                all_loss.append(candidate_loss)

            if all_loss and (min(all_loss) < current_loss):

                index_min = candidate_indexes[np.argmin(all_loss)]
                index_min = np.array([0, 1]) + index_min

                delta_hat[index_min] = 1 - delta_hat[index_min]

                current_loss = min(all_loss)

                continue_swap = True
            else:
                continue_swap = False

        if iteration > 100000:

            continue_add = False
            continue_swap = False

        delta_final = delta_hat
        
    return delta_final