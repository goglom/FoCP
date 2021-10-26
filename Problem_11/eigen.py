import numpy as np
from tma import tma_solver


def back_iteration_eigen_solver(a: np.array, b: np.array, c: np.array, start_eigvec: np.array, iters: int):
    eigvec = start_eigvec / np.linalg.norm(start_eigvec)
    eigval = np.nan

    for _ in range(iters):
        eigvec = tma_solver(a, b, c, eigvec)
        norm = np.linalg.norm(eigvec)
        eigval = 1 / norm
        eigvec /= norm
        
    return eigvec / np.linalg.norm(eigvec), eigval
