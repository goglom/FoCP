import numpy as np
from tma import tma_solver


def back_iteration_eigen_solver(a: np.array, b: np.array, c: np.array, start_eigvec: np.array, iters: int):
    eigvec = start_eigvec
    eigval = np.nan

    for _ in range(iters):
        new_eigvec = tma_solver(a, b, c, eigvec)
        eigval = np.linalg.norm(eigvec) / np.linalg.norm(new_eigvec)
        eigvec = new_eigvec
        
    return eigvec / np.linalg.norm(eigvec), eigval
