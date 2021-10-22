import numpy as np


def back_iteration_eigen_solver(matrix: np.array, start_eigvec: np.array, start_eigval: float, tol: float, lin_solver):
    assert matrix.dtype == float and matrix.shape[0] == matrix.shape[1]
    n = matrix.shape[0]
    prev_eigval = start_eigval
    prev_eigvec = start_eigvec
    error = np.inf

    while error > tol:
        eigvec = lin_solver(matrix - prev_eigval * np.identity(n, dtype=float), prev_eigvec)
        eigval = prev_eigval + np.mean(prev_eigvec / eigvec)
        error = abs(abs(eigval) - abs(prev_eigval))
        prev_eigval = eigval
        prev_eigvec = eigvec

    return eigvec / np.linalg.norm(eigvec), eigval
