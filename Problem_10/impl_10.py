import numpy as np
from tma import tma
from numba import jit

@jit(nopython=True)
def heat_eq_solver(x: np.array, t: np.array, sigma: float, y_t0: np.array, edge_cond: np.array):
    dx = x[1] - x[0]
    dt = t[1] - t[0]

    a = np.full((x.shape[0] - 1), -sigma * dt / dx**2 / 2)
    c = a.copy()
    a[0] = 0.0
    c[-1] = 0.0
    b = np.full_like(a, 1 + sigma * dt / dx**2)
    
    y = np.zeros((t.shape[0], x.shape[0]))
    y[0, :] = y_t0
    
    for i in range(0, t.shape[0] - 1):
        d = np.array([y[i, j] * (1 - dt / dx) + dt / (2 * dx) * (y[i, j + 1] + y[i, j - 1])\
             for j in range(1, a.shape[0] - 1)])

        y[i + 1, :] = tma(x, a, b, c, d, edge_cond)
    return y
