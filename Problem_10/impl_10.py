import numpy as np
import os
import sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from Problem_9.impl_9 import tma
from numba import jit

@jit(nopython=True)
def heat_eq_solver(x: np.array, t: np.array, sigma: float, y_t0: np.array, edge_cond: np.array):
    dx = x[1] - x[0]
    dt = t[1] - t[0]

    a = np.full_like(x, -sigma * dt / dx**2)
    b = np.full_like(x, 1 + 2 * sigma * dt / dx**2)
    c = a.copy()
    d = np.zeros((t.shape[0], x.shape[0]))
    d[0, :] = y_t0
    
    for i in range(1, t.shape[0]):
        d[i, :] = tma(x, a, b, c, d[i - 1, :], edge_cond)
    return d
