import numpy as np
from numba import jit

@jit(nopython=True)
def tma(x: np.array, a: np.array, b: np.array, c: np.array, d: np.array, cond: np.array) -> np.array:
    alpha = np.zeros_like(x)
    beta = np.zeros_like(x)
    y = np.zeros_like(x)

    alpha[0] = 0.0
    beta[0] = cond[0]

    for i in range(1, alpha.shape[0] - 1):
        alpha[i] = - a[i] / (b[i] + c[i] * alpha[i - 1])
        beta[i] = (d[i] - c[i] * beta[i - 1]) / (b[i] + c[i] * alpha[i - 1])

    y[-1] = cond[1]

    for j in range(y.shape[0] - 2, -1, -1):
        y[j] = alpha[j] * y[j + 1] + beta[j]

    return y
