import numpy as np
from numba import jit

@jit(nopython=True)
def tma(x: np.array, a: np.array, b: np.array, c: np.array, d: np.array, cond) -> np.array:
    alpha = np.zeros_like(x)
    beta = np.zeros_like(x)
    y = np.zeros_like(x)

    alpha[0] = 0.0
    beta[0] = cond[0]

    for i in range(1, alpha.shape[0]):
        alpha[i] = - a[i] / (b[i] + c[i] * alpha[i - 1])
        beta[i] = (d[i] - c[i] * beta[i - 1]) / (b[i] + c[i] * alpha[i - 1])

    y[-1] = cond[1]

    for j in range(y.shape[0] - 2, -1, -1):
        y[j] = alpha[j] * y[j + 1] + beta[j]

    return y


def dif_solver(x:np.array, ic: np.array, p: np.array = None, q: np.array = None, r: np.array = None) -> np.array:
    """
        This function provide solution to equation:
            y'' + p(x) * y' + q(x) = r(x), x in [a, b]
        with initial conditios:
            y(a) = ic[0];

            y(b) = ic[1]
    """
    p = np.zeros_like(x) if p is None else p
    q = np.zeros_like(x) if q is None else q
    r = np.zeros_like(x) if r is None else r
    assert x.shape == p.shape and p.shape == q.shape and  p.shape == r.shape

    h = x[1] - x[0]
    a = 1 - p * h / 2
    b = -2 + q * h**2
    c = 1 + p * h / 2
    d = r * h **2

    return tma(x, a, b, c, d, ic)
