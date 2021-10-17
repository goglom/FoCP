import numpy as np
from numba import jit

@jit(nopython=True)
def tma_solver(x:np.array, ct: np.array, p: np.array = None, q: np.array = None, r: np.array = None) -> np.array:
    """
        This function provide solution to equation:
            y'' + p(x) * y' + q(x) = r(x), x in [a, b]
        with initial conditios:
            ct[0, 0] * y(a) + ct[0, 1] * y'(a) = ct[0, 2];

            ct[1, 0] * y(b) + ct[1, 1] * y'(b) = ct[1, 2]
    """
    p = np.zeros_like(x) if p is None else p
    q = np.zeros_like(x) if q is None else q
    r = np.zeros_like(x) if r is None else r
    assert x.shape == p.shape and p.shape == q.shape and  p.shape == r.shape and ct.shape == (2, 3)

    h = x[1] - x[0]
    ksi = np.zeros_like(x)
    eta = np.zeros_like(x)
    y = np.zeros_like(x)

    ksi[0] = -ct[0, 1] / (h * ct[0, 0] - ct[0, 1])
    eta[0] = h * ct[0, 2] / (h * ct[0, 0] - ct[0, 1])

    a = 1 / h**2 - p / (2 * h)
    b = 2 / h**2 - q
    c = 1 / h**2 + p / 2 * h
    d = r

    for i in range(ksi.shape[0] - 1):
        ksi[i + 1] = c[i] / (b[i] - a[i] * ksi[i])
        eta[i + 1] = (eta[i] * a[i] - d[i]) / (b[i] - a[i] * ksi[i])

    y[-1] = (ct[1, 1] * eta[-1] + h * ct[1, 2]) / (ct[1, 1] * (1 - ksi[-1]) + h * ct[1, 0])

    for j in range(y.shape[0] - 1, 0, -1): 
        y[j - 1] = y[j] * ksi[j] + eta[j]

    return y