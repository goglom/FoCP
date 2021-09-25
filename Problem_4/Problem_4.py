import numpy as np
import matplotlib.pyplot as plt
from integrals import integral_trapeze, integral_simpson


def bessel_function(m: int, precision=8):
    def bessel(x: float):
        integrand_bessel = lambda t: np.cos(m * t - x * np.sin(t))
        return integral_simpson(integrand_bessel, (0, np.pi), 2 ** precision) / np.pi

    return np.vectorize(bessel)


def _derivatives_grid(function, bounds: tuple, steps_num: int):
    step = (bounds[1] - bounds[0]) / (steps_num - 1)
    x_grid = np.linspace(bounds[0] - step, bounds[1] + step, steps_num + 2)
    f_grid = function(x_grid)
    derivatives = np.zeros(steps_num, float)
    double_step = 2 * step

    for i in range(len(derivatives)):
        derivatives[i] = (f_grid[i + 2] - f_grid[i]) / double_step
    return derivatives


def derivatives_grid(function, bounds: tuple, steps_num: int, delta: float):
    x_grid = np.linspace(bounds[0], bounds[1], steps_num)
    x_grid = np.stack((x_grid - delta, x_grid + delta))
    f_grid = function(x_grid)
    double_delta = 2 * delta
    derivatives = (f_grid[1] - f_grid[0]) / double_delta
    return derivatives



N = 1024
interval = (0, 2 * np.pi)
x = np.linspace(*interval, N)
a = derivatives_grid(bessel_function(0), interval, N, 1e-2)
b = bessel_function(1)(x)
err = np.sqrt(np.sum((a + b)**2))
plt.title(f'L2 norm of difference: {err}')

plt.plot(x, a + b)
plt.grid()
plt.show()



