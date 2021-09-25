import numpy as np
from numba import jit


@jit(nopython=True)
def _trapeze_square(func_values, points):
    diff = points[1] - points[0]
    if diff < 0:
        raise RuntimeWarning("left bound greater than right bond")
    return (func_values[0] + func_values[1]) * diff / 2


@jit(nopython=True)
def _simpson_square(func_vals: tuple, bounds: tuple):
    diff = bounds[2] - bounds[0]
    if diff < 0:
        raise RuntimeWarning("left bound greater than right bond")
    return (func_vals[0] + 4 * func_vals[1] + func_vals[2]) * diff / 6


def integral_trapeze(func, bounds: tuple, intervals_num: int) -> float:
    steps = np.linspace(*bounds, intervals_num + 1)
    func_values = func(steps)
    result = 0.0
    for i in range(intervals_num):
        result += _trapeze_square(func_values[i: i + 2], steps[i: i + 2])
    return result


def integral_simpson(func, bounds: tuple, intervals_num: int) -> float:
    if intervals_num % 2 != 0:
        raise RuntimeError("Intervals number must be even")

    steps = np.linspace(*bounds, intervals_num + 1)
    func_values = func(steps)
    result = 0.0
    for i in range(0, intervals_num - 1, 2):
        result += _simpson_square(func_values[i: i + 3], steps[i: i + 3])
    return result

