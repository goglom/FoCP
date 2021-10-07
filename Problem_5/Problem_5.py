import numpy as np
from matplotlib import pyplot as plt


def _poly_newton_coefficient(x: np.array, y: np.array):
    m = len(x)
    a = np.copy(y)

    for k in range(1, m):
        a[k:m] = (a[k:m] - a[k - 1])/(x[k:m] - x[k - 1])

    return a


def newton_polynomial(x_data: np.array, y_data: np.array):
    def _poly_func(x: np.number):
        p_degree = len(x_data) - 1
        a = _poly_newton_coefficient(x_data, y_data)
        p = a[p_degree]
        for k in range(1, p_degree + 1):
            p = a[p_degree - k] + (x - x_data[p_degree - k]) * p
        return p
    return np.vectorize(_poly_func)


def lagrange_polynomial(x_data: np.array, y_data: np.array):
    def _help_func(i: int, x: np.number):
        result = 1
        for j in range(len(x_data)):
            if i != j:
                result *= x - x_data[j]
        return result

    def _poly_func(x: np.number):
        sum_val = 0
        for i in range(len(x_data)):
            sum_val += y_data[i] * _help_func(i, x) / _help_func(i, x_data[i])
        return sum_val
    return np.vectorize(_poly_func)


n_vals = [i for i in range(4, 15 + 1)]
x_draw_grid = np.linspace(0.1, 3, 1000)
plt.figure(figsize=(14, 6))

method = lagrange_polynomial

for n in n_vals:
    x_data = np.arange(0, n) / n + 1
    y_data = np.log(x_data)
    y_interpolated = method(x_data, y_data)

    y_draw_1 = np.abs(y_interpolated(x_draw_grid) - np.log(x_draw_grid))
    y_draw_2 = y_interpolated(x_draw_grid)
    plt.subplot(1, 2, 1)
    plt.plot(x_draw_grid, y_draw_1, label=f'n = {n}')
    plt.subplot(1, 2, 2)
    plt.plot(x_draw_grid, y_draw_2, label=f'n = {n}')

plt.title("Function values")
plt.plot(x_draw_grid, np.log(x_draw_grid), "--", color='black', linewidth=2, label='original')
plt.grid()
plt.legend()
#plt.xscale("log", base=2)

plt.title("Function diffs")
plt.subplot(1, 2, 1)
#plt.yscale("log")
plt.grid()
plt.legend()

plt.show()



