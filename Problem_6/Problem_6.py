import numpy as np
from matplotlib import pyplot as plt


def simple_eiler_method(x_grid: np.array, func, initial_condition: float):
    y_res = np.zeros_like(x_grid)
    y_res[0] = initial_condition

    for i in range(1, len(y_res)):
        step = x_grid[i] - x_grid[i - 1]
        y_res[i] = y_res[i - 1] + step * func(x_grid[i - 1], y_res[i - 1])

    return y_res


def runge_kutta_second_method(x_grid: np.array, func, initial_condition, alpha: float = 3/4):
    y_res = np.zeros_like(x_grid)
    y_res[0] = initial_condition

    for i in range(1, len(y_res)):
        x, y = x_grid[i - 1], y_res[i - 1]
        step = x_grid[i] - x_grid[i - 1]
        k = func(x, y)
        factor = step / (2 * alpha)
        y_res[i] = y + step * ((1 - alpha) * k + alpha * func(x + factor, y + factor * k))

    return y_res


def runge_kutta_fourth_method(x_grid: np.array, func, initial_condition):
    y_res = np.zeros_like(x_grid)
    y_res[0] = initial_condition

    for i in range(1, len(y_res)):
        x, y = x_grid[i - 1], y_res[i - 1]
        h = x_grid[i] - x_grid[i - 1]
        k1 = func(x, y)
        k2 = func(x + h / 2, y + h * k1 / 2)
        k3 = func(x + h / 2, y + h * k2 / 2)
        k4 = func(x + h, y + h * k3)
        y_res[i] = y + (k1 + 2 * k2 + 2 * k3 + k4) * h / 6
    return y_res


f = lambda x, y: -y

steps = 64
x_grid = np.linspace(0, 5, steps)

y_ground_truth = np.exp(-x_grid)
y_calc = [
    simple_eiler_method(x_grid, f, 1.0),
    runge_kutta_second_method(x_grid, f, 1.0),
    runge_kutta_fourth_method(x_grid, f, 1.0)
    ]
names = ["Eiler", "Runge-Kutta 2", "Runge-Kutta 4"]

plt.figure(figsize=(8, 6))

for i, (y, name) in enumerate(zip(y_calc, names)):
    plt.subplot(3, 1, 1 + i)
    err = np.sqrt(np.sum((y - y_ground_truth)**2))
    plt.title(name + f' (l2 error = {err:.1e})')
    plt.plot(x_grid, (y_ground_truth - y), label=name + f' err = {err:.1e}')

plt.grid()
plt.legend()
plt.tight_layout()
plt.show()

