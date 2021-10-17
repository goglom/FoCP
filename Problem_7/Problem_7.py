import matplotlib
import numpy as np
from typing import Callable
from matplotlib import pyplot as plt
from itertools import product

MyCallable = Callable[[float, float, float], float]

def runge_kutta_second_method(x_grid: np.array, func1:MyCallable, func2: MyCallable, initial_conditions: tuple[float, float], alpha: float = 3/4) -> np.array:
    y_res = np.zeros((2, len(x_grid)))
    y_res[0][0] = initial_conditions[0]
    y_res[1][0] = initial_conditions[1]

    for i in range(1, len(x_grid)):
        x, y = x_grid[i - 1], (y_res[0][i - 1], y_res[1][i - 1])
        step = x_grid[i] - x_grid[i - 1]
        k = (func1(x, y[0], y[1]), func2(x, y[0], y[1]))
        factor = step / (2 * alpha)
        y_res[0][i] = y[0] + step * (
                (1 - alpha) * k[0] + alpha * func1(x + factor, y[0] + factor * k[0], y[1] + factor * k[1]))
        y_res[1][i] = y[1] + step * (
                    (1 - alpha) * k[1] + alpha * func2(x + factor, y[0] + factor * k[0], y[1] + factor * k[1]))

    return y_res


t_grid = np.arange(0, 10, 1e-2)
functions = (
    lambda t, x, y: 10 * x - 2 * x * y,
    lambda t, x, y: 2 * x * y - 10 * y
)

cond = np.arange(0.5, 0.6, 0.5)
conditions = [(0.5, 0.5), (4.9, 4.9)]#[(x, y) for x, y in product(cond, cond)]

cm_subsection = np.linspace(0., 1., len(conditions)) 
colors = [ matplotlib.cm.jet(x) for x in cm_subsection ]

plt.figure(figsize=(8, 7))
plt.title(f'Start time = {t_grid[0]}; end = {t_grid[-1]}; step = {t_grid[1] - t_grid[0]:.1e}')
for cond, color in zip(conditions, colors):
    xy_solution = runge_kutta_second_method(t_grid, *functions, cond)
    plt.plot(*cond, 'o', ms=4, c=color)
    plt.arrow(*cond, *((xy_solution[:, 1] - xy_solution[:, 0])* 1.5), width=0.03, facecolor='red', edgecolor='none')
    plt.plot(xy_solution[0], xy_solution[1], c=color)
    

plt.grid()
plt.show()
