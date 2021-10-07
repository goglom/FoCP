import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from itertools import product, chain

def runge_kutta_second_method(x_grid: np.array, func: tuple, initial_conditions: tuple[float, float], alpha: float = 3/4):
    y_res = np.zeros((2, len(x_grid)))
    y_res[0][0] = initial_conditions[0]
    y_res[1][0] = initial_conditions[1]

    for i in range(1, len(x_grid)):
        x, y = x_grid[i - 1], (y_res[0][i - 1], y_res[1][i - 1])
        step = x_grid[i] - x_grid[i - 1]
        k = (func[0](x, y[0], y[1]), func[1](x, y[0], y[1]))
        factor = step / (2 * alpha)
        y_res[0][i] = y[0] + step * (
                (1 - alpha) * k[0] + alpha * func[0](x + factor, y[0] + factor * k[0], y[1] + factor * k[1]))
        y_res[1][i] = y[1] + step * (
                    (1 - alpha) * k[1] + alpha * func[1](x + factor, y[0] + factor * k[0], y[1] + factor * k[1]))

    return y_res


t_grid = np.linspace(0, 1, 200)
functions = (
    lambda t, x, y: 10 * x - 2 * x * y,
    lambda t, x, y: 2 * x * y - 10 * y
)

cond = range(1, 5 + 1)
conditions = [(x, y) for x, y in product(cond, cond)]

cm_subsection = np.linspace(0., 1., len(conditions)) 
colors = [ matplotlib.cm.jet(x) for x in cm_subsection ]

for cond, color in zip(conditions, colors):
    xy_solution = runge_kutta_second_method(t_grid, functions, cond)
    plt.plot(*cond, 'o', ms=4, c=color)
    plt.arrow(*cond, *((xy_solution[:, 1] - xy_solution[:, 0])* 2.5), width=0.05, facecolor='red', edgecolor='none')
    plt.plot(xy_solution[0], xy_solution[1], lw=2, c=color)
    

plt.grid()
plt.show()
