import numpy as np
from matplotlib import pyplot as plt


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

xy_solution = runge_kutta_second_method(t_grid, functions, (1, 1.0))
plt.plot(xy_solution[0], xy_solution[1], 'k-o', markersize=4, linewidth=1, mfc='red', mec='red')
plt.show()
