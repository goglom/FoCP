import numpy as np
import numpy.linalg as lin
import matplotlib
from matplotlib import pyplot as plt
from numba import jit


@jit(nopython=True)
def euler_implicit(x_grid: np.array, matrix: np.array, initital_cond: tuple[float, float]):
    y_vec = np.zeros((len(x_grid), 2))
    y_vec[0] = initital_cond

    for i in range(1, len(x_grid)):
        h = x_grid[i] - x_grid[i - 1]
        y_vec[i] = lin.inv(np.identity(2) - h * matrix) @ y_vec[i - 1]
    
    return np.swapaxes(y_vec, 0, 1)


@jit(nopython=True)
def euler_explicit(x_grid: np.array, matrix: np.array, initital_cond: tuple[float, float]):
    y_vec = np.zeros((2, len(x_grid)))
    y_vec[0][0] = initital_cond[0]
    y_vec[1][0] = initital_cond[1]
    

    for i in range(1, len(x_grid)):
        h = x_grid[i] - x_grid[i - 1]
        y_vec[0][i] = y_vec[0][i - 1] + h * (matrix[0, 0] * y_vec[0][i - 1] + matrix[0, 1] * y_vec[1][i - 1])
        y_vec[1][i] = y_vec[1][i - 1] + h * (matrix[1, 0] * y_vec[0][i - 1] + matrix[1, 1] * y_vec[1][i - 1])
        
    return y_vec#np.swapaxes(y_vec, 0, 1)


x = np.arange(0, 0.01, 1e-5)
mat = np.array([
    [998., 1998.],
    [-99., -1999.],
])
cond = (1, 1e-7)

y = euler_explicit(x, mat, cond)

plt.figure(figsize=(8, 6))
plt.plot(y[0], y[1])
# for i in range(2):
#     plt.subplot(2,1,1 + i)
#     plt.plot(x, y[i])
#     print(x[0], y[i][0])
#     plt.grid()

plt.show()