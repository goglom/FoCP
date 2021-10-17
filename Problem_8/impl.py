import numpy as np
import numpy.linalg as lin
from matplotlib import pyplot as plt
import matplotlib

def euler_implicit(x_grid: np.array, matrix: np.array, initital_cond: tuple[float, float]):
    y_vec = np.zeros((len(x_grid), 2))
    y_vec[0] = initital_cond

    for i in range(1, len(x_grid)):
        h = x_grid[i] - x_grid[i - 1]
        y_vec[i] = lin.inv(np.identity(2) - h * matrix) @ y_vec[i - 1]
    
    return np.swapaxes(y_vec, 0, 1)


def euler_explicit(x_grid: np.array, matrix: np.array, initital_cond: tuple[float, float]):
    y_vec = np.zeros((len(x_grid), len(initital_cond)))
    y_vec[0] = initital_cond
    
    for i in range(x_grid.shape[0] - 1):
        h = x_grid[i + 1] - x_grid[i]
        y_vec[i + 1] = y_vec[i] + h * matrix @ y_vec[i]
        
    return y_vec.swapaxes(0, 1)

def main():
    x = np.linspace(0, 1, 100)
    mat = np.array([
        [998., 1998.],
        [-999., -1999.],
    ])

    cond = (1, 1)
    y = euler_implicit(x, mat, cond)

    plt.figure(figsize=(8, 6))
    plt.plot(y[0], y[1])

    plt.figure(figsize=(8, 6))
    for i in range(2):
        plt.subplot(2,1,1 + i)
        plt.plot(x, y[i])
        plt.grid()

    plt.show()


if __name__ == "__main__":
    main()