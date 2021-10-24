import numpy as np
from matplotlib import pyplot as plt
from eigen import back_iteration_eigen_solver

n = 1000
x_left, x_right = -10, 10
x = np.linspace(x_left, x_right, n)
h = x[1] - x[0]

a = np.full(x.shape[0] - 1, -1 / (2 * h**2), dtype=float)
b = np.full(x.shape[0], 1 / h**2, dtype=float)
b += x**2 / 2
gen = np.random.RandomState(2)
d = np.ones_like(x)#gen.random_sample(x.shape)


eigenvec, eigenval =  back_iteration_eigen_solver(a, b, a, d, 10)

print(eigenval)

plt.plot(x, eigenvec)

gt = np.exp(-x**2 / 2)
gt /= np.linalg.norm(gt)
plt.plot(x, gt)

plt.plot()
plt.show()
