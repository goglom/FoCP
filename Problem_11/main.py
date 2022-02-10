import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
from eigen import back_iteration_eigen_solver

n = 1000
iters = 100
x_left, x_width = -10, 20


fig, ax = plt.subplots()
fig.subplots_adjust(bottom=0.25, left=0.1)

ax_itres = plt.axes([0.10, 0.05, 0.25, 0.03])
ax_n =  plt.axes([0.10, 0.10, 0.25, 0.03])
ax_xw = plt.axes([0.50, 0.10, 0.25, 0.03])

slider_iters = Slider(ax_itres, "Iters", 0, 50, iters, valstep=1)
slider_xw = Slider(ax_xw, "width", 1, 40, x_width, valstep=1)
slider_n = Slider(ax_n, "N", 2, 2000, n, valstep=1)

PLANK = 6.582e-16
MASS = 0.51 * 9.11e-31

def update(*args):
    iters = slider_iters.val
    x_width = slider_xw.val
    n = slider_n.val

    x = np.linspace(-x_width / 2, x_width / 2, n)
    h = (x[1] - x[0]) * 1e-9
    u = np.where(np.abs(x) < 2, 0, 0.3)

    factor = PLANK**2 / MASS

    a = np.full(x.shape[0] - 1, -factor /  (h**2 * 2), dtype=float) 
    b = np.full(x.shape[0], factor / h**2, dtype=float)
    b += u
    gen = np.random.RandomState(2)
    d = gen.random_sample(x.shape)

    eigenvec, eigenval =  back_iteration_eigen_solver(a, b, a, d, iters)
    gt = np.exp(-x**2 / 2)
    gt /= np.linalg.norm(gt)

    ax.clear()
    ax.set_title(f"Eigen value = {eigenval}")
    ax.plot(x, eigenvec, lw=4, label="my solution")
    #ax.plot(x, gt, label="real solution")
    #ax.set_ylim(0, 0.15)
    ax.legend()
    ax.grid()


slider_iters.on_changed(update)
slider_xw.on_changed(update)
slider_n.on_changed(update)

update()

plt.show()
