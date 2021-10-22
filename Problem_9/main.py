import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from impl_9 import dif_solver

init_conds = np.array([1, 1], dtype=float)
x= np.linspace(0, np.pi, 30)
r = np.sin(x)

fig, axes = plt.subplots()
fig.subplots_adjust(bottom=0.25)

size = 0.25, 0.03
start = 0.1, 0.05
step = 0.4, 0.05

ax_ca = plt.axes([start[0], start[1] + step[1], *size])
ax_cb = plt.axes([start[0] + step[0], start[1] + step[1], *size])

slider_ca = None
slider_cb = None
min, max = -4., 4
alowed_vals = np.arange(min, max, 0.1)
alowed_vals = alowed_vals[alowed_vals != 0.0]


def update(*args):
    global x, r, axes, fig, sliders_ca, sliders_cb, min, max

    init_conds[0] = slider_ca.val
    init_conds[1] = slider_cb.val

    y = dif_solver(x, init_conds, r=r)
    axes.clear()
    axes.plot(x, y, label="TMA")
    axes.set_xticks([0, np.pi/2, np.pi])
    axes.set_xticklabels(["0", "π/2", "π"])
    axes.legend()
    axes.set_ylim(min, max)
    axes.grid()
    fig.canvas.draw_idle()


slider_ca= Slider(ax_ca, "y(a)", min, max, init_conds[0], valstep=alowed_vals)
slider_ca.on_changed(update)
slider_cb = Slider(ax_cb, "y(b)", min, max, init_conds[1], valstep=alowed_vals)
slider_cb.on_changed(update)

update()
plt.show()