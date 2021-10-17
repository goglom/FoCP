import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from impl import tma_solver

init_conds_mat = np.array([
    [1, 0, 1],
    [1, 0, 1]
], dtype=float)

x= np.linspace(0, np.pi, 1000)
r = np.sin(x)

fig, axes = plt.subplots()
fig.subplots_adjust(bottom=0.25)

size = 0.25, 0.03
start = 0.1, 0.05
step = 0.4, 0.05
sliders_num = 3

ax_ca = [plt.axes([start[0], start[1] + step[1] * i, *size]) for i in range(sliders_num)]
ax_cb = [plt.axes([start[0] + step[0], start[1] + step[1] * i, *size]) for i in range(sliders_num)]

sliders_ca = []
sliders_cb = []
min, max = -10., 10
alowed_vals = np.arange(min, max, 0.1)
alowed_vals = alowed_vals[alowed_vals != 0.0]


def update(*args):
    global x, r, axes, fig, sliders_ca, sliders_cb

    for i, (ca, cb) in enumerate(zip(sliders_ca, sliders_cb)):
        init_conds_mat[0, i] = ca.val
        init_conds_mat[1, i] = cb.val

    y = tma_solver(x, init_conds_mat, r=r)
    axes.clear()
    axes.plot(x, y, label="TMA")
    axes.set_xticks([0, np.pi/2, np.pi])
    axes.set_xticklabels(["0", "π/2", "π"])
    axes.legend()
    axes.grid()
    fig.canvas.draw_idle()


for i in range(3):
    sliders_ca.append(
        Slider(ax_ca[i], f"left c {i}", min, max, init_conds_mat[0, i], valstep=alowed_vals)
    )
    sliders_ca[i].on_changed(update)
    sliders_cb.append(
        Slider(ax_cb[i], f"right c {i}", min, max, init_conds_mat[1, i], valstep=alowed_vals)
    )
    sliders_cb[i].on_changed(update)

update()
plt.show()