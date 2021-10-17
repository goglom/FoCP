from impl import euler_implicit, euler_explicit

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from itertools import chain

def ground_truth(t, cond):
    alpha = cond[1] + cond[0]
    beta = -2 * cond[1] - cond[0]
    u = 2 * alpha * np.exp(-t) + beta * np.exp(-1000 * t)
    v = -alpha * np.exp(-t) - beta * np.exp(-1000 * t)
    return np.stack([u, v])

t = np.linspace(0, 1, 1000)
mat = np.array([
    [998., 1998.],
    [-999., -1999.],
])
cond = [1., 1.]
sols = euler_explicit(t, mat, cond), euler_implicit(t, mat, cond), ground_truth(t, cond)

fig, axes = plt.subplots(1, 3)
fig.subplots_adjust(bottom=0.25)
ut_lines = []
vt_lines = []
uv_lines = []


axes[0].set_title("u(t)")
axes[1].set_title("v(t)")
axes[2].set_title("u(v)")

ut_lines.append(axes[0].plot(t, sols[0][0], label="implicit")[0])
ut_lines.append(axes[0].plot(t, sols[1][0], label="explicit")[0])
ut_lines.append(axes[0].plot(t, sols[2][0], label="ground truth")[0])


vt_lines.append(axes[1].plot(t, sols[0][1], label="implicit")[0])
vt_lines.append(axes[1].plot(t, sols[1][1], label="explicit")[0])
vt_lines.append(axes[1].plot(t, sols[2][1], label="ground truth")[0])


uv_lines.append(axes[2].plot(sols[0][0], sols[0][1], label="implicit")[0])
uv_lines.append(axes[2].plot(sols[1][0], sols[1][1], label="explicit")[0])
uv_lines.append(axes[2].plot(sols[2][0], sols[2][1], label="ground truth")[0])

ax_time = plt.axes([0.25, 0.15, 0.65, 0.03])
ax_cond_u = plt.axes([0.25, 0.1, 0.65, 0.03])
ax_cond_v = plt.axes([0.25, 0.05, 0.65, 0.03])

allowed_n = np.concatenate([range(10, 100, 10), range(100, 1000 + 1, 100)])
allowed_cond = [0.1, 0.5, 1.0, 1.5, 2.0, 3.0]

slider_n = Slider(
    ax_time, "N", 10, 1000,
    valinit=1000, valstep=allowed_n,
    color="green"
)
slider_cond_u = Slider(
    ax_cond_u, "u cond", min(allowed_cond), max(allowed_cond),
    valinit=1., valstep=allowed_cond,
    color='red'
)
slider_cond_v = Slider(
    ax_cond_v, "v cond", min(allowed_cond), max(allowed_cond),
    valinit=1., valstep=allowed_cond,
    color='blue'
)

def update(val):
    t = np.linspace(0, 1., slider_n.val)
    cond[0] = slider_cond_u.val
    cond[1] = slider_cond_v.val

    sols = euler_explicit(t, mat, cond), euler_implicit(t, mat, cond), ground_truth(t, cond)
    x_datas = [
        t,t,t, 
        t,t,t, 
        sols[0][0], sols[1][0],sols[2][0]
        ]
    y_datas = [
        sols[0][0],sols[1][0], sols[2][0], 
        sols[0][1],sols[1][1], sols[2][1], 
        sols[0][1],sols[1][1], sols[2][1]
        ]

    for line, x, y in zip(chain(ut_lines, vt_lines, uv_lines), x_datas, y_datas):
        line.set_xdata(x)
        line.set_ydata(y)
    
    fig.canvas.draw_idle()

for ax in axes:
    ax.legend()

slider_n.on_changed(update)
slider_cond_u.on_changed(update)
slider_cond_v.on_changed(update)

plt.show()
