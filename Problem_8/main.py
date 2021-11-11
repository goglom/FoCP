from impl import euler_implicit, euler_explicit
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def ground_truth(t, cond):
    alpha = cond[1] + cond[0]
    beta = -2 * cond[1] - cond[0]
    u = 2 * alpha * np.exp(-t) + beta * np.exp(-1000 * t)
    v = -alpha * np.exp(-t) - beta * np.exp(-1000 * t)
    return np.stack([u, v])

mat = np.array([
    [998., 1998.],
    [-999., -1999.],
])
cond = [1., 1.]

fig, axes = plt.subplots(1, 3)
fig.subplots_adjust(bottom=0.25)

ax_time = plt.axes([0.25, 0.15, 0.65, 0.03])
ax_cond_u = plt.axes([0.25, 0.1, 0.65, 0.03])
ax_cond_v = plt.axes([0.25, 0.05, 0.65, 0.03])

allowed_n = np.concatenate([range(10, 100, 10), range(100, 1000 + 1, 100)])
allowed_cond = [0.1, 0.5, 1.0, 1.5, 2.0, 3.0]

slider_n = Slider(ax_time, "N", 10, 8000,valinit=1000, valstep=10, color="green")
slider_cond_u = Slider( ax_cond_u, "u cond", min(allowed_cond), max(allowed_cond),valinit=1., valstep=allowed_cond,color='red')
slider_cond_v = Slider(ax_cond_v, "v cond", min(allowed_cond), max(allowed_cond),valinit=1., valstep=allowed_cond,color='blue')

def update(*args):
    n =  slider_n.val
    t = np.concatenate([np.linspace(0.0, 0.01, int(n *0.7)), np.linspace(0.01, 1, int(n * 0.3))])
    cond[0] = slider_cond_u.val
    cond[1] = slider_cond_v.val

    eu_ex = euler_explicit(t, mat, cond)
    eu_im = euler_implicit(t, mat, cond)
    gt = ground_truth(t, cond)
    
    for ax in axes:
        ax.clear()

    axes[0].set_title("u(t)")
    axes[1].set_title("v(t)")
    axes[2].set_title("u(v)")

    axes[0].plot(t, eu_im[0], "s", ms=2, label="implicit")
    axes[0].plot(t, eu_ex[0], "s", ms=2, label="explicit")
    axes[0].plot(t, gt[0], label="ground truth")

    axes[1].plot(t, eu_im[1], "s", ms=2, label="implicit")
    axes[1].plot(t, eu_ex[1], "s", ms=2, label="explicit")
    axes[1].plot(t, gt[1], label="ground truth")

    axes[2].plot(eu_im[0], eu_im[1], "s", ms=2, label="implicit")
    axes[2].plot(eu_ex[0], eu_ex[1], "s", ms=2, label="explicit")
    axes[2].plot(gt[0], gt[1], label="ground truth")

    axes[0].set_ylim(np.min(gt[0]) -0.1, np.max(gt[0]) * 1.1)
    axes[1].set_ylim(np.min(gt[1]) -0.1, np.max(gt[1]) * 1.1)

    axes[2].set_xlim(np.min(gt[0]) -0.1, np.max(gt[0]) * 1.1)
    axes[2].set_ylim(np.min(gt[1]) -0.1, np.max(gt[1]) * 1.1)
    

    for ax in axes:
        ax.legend()
        ax.grid()

slider_n.on_changed(update)
slider_cond_u.on_changed(update)
slider_cond_v.on_changed(update)

update()

plt.show()
