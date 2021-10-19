import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from impl_10 import heat_eq_solver

L = 1
phi = lambda x: 24/4 * x * (1 - x / L)**2
x = np.linspace(0, L, 1000)
t = np.linspace(0, 4, 64)
u_edge_cond = np.array([0., 0.])
u_t0 = phi(x)

fig, axis = plt.subplots()
fig.subplots_adjust(bottom=0.25, top=0.90, left=0.15, right=0.75)

size = 0.25, 0.03
start = 0.1, 0.05
step = 0.4, 0.05

ax_cond_left = plt.axes([0.15, 0.05, 0.4, 0.05])
ax_cond_right = plt.axes([0.15, 0.15, 0.4, 0.05])
ax_colorbar = plt.axes([0.80, 0.25, 0.05, 0.65])

slider_cond_left = None
slider_cond_right = None
min, max = 0, 1
allowed_T = np.linspace(min, max, 100)

image = None
def update(*args):
    u_edge_cond[0] = slider_cond_left.val
    u_edge_cond[1] = slider_cond_right.val

    u = heat_eq_solver(x, t, 1.0, u_t0, u_edge_cond)

    axis.clear()
    
    extent = [x[0], x[-1], t[0], t[-1]]
    image = axis.imshow(u, 'hot', aspect='auto', extent=extent)
    fig.colorbar(image, cax=ax_colorbar)
    
    fig.canvas.draw_idle()

slider_cond_left = Slider(ax_cond_left, "T(a)", min, max, u_edge_cond[0], valstep=allowed_T)
slider_cond_right = Slider(ax_cond_right, "T(b)", min, max, u_edge_cond[1], valstep=allowed_T) 

slider_cond_left.on_changed(update)
slider_cond_right.on_changed(update)

update()

plt.show()
