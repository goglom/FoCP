import numpy as np
import itertools
import matplotlib.pyplot as plt
from numba import jit

@jit(nopython=True)
def solve(r, phi, dt, inner_cond, outer_cond, iters: int=1000):
    n_r = r.shape[0]
    n_phi = phi.shape[0]
    u = np.zeros((n_r, n_phi))
    u_next = np.zeros_like(u)

    for _ in range(iters):
        u_next[0] = inner_cond
        u_next[-1] = outer_cond

        for i in range(1, n_r - 1):
            d_r = r[i + 1] - r[i]

            for j in range(0, n_phi):
                
                j_next = (j + 1) % n_phi
                d_phi = phi[j_next] - phi[j]
                term1 = r[i + 1] / r[i] * (u[i + 1, j] - u[i, j]) / d_r**2  #(u[i + 1, j] - u[i, j]) / (d_r * r[i])
                term2 = -(u[i, j] - u[i - 1, j]) / d_r**2                   #(u[i + 1, j] - 2 * u[i, j] + u[i - 1, j]) / d_r**2
                term3 = (u[i, j_next] - 2 * u[i, j] + u[i, j - 1]) / (d_phi * r[i])**2
                u_next[i , j] = dt * (term1 + term2 + term3) + u[i, j]

        u = u_next
    return u


n_r = 80
n_phi = 90

a = 0.5
b = 1

r = np.linspace(a, b, n_r)
phi = np.linspace(0, 2 * np.pi, n_phi)
d_r = r[1] - r[0]
d_phi = phi[1] - phi[0]
dt = 0.5 / (1 / d_r**2 +  1 / d_phi**2)

K = 1.0
G = 1.0

inner_cond = K * np.sin(phi)
outer_cond = G


B_0 = G / (np.log(b) - np.log(a))
A_0 = -B_0 * np.log(a)
D_1 = K / (1 / a - a / b**2)
B_1 = -D_1 / b**2


r_g, phi_g = np.meshgrid(r, phi, indexing='ij')

u_gt = A_0 + B_0*np.log(r_g) + np.sin(phi_g)*(B_1 * r_g + D_1 / r_g)
#u = solve(r, phi, dt, inner_cond, outer_cond, 50000)
    

def draw(polar=True):
    proj = 'polar' if polar else None

    sols = [
        solve(r, phi, dt, inner_cond, outer_cond, 100),
        solve(r, phi, dt, inner_cond, outer_cond, 5000),
        solve(r, phi, dt, inner_cond, outer_cond, 50000)
    ]

    

    names = [
        '100',
        '5000',
        '100000'
    ]
    plt.figure(dpi=100, figsize=(10, 5))

    for i, (sol, name) in enumerate(zip(sols, names)):
        plt.subplot(1, len(sols), 1 + i, projection=proj)
        plt.title(name)
        if (polar):
            plt.pcolormesh(phi, r, sol, shading='auto')
        else:
            plt.imshow(sol,aspect='auto', interpolation="nearest")
            plt.colorbar()

    plt.tight_layout()
    plt.show()

draw()
