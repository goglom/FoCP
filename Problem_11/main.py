"""
Kek, its main.py file
"""
import numpy as np
from tma import matrix_wrapper_tma
from eigen import back_iteration_eigen_solver

mat = np.array([
        [-2, 1, 0, 0],
        [1, -2, 1, 0],
        [0, 1, -2, 1],
        [0, 0, 1, -2]
    ], dtype=float)

print(
    back_iteration_eigen_solver(mat, np.array([-10, 1.6, -1.6, 0]), 0, 1e-11, matrix_wrapper_tma)
)
