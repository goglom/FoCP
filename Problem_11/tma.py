import numpy as np


def tma_solver(a: np.array, b: np.array, c: np.array, d: np.array) -> np.array:
    assert a.ndim == 1 and b.ndim == 1 and c.ndim == 1 and d.ndim == 1
    assert a.shape[0] == c.shape[0] and a.shape[0] == b.shape[0] - 1
    n = d.shape[0]
    b_new = b.astype(dtype=float)
    d_new = d.astype(dtype=float)

    for i in range(1, n):
        factor = a[i - 1] / b_new[i - 1]
        b_new[i] = b_new[i] - factor * c[i - 1] 
        d_new[i] = d_new[i] - factor * d_new[i - 1] 

    x = b_new.astype(dtype=float)
    x[-1] = d_new[-1] / b_new[-1]

    for i in range(n - 2, 0 - 1, -1):
        x[i] = (d_new[i] - c[i] * x[i + 1]) / b_new[i]

    return x


def matrix_wrapper_tma(matrix: np.array, d: np.array) -> np.array:
    return tma_solver(np.diag(matrix, 1), np.diag(matrix, 0), np.diag(matrix, -1), d)


def main():
    """
    Haha-main...
    """
    mat = np.array([
        [-2, 1, 0, 0],
        [1, -2, 1, 0],
        [0, 1, -2, 1],
        [0, 0, 1, -2]
    ], dtype=float)
    d = np.array([1, 9, 4, 3])
    a = np.array([1, 1, 1.])
    c = np.array([1, 1, 1])
    b = np.array([-2, -2, -2, -2])

    my_sol = matrix_wrapper_tma(mat, d)
    np_sol = np.linalg.solve(mat, d)
    print(np_sol)
    print(my_sol)


if __name__ == '__main__':
    main()
