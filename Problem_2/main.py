import numpy as np
import matplotlib.pyplot as plt


def sign_compare(x, y):
    return np.sign(x) == np.sign(y)


def dichotomy(func, l_bound, r_bound, exp_error):
    if func(l_bound) == 0:
        return l_bound
    if func(r_bound) == 0:
        return r_bound

    bounds = [l_bound, r_bound]
    middle_x = np.mean(bounds)
    middle_val = func(middle_x)
    lb_val = func(bounds[0])
    steps = [middle_x]

    if sign_compare(lb_val, func(r_bound)):
        raise RuntimeError(f'Bad range for dichotomy method: f(a) * f(b) > 0')

    while np.abs(middle_val) >= exp_error:
        if np.abs(middle_val) == 0:
            break
        elif not sign_compare(lb_val, middle_val):
            bounds[1] = middle_x
        else:
            bounds[0] = middle_x
            lb_val = func(bounds[0])

        middle_x = np.mean(bounds)
        steps.append(middle_x)
        middle_val = func(middle_x)

    return middle_x, np.array(steps)


def iteration_method(func, x0, exp_err, recurrent_eq):
    steps = [x0]
    iters = 0
    cur_x = x0
    cur_val = func(cur_x)
    while np.abs(cur_val) >= exp_err:
        if iters > 5000:
            raise RuntimeError('Iterations method does not convergence by 5000 iterations')

        cur_x = recurrent_eq(cur_x)
        steps.append(cur_x)
        cur_val = func(cur_x)
        iters += 1
    return cur_x, np.array(steps)


def derivative(func, point, delta):
    return (func(point + delta) - func(point)) / delta


def newton_method(func, f_deriv, x0, exp_err):
    def rec_eq(x_n):
        return x_n - func(x_n) / f_deriv(x_n)

    return iteration_method(func, x0, exp_err, rec_eq)


def simple_iterations_method(func, factor, x0, exp_err):
    def req_eq(x_n):
        return x_n - factor * func(x_n)

    return iteration_method(func, x0, exp_err, req_eq)


def main():
    a = 1
    u0 = 45

    delta = 1e-10

    f = lambda x: 1 / np.tan(np.sqrt(a ** 2 * u0 * (1 - x))) - np.sqrt(1 / x - 1)
    deriv_f = lambda x: (f(x + delta) - f(x)) / delta

    eps = 1e-10
    left_bound = 1 - (np.pi / a) ** 2 / u0 + eps
    if left_bound < 0:
        left_bound = eps
    right_bound = 1 - eps

    solutions = [
        dichotomy(f, left_bound, right_bound, eps),
        simple_iterations_method(f, 0.05, (left_bound + right_bound) / 2, eps),
        newton_method(f, deriv_f, (left_bound + right_bound) / 2, eps)
    ]
    names = ["Dichotomy", "Simple iterations", "Newton"]
    x = np.linspace(eps, right_bound, 1000)
    y = f(x)
    plt.figure(figsize=(16, 6))

    for i in range(len(solutions)):
        solution, iters = solutions[i]
        print(solution)

        plt.subplot(1, 3, i + 1)
        plt.title(names[i] + f", iters num: {len(iters)}")
        plt.plot(x, y)
        plt.plot(iters, f(iters), 'o')

        plt.ylim(-4, 4)
        plt.grid()

    plt.show()


if __name__ == "__main__":
    main()
