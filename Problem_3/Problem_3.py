import numpy as np
import matplotlib.pyplot as plt
from integrals import integral_trapeze, integral_simpson
from scipy.stats import linregress

interval = (0, 1.0)
n = 15
interval_nums = np.array([2**i for i in range(1, n + 1)])

functions = [
    lambda x: 1 / (x**2 + 1),
    lambda x: np.exp(np.sin(x))
]
plt.figure(figsize=(14, 6))

for i, f in enumerate(functions):
    results = [
        [integral_trapeze(f, interval, i) for i in interval_nums],
        [integral_simpson(f, interval, i) for i in interval_nums],
    ]
    diffs = [np.array([np.abs(res[i + 1] - res[i]) for i in range(len(res) - 1)]) for res in results]
    names = ["Trapeze", "Simpson"]
    plt.subplot(1, 2, 1 + i)
    plt.title(f'Function: {i + 1}')

    for diff, name in zip(diffs, names):
        slope = linregress(np.log(interval_nums[1:]), np.log(diff))[0]
        #factor = np.round(np.log(diff[-3] / diff[0]) / np.log(interval_nums[-3] / interval_nums[1]), 1)
        plt.plot(interval_nums[1:], diff, label=f'{name}: {np.round(slope, 1)}')
    plt.legend()
    plt.xscale("log", base=2)
    plt.yscale("log")
    plt.grid()

plt.show()

