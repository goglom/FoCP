import numpy as np
import numpy.fft as fft
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def HenningWindow(x: np.array) -> np.array:
    result = np.zeros_like(x)
    N = x.shape[0]
    for i in range(N):
        result[i] = x[i] * 0.5 *(1 - np.cos((2 * np.pi * i) / (N - 1)))

    return result

kot_freq = 30
start = 0.7
width = 20 * np.pi

fig, axes = plt.subplots(1, 2)
fig.subplots_adjust(bottom=0.25, left=0.1)

ax_kfreq = plt.axes([0.10, 0.05, 0.25, 0.03])
ax_width =  plt.axes([0.10, 0.10, 0.25, 0.03])
ax_start = plt.axes([0.50, 0.10, 0.25, 0.03])

slider_kfreq = Slider(ax_kfreq, "Kot freq", 0, 100, kot_freq)
slider_start = Slider(ax_start, "start", 0, 10, start, valstep=0.1)
slider_width = Slider(ax_width, "width", 0.5, 20, width, valstep=0.5)

def update(*args):
    start = slider_start.val
    kot_freq = slider_kfreq.val
    width = slider_width.val

    t = np.arange(start, start + width, np.pi/ kot_freq)
    f = np.sin(5.1 * t) + 0.002 * np.sin(25.5 * t)

    data = [f, HenningWindow(f)]
    titles = ["Прямоугольное окно", "Окно Ханна"]    
    fft_data = np.abs(fft.fft(data))
    timestep = t[1] - t[0]
    freq = fft.fftfreq(t.size, d=timestep) * np.pi * 2
    mask = freq > 0
    freq = freq[mask]
    max_fft = np.max(fft_data)

    for ax in axes:
        ax.clear()

    for ff, title, ax in zip(fft_data, titles, axes):
        ff = ff[mask]
        ax.set_title(title)
        ax.grid()
        ax.plot(freq, ff, "-s")
        ax.set_xlabel("w")
        ax.set_ylabel("abs")

update()

for s in (slider_width, slider_kfreq, slider_start):
    s.on_changed(update)

plt.show()
