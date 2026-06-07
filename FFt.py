import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

files = [
    "C1PROPER.csv",
    "C2.csv",
    "C3.csv",
    "C4.csv",
    "C5.csv"
]

Fs = 392

fig, axes = plt.subplots(3, 2, figsize=(14, 10))

axes = axes.flatten()

for i, file in enumerate(files):

    data = pd.read_csv(file)

    signal = data["BIOZI"].values

    # Remove DC component
    signal = signal - np.mean(signal)

    # FFT
    N = len(signal)

    fft_values = np.fft.fft(signal)
    fft_freqs = np.fft.fftfreq(N, d=1/Fs)

    positive_freqs = fft_freqs[:N//2]
    positive_fft = np.abs(fft_values[:N//2])

    axes[i].plot(positive_freqs, positive_fft)

    axes[i].set_xlim(0, 6)
    axes[i].set_title(f"FFT - {file}")
    axes[i].set_xlabel("Frequency (Hz)")
    axes[i].set_ylabel("Magnitude")
    axes[i].grid(True)

plt.tight_layout()
plt.show()