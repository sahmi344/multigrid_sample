import pandas as pd
import matplotlib.pyplot as plt

files = [
    "C1PROPER.csv",
    "C2.csv",
    "C3.csv",
    "C4.csv",
    "C5.csv"
]

plt.figure(figsize=(12, 5))

for file in files:
    data = pd.read_csv(file)

    time = data["timestamp"]
    signal = data["BIOZI"]

    plt.plot(time, signal, label=file)

plt.title("Bioimpedance Signals")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.legend()

plt.show()