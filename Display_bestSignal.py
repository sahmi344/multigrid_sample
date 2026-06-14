import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ==========================================
# USER SETTINGS
# ==========================================

files = [
    "C1PROPER.csv",
    "C2.csv",
    "C3.csv",
    "C4.csv",
    "C5.csv"
]

Fs = 391  # Sampling Frequency

# Quality Score Weights
W_BANDPOWER = 0.6
W_SNR       = 0.3
W_BASELINE  = 0.1

# ==========================================
# STORAGE
# ==========================================

results = []
signals_store = {}

# ==========================================
# PROCESS ALL SIGNALS
# ==========================================

for file in files:

    data = pd.read_csv(file)

    signal = data["BIOZI"].values

    signals_store[file] = signal

    # Remove DC Component
    signal_dc = signal - np.mean(signal)

    N = len(signal_dc)

    fft_values = np.fft.fft(signal_dc)
    fft_freqs = np.fft.fftfreq(N, d=1/Fs)

    positive_mask = fft_freqs >= 0

    freqs = fft_freqs[positive_mask]
    fft_mag = np.abs(fft_values[positive_mask])

    fft_power = fft_mag ** 2

    # Desired Frequency Band
    band_mask = (freqs >= 0.3) & (freqs <= 50.0)

    signal_energy = np.sum(fft_power[band_mask])

    noise_energy = np.sum(fft_power[~band_mask])

    if noise_energy < 1e-12:
        noise_energy = 1e-12

    # Band Power Ratio
    band_power_ratio = (
        signal_energy /
        (signal_energy + noise_energy)
    )

    # SNR
    snr = 10 * np.log10(
        signal_energy / noise_energy
    )

    # Baseline Drift
    baseline_drift = np.std(signal)

    results.append({

        "file": file,

        "signal_energy": signal_energy,

        "noise_energy": noise_energy,

        "band_power_ratio": band_power_ratio,

        "snr": snr,

        "baseline_drift": baseline_drift
    })

# ==========================================
# NORMALIZATION
# ==========================================

def normalize(x):

    xmin = np.min(x)
    xmax = np.max(x)

    if xmax - xmin < 1e-12:
        return np.ones_like(x)

    return (x - xmin) / (xmax - xmin)

band_ratios = np.array(
    [r["band_power_ratio"] for r in results]
)

snrs = np.array(
    [r["snr"] for r in results]
)

baselines = np.array(
    [r["baseline_drift"] for r in results]
)

band_norm = normalize(band_ratios)

snr_norm = normalize(snrs)

baseline_norm = normalize(baselines)

# Lower Baseline Drift = Better
baseline_score = 1 - baseline_norm

# ==========================================
# QUALITY SCORE
# ==========================================

scores = (
    W_BANDPOWER * band_norm +
    W_SNR * snr_norm +
    W_BASELINE * baseline_score
)

scores = scores * 100

for i in range(len(results)):
    results[i]["quality_score"] = scores[i]

# ==========================================
# SORT RESULTS
# ==========================================

results_sorted = sorted(
    results,
    key=lambda x: x["quality_score"],
    reverse=True
)

best_file = results_sorted[0]["file"]

# ==========================================
# PRINT RESULTS
# ==========================================

print("\n")
print("=" * 60)
print(" SIGNAL QUALITY RANKING ")
print("=" * 60)

for rank, r in enumerate(results_sorted, start=1):

    print(
        f"{rank}. "
        f"{r['file']:12s} "
        f"Score={r['quality_score']:.2f} "
        f"SNR={r['snr']:.2f} dB "
        f"BandRatio={r['band_power_ratio']:.4f}"
    )

print("\nBEST SIGNAL =", best_file)

# ==========================================
# DASHBOARD
# ==========================================

fig = plt.figure(figsize=(18, 12))

gs = GridSpec(4, 3, figure=fig)

fig.suptitle(
    "BIOIMPEDANCE SIGNAL QUALITY ASSESSMENT DASHBOARD",
    fontsize=20,
    fontweight='bold'
)

colors = [
    'blue',
    'green',
    'orange',
    'red',
    'purple'
]

# ==========================================
# FFT PLOTS
# ==========================================

for idx, file in enumerate(files):

    row = idx // 3
    col = idx % 3

    ax = fig.add_subplot(gs[row, col])

    data = pd.read_csv(file)

    signal = data["BIOZI"].values

    signal = signal - np.mean(signal)

    N = len(signal)

    fft_values = np.fft.fft(signal)

    fft_freqs = np.fft.fftfreq(
        N,
        d=1/Fs
    )

    positive_mask = fft_freqs >= 0

    freqs = fft_freqs[positive_mask]

    fft_mag = np.abs(
        fft_values[positive_mask]
    )

    ax.plot(
        freqs,
        fft_mag,
        color=colors[idx]
    )

    ax.axvspan(
        0.3,
        3,
        color='lightgreen',
        alpha=0.4
    )

    ax.set_xlim(0, 6)

    ax.set_title(file)

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude")

    ax.grid(True)

# ==========================================
# METRICS TABLE
# ==========================================

ax_table = fig.add_subplot(gs[1, 2])

ax_table.axis('off')

table_data = []

for r in results_sorted:

    table_data.append([
        r["file"],
        f"{r['quality_score']:.1f}",
        f"{r['snr']:.2f}",
        f"{r['band_power_ratio']:.3f}"
    ])

table = ax_table.table(
    cellText=table_data,
    colLabels=[
        "Signal",
        "Score",
        "SNR(dB)",
        "BandRatio"
    ],
    loc='center'
)

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.5)

ax_table.set_title(
    "Performance Metrics"
)

# ==========================================
# BAR CHART
# ==========================================

ax_bar = fig.add_subplot(gs[2, :])

files_sorted = [
    r["file"]
    for r in results_sorted
]

scores_sorted = [
    r["quality_score"]
    for r in results_sorted
]

bar_colors = []

best_signal = results_sorted[0]["file"]
worst_signal = results_sorted[-1]["file"]

for f in files_sorted:

    if f == best_signal:
        bar_colors.append("gold")

    elif f == worst_signal:
        bar_colors.append("firebrick")

    else:
        bar_colors.append("steelblue")

bars = ax_bar.bar(
    files_sorted,
    scores_sorted,
    color=bar_colors
)

for bar in bars:

    height = bar.get_height()

    ax_bar.text(
        bar.get_x() + bar.get_width()/2,
        height + 1,
        f"{height:.1f}",
        ha='center'
    )

ax_bar.set_title(
    "Quality Score Ranking"
)

ax_bar.set_ylabel(
    "Score"
)

ax_bar.grid(
    True,
    axis='y'
)

# ==========================================
# BEST SIGNAL
# ==========================================

ax_best = fig.add_subplot(gs[3, 0:2])

best_waveform = signals_store[
    best_signal
]

ax_best.plot(
    best_waveform,
    color='goldenrod',
    linewidth=1.5
)

ax_best.set_title(
    f"BEST SIGNAL : {best_signal}"
)

ax_best.set_xlabel(
    "Samples"
)

ax_best.set_ylabel(
    "Amplitude"
)

ax_best.grid(True)

# ==========================================
# SUMMARY PANEL
# ==========================================

ax_summary = fig.add_subplot(gs[3, 2])

ax_summary.axis('off')

best = results_sorted[0]

summary_text = (
    "🏆 BEST CONFIGURATION\n\n"
    f"Signal : {best['file']}\n\n"
    f"Quality Score : {best['quality_score']:.2f}/100\n\n"
    f"SNR : {best['snr']:.2f} dB\n\n"
    f"Band Ratio : {best['band_power_ratio']:.3f}\n\n"
    f"Signal Energy : {best['signal_energy']:.2e}\n\n"
    f"Noise Energy : {best['noise_energy']:.2e}"
)

ax_summary.text(
    0.02,
    0.95,
    summary_text,
    fontsize=12,
    verticalalignment='top',
    bbox=dict(
        boxstyle="round",
        facecolor="lightyellow",
        alpha=0.9
    )
)

# ==========================================
# SAVE
# ==========================================

plt.tight_layout()

plt.savefig(
    "Bioimpedance_Quality_Dashboard.png",
    dpi=300,
    bbox_inches='tight'
)

plt.show()