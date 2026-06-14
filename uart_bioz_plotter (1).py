"""
Real-Time UART BioZ Signal Plotter + Signal Processing
=======================================================
Data Format : config,timestamp,bioz_value
Example     : 1,1234,2048

ADC         : 12-bit  (0-4095) → mV = raw/4095 * 3300
Sampling    : 500 sps
Buffer      : 15 000 samples per config (30 s)
Processing  : triggers automatically when each config buffer fills

Metrics
-------
  Band Power Ratio  : power in 0.3–3 Hz / total power  (FFT + Hanning window)
  Peak-to-Peak      : max – min in mV  (hard-fail if < 300 mV)

Quality Score (per config)
--------------------------
  norm_BPR  = min-max across all processed configs so far
  norm_P2P  = min-max across all processed configs so far
  score     = 0.5 * norm_BPR + 0.5 * norm_P2P
  score     = 0  if P2P < 300 mV  (hard penalty)

Best Config : argmax(score) after all 6 complete
"""

import sys, threading, collections, time, math
import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
import serial
import serial.tools.list_ports

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# ── Constants ─────────────────────────────────────────────────────────────────

NUM_CONFIGS   = 6
FS            = 500                      # samples per second
BUFFER_SIZE   = FS * 5                  # 15 000 samples = 30 s
ADC_MAX       = 4095
ADC_VREF      = 3300.0                   # mV
BAND_LO       = 45                      # Hz
BAND_HI       = 55                      # Hz
P2P_THRESHOLD = 300.0                    # mV
W_BPR         = 0.5
W_P2P         = 0.5
UPDATE_MS     = 50
BAUD_DEFAULT  = 115200

CONFIG_COLORS = {
    1: "#E63946", 2: "#2196F3", 3: "#4CAF50",
    4: "#FF9800", 5: "#9C27B0", 6: "#00BCD4",
}

# ── Shared state ──────────────────────────────────────────────────────────────

data_lock   = threading.Lock()

# Live plot buffer (current config only)
live_ts     = collections.deque(maxlen=500)
live_bz     = collections.deque(maxlen=500)

# Per-config processing buffers
proc_buf    = {i: [] for i in range(1, NUM_CONFIGS + 1)}   # raw ADC
buf_done    = {i: False for i in range(1, NUM_CONFIGS + 1)}

# Results per config (filled after processing)
results     = {}   # cfg -> dict(bpr, p2p_mv, norm_bpr, norm_p2p, score, passed_p2p)

active_cfg  = [None]
cfg_changed = [False]
running     = [False]

# Queue for processing jobs (cfg number)
proc_queue  = collections.deque()

# ── Signal Processing ─────────────────────────────────────────────────────────

def compute_metrics(raw_samples):
    """Returns (band_power_ratio, peak_to_peak_mV)."""
    sig = np.array(raw_samples, dtype=np.float64)
    sig_mv = sig / ADC_MAX * ADC_VREF

    # Peak-to-peak
    p2p = float(np.max(sig_mv) - np.min(sig_mv))

    # Band power ratio via FFT with Hanning window
    n      = len(sig_mv)
    window = np.hanning(n)
    sig_w  = (sig_mv - np.mean(sig_mv)) * window   # remove DC, apply window
    fft_mag = np.abs(np.fft.rfft(sig_w))
    power   = fft_mag ** 2
    freqs   = np.fft.rfftfreq(n, d=1.0 / FS)

    band_mask  = (freqs >= BAND_LO) & (freqs <= BAND_HI)
    p_band     = float(np.sum(power[band_mask]))
    p_total    = float(np.sum(power))
    bpr        = p_band / p_total if p_total > 0 else 0.0

    return bpr, p2p


def recompute_scores():
    """Min-max normalise across all processed configs and recalculate scores."""
    if not results:
        return
    bpr_vals = [v["bpr"]   for v in results.values()]
    p2p_vals = [v["p2p_mv"] for v in results.values()]

    bpr_min, bpr_max = min(bpr_vals), max(bpr_vals)
    p2p_min, p2p_max = min(p2p_vals), max(p2p_vals)

    for cfg, v in results.items():
        norm_bpr = (v["bpr"]    - bpr_min) / (bpr_max - bpr_min + 1e-12)
        norm_p2p = (v["p2p_mv"] - p2p_min) / (p2p_max - p2p_min + 1e-12)
        score    = W_BPR * norm_bpr + W_P2P * norm_p2p
        if not v["passed_p2p"]:
            score = 0.0
        v["norm_bpr"] = norm_bpr
        v["norm_p2p"] = norm_p2p
        v["score"]    = score


def processing_worker():
    """Background thread: pops configs from proc_queue and processes them."""
    while running[0]:
        if proc_queue:
            cfg = proc_queue.popleft()
            with data_lock:
                samples = list(proc_buf[cfg])

            bpr, p2p = compute_metrics(samples)
            passed   = p2p >= P2P_THRESHOLD

            with data_lock:
                results[cfg] = dict(
                    bpr=bpr, p2p_mv=p2p,
                    norm_bpr=0.0, norm_p2p=0.0,
                    score=0.0, passed_p2p=passed
                )
                recompute_scores()

            print(f"[DSP] Config {cfg} → BPR={bpr:.4f}  P2P={p2p:.1f}mV  "
                  f"Pass={passed}")
        else:
            time.sleep(0.05)


# ── Serial / Demo readers ─────────────────────────────────────────────────────

def _ingest(cfg, ts, raw):
    """Called from reader threads. Thread-safe ingestion."""
    with data_lock:
        changed = (cfg != active_cfg[0])
        if changed:
            live_ts.clear()
            live_bz.clear()
            active_cfg[0]  = cfg
            cfg_changed[0] = True

        live_ts.append(ts)
        live_bz.append(raw / ADC_MAX * ADC_VREF)

        if not buf_done[cfg]:
            proc_buf[cfg].append(raw)
            if len(proc_buf[cfg]) >= BUFFER_SIZE:
                buf_done[cfg] = True
                proc_queue.append(cfg)
                print(f"[BUF] Config {cfg} buffer full → queued for DSP")


def serial_reader(port_name, baud):
    try:
        ser = serial.Serial(port_name, baud, timeout=1)
    except serial.SerialException as exc:
        print(f"[ERROR] {exc}")
        running[0] = False
        return

    buf = ""
    while running[0]:
        try:
            raw = ser.read(ser.in_waiting or 1)
            if raw:
                buf += raw.decode("ascii", errors="ignore")
                while "\n" in buf:
                    line, buf = buf.split("\n", 1)
                    parts = line.strip().split(",")
                    if len(parts) != 3:
                        continue
                    try:
                        cfg, ts, bval = int(parts[0]), int(parts[1]), float(parts[2])
                    except ValueError:
                        continue
                    if cfg not in range(1, NUM_CONFIGS + 1):
                        continue
                    _ingest(cfg, ts, bval)
        except serial.SerialException:
            break
    ser.close()


def demo_reader():
    t = 0
    for cfg in range(1, NUM_CONFIGS + 1):
        # Give each config a distinct signal character for demo variety
        amp   = 400 + cfg * 80
        noise = 20 + cfg * 5
        for _ in range(BUFFER_SIZE + 20):
            if not running[0]:
                return
            mv  = 1650 + amp * math.sin(2 * math.pi * 1.2 * t / FS) + \
                         50  * math.sin(2 * math.pi * 5.0 * t / FS)  + \
                         np.random.normal(0, noise)
            raw = int(np.clip(mv / ADC_VREF * ADC_MAX, 0, ADC_MAX))
            _ingest(cfg, t * 2, raw)   # ts in ms (500sps → 2 ms/sample)
            t += 1
            time.sleep(1 / FS)

# ── GUI ───────────────────────────────────────────────────────────────────────

class BioZPlotter(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Real-Time BioZ Signal Plotter + Quality Analyser")
        self.configure(bg="#1A1A2E")
        self.resizable(True, True)
        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── UI build ──────────────────────────────────────────────────────────────

    def _build_ui(self):
        s = ttk.Style(self)
        s.theme_use("clam")
        s.configure("D.TFrame",    background="#1A1A2E")
        s.configure("D.TLabel",    background="#1A1A2E", foreground="#E0E0E0",
                    font=("Segoe UI", 10))
        s.configure("H.TLabel",    background="#1A1A2E", foreground="#4FC3F7",
                    font=("Segoe UI", 11, "bold"))
        s.configure("D.TButton",   background="#0F3460", foreground="#E0E0E0",
                    font=("Segoe UI", 10, "bold"), borderwidth=0)
        s.configure("Stp.TButton", background="#7B0000", foreground="#E0E0E0",
                    font=("Segoe UI", 10, "bold"), borderwidth=0)
        s.configure("D.TCombobox", fieldbackground="#0F3460", foreground="#E0E0E0",
                    background="#0F3460")
        s.configure("D.TEntry",    fieldbackground="#0F3460", foreground="#E0E0E0")

        # ── Control bar ───────────────────────────────────────────────────────
        ctrl = ttk.Frame(self, style="D.TFrame", padding=(10, 7))
        ctrl.pack(fill="x", side="top")

        ttk.Label(ctrl, text="Port:", style="D.TLabel").pack(side="left", padx=(0, 3))
        self.port_var = tk.StringVar()
        ports = [p.device for p in serial.tools.list_ports.comports()]
        cb = ttk.Combobox(ctrl, textvariable=self.port_var, values=ports,
                          width=13, style="D.TCombobox")
        cb.pack(side="left", padx=(0, 8))
        if ports: cb.current(0)

        ttk.Label(ctrl, text="Baud:", style="D.TLabel").pack(side="left", padx=(0, 3))
        self.baud_var = tk.StringVar(value=str(BAUD_DEFAULT))
        ttk.Entry(ctrl, textvariable=self.baud_var, width=8,
                  style="D.TEntry").pack(side="left", padx=(0, 8))

        self.start_btn = ttk.Button(ctrl, text="▶ Start",  style="D.TButton",
                                    command=self._start_serial)
        self.start_btn.pack(side="left", padx=3)
        self.demo_btn  = ttk.Button(ctrl, text="⚡ Demo",  style="D.TButton",
                                    command=self._start_demo)
        self.demo_btn.pack(side="left", padx=3)
        self.stop_btn  = ttk.Button(ctrl, text="■ Stop",   style="Stp.TButton",
                                    command=self._stop, state="disabled")
        self.stop_btn.pack(side="left", padx=3)
        ttk.Button(ctrl, text="🗑 Clear", style="D.TButton",
                   command=self._clear).pack(side="left", padx=3)

        # Config progress badges
        badge_f = ttk.Frame(ctrl, style="D.TFrame")
        badge_f.pack(side="right", padx=8)
        self.badges = {}
        for i in range(1, NUM_CONFIGS + 1):
            lbl = tk.Label(badge_f, text=f" C{i} ", bg="#1A1A2E",
                           fg=CONFIG_COLORS[i], font=("Segoe UI", 9, "bold"),
                           padx=5, pady=2)
            lbl.pack(side="left", padx=2)
            self.badges[i] = lbl

        # ── Figure: 2 rows — live signal (top) + scores bar (bottom) ─────────
        self.fig = plt.figure(figsize=(14, 8), facecolor="#0D0D1A")
        gs  = gridspec.GridSpec(2, 1, figure=self.fig,
                                height_ratios=[2.2, 1], hspace=0.45)

        # Live signal axes
        self.ax_sig = self.fig.add_subplot(gs[0])
        self._style_ax(self.ax_sig)
        self.ax_sig.set_xlabel("Timestamp (ms)", color="#9E9E9E", fontsize=9)
        self.ax_sig.set_ylabel("BioZ (mV)",      color="#9E9E9E", fontsize=9)
        (self.sig_line,) = self.ax_sig.plot([], [], color="#E0E0E0", linewidth=1.6)
        self.sig_wm = self.ax_sig.text(0.01, 0.95, "", transform=self.ax_sig.transAxes,
                                       fontsize=24, fontweight="bold", alpha=0.12,
                                       color="#FFFFFF", va="top")

        # Score bar axes
        self.ax_bar = self.fig.add_subplot(gs[1])
        self._style_ax(self.ax_bar)
        self.ax_bar.set_title("Quality Scores (updates after each config)", 
                              color="#9E9E9E", fontsize=9, pad=6)
        self.ax_bar.set_ylabel("Score", color="#9E9E9E", fontsize=9)
        self.ax_bar.set_xticks(range(1, NUM_CONFIGS + 1))
        self.ax_bar.set_xticklabels([f"C{i}" for i in range(1, NUM_CONFIGS + 1)],
                                    color="#9E9E9E")
        self.ax_bar.set_xlim(0.4, NUM_CONFIGS + 0.6)
        self.ax_bar.set_ylim(0, 1.05)
        self.bar_rects = self.ax_bar.bar(
            range(1, NUM_CONFIGS + 1),
            [0] * NUM_CONFIGS,
            color=[CONFIG_COLORS[i] for i in range(1, NUM_CONFIGS + 1)],
            alpha=0.25, width=0.55, zorder=2
        )
        self.bar_labels = [
            self.ax_bar.text(i, 0.02, "", ha="center", va="bottom",
                             color="#E0E0E0", fontsize=8)
            for i in range(1, NUM_CONFIGS + 1)
        ]
        # Best config annotation
        self.best_ann = self.ax_bar.text(
            0.5, 0.88, "", transform=self.ax_bar.transAxes,
            ha="center", color="#FFD700", fontsize=10, fontweight="bold"
        )

        canvas = FigureCanvasTkAgg(self.fig, master=self)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=(0, 2))
        self.canvas = canvas

        tb_frame = ttk.Frame(self, style="D.TFrame")
        tb_frame.pack(fill="x", side="bottom")
        NavigationToolbar2Tk(canvas, tb_frame)

        # ── Results table (hidden until processing starts) ────────────────────
        self.table_frame = ttk.Frame(self, style="D.TFrame", padding=(8, 4))
        self.table_frame.pack(fill="x", side="bottom")
        self._build_table()

        # Status bar
        self.status_var = tk.StringVar(value="Ready — connect a port or run the demo.")
        ttk.Label(self, textvariable=self.status_var, style="D.TLabel",
                  anchor="w", padding=(10, 2)).pack(fill="x", side="bottom")

        self._ani = animation.FuncAnimation(
            self.fig, self._animate, interval=UPDATE_MS,
            blit=False, cache_frame_data=False
        )

    def _style_ax(self, ax):
        ax.set_facecolor("#0D0D1A")
        ax.tick_params(colors="#9E9E9E", labelsize=8)
        for sp in ax.spines.values():
            sp.set_color("#333366")
        ax.grid(True, color="#1E1E3A", linewidth=0.7, linestyle="--", zorder=0)

    def _build_table(self):
        cols = ("Config", "BPR", "P2P (mV)", "P2P Pass", "Norm BPR", "Norm P2P", "Score")
        self.tv = ttk.Treeview(self.table_frame, columns=cols, show="headings",
                               height=6, style="D.Treeview")
        widths  = [60, 70, 80, 70, 80, 80, 70]
        for col, w in zip(cols, widths):
            self.tv.heading(col, text=col)
            self.tv.column(col, width=w, anchor="center")

        style = ttk.Style()
        style.configure("D.Treeview", background="#0D0D1A", foreground="#E0E0E0",
                        fieldbackground="#0D0D1A", font=("Consolas", 9),
                        rowheight=20)
        style.configure("D.Treeview.Heading", background="#0F3460",
                        foreground="#E0E0E0", font=("Segoe UI", 9, "bold"))
        style.map("D.Treeview", background=[("selected", "#1A3A6E")])

        self.tv.pack(fill="x", side="left", expand=True)
        self._table_rows = {}   # cfg -> iid

    # ── Animation ─────────────────────────────────────────────────────────────

    def _animate(self, _frame):
        with data_lock:
            xs      = list(live_ts)
            ys      = list(live_bz)
            cfg     = active_cfg[0]
            snap_r  = dict(results)
            buf_progress = {i: len(proc_buf[i]) for i in range(1, NUM_CONFIGS + 1)}
            done_flags   = dict(buf_done)

        # ── Live signal ───────────────────────────────────────────────────────
        if cfg is not None:
            color = CONFIG_COLORS[cfg]
            self.sig_line.set_data(xs, ys)
            self.sig_line.set_color(color)
            self.ax_sig.set_title(
                f"Live Signal — Config {cfg}  "
                f"({'Buffer full — processing done' if done_flags[cfg] else f'Buffer: {buf_progress[cfg]}/{BUFFER_SIZE} samples'})",
                color=color, fontsize=10, pad=8
            )
            self.sig_wm.set_text(f"C{cfg}")
            self.sig_wm.set_color(color)
            if xs:
                self.ax_sig.relim()
                self.ax_sig.autoscale_view()

        # ── Badge update ──────────────────────────────────────────────────────
        for i, lbl in self.badges.items():
            if done_flags[i]:
                lbl.config(bg=CONFIG_COLORS[i], fg="#0D0D1A",
                           font=("Segoe UI", 9, "bold"), text=f" C{i}✓ ")
            elif i == cfg:
                lbl.config(bg=CONFIG_COLORS[i], fg="#0D0D1A",
                           font=("Segoe UI", 9, "bold"), text=f" C{i} ")
            else:
                lbl.config(bg="#1A1A2E", fg=CONFIG_COLORS[i],
                           font=("Segoe UI", 9), text=f" C{i} ")

        # ── Score bar + table ─────────────────────────────────────────────────
        if snap_r:
            best_cfg  = max(snap_r, key=lambda c: snap_r[c]["score"])
            best_score = snap_r[best_cfg]["score"]

            for i, rect in enumerate(self.bar_rects):
                c = i + 1
                if c in snap_r:
                    sc = snap_r[c]["score"]
                    rect.set_height(sc)
                    rect.set_alpha(1.0 if c == best_cfg else 0.55)
                    # gold border for best
                    rect.set_edgecolor("#FFD700" if c == best_cfg else "none")
                    rect.set_linewidth(2 if c == best_cfg else 0)
                    self.bar_labels[i].set_text(f"{sc:.3f}")
                    self.bar_labels[i].set_y(sc + 0.01)
                else:
                    rect.set_height(0)
                    rect.set_alpha(0.2)
                    self.bar_labels[i].set_text("")

            # Best annotation
            if len(snap_r) == NUM_CONFIGS:
                self.best_ann.set_text(
                    f"★ Best Configuration: C{best_cfg}  "
                    f"(Score {best_score:.3f})"
                )
            else:
                self.best_ann.set_text(
                    f"Processed {len(snap_r)}/{NUM_CONFIGS} configs …"
                )

            # Table
            self._refresh_table(snap_r, best_cfg if len(snap_r) == NUM_CONFIGS else None)

        # Status bar
        if cfg is not None:
            pct = min(100, buf_progress[cfg] * 100 // BUFFER_SIZE)
            done_count = sum(1 for v in done_flags.values() if v)
            self.status_var.set(
                f"Config {cfg} — buffer {pct}%  |  "
                f"Configs processed: {done_count}/{NUM_CONFIGS}"
                + (f"  |  Latest: {ys[-1]:.1f} mV" if ys else "")
            )

    def _refresh_table(self, snap_r, best_cfg):
        for cfg in range(1, NUM_CONFIGS + 1):
            if cfg not in snap_r:
                continue
            v   = snap_r[cfg]
            tag = "best" if cfg == best_cfg else ("fail" if not v["passed_p2p"] else "")
            row = (
                f"C{cfg}",
                f"{v['bpr']:.4f}",
                f"{v['p2p_mv']:.1f}",
                "✓" if v["passed_p2p"] else "✗",
                f"{v['norm_bpr']:.4f}",
                f"{v['norm_p2p']:.4f}",
                f"{v['score']:.4f}",
            )
            if cfg in self._table_rows:
                self.tv.item(self._table_rows[cfg], values=row, tags=(tag,))
            else:
                iid = self.tv.insert("", "end", values=row, tags=(tag,))
                self._table_rows[cfg] = iid

        self.tv.tag_configure("best", foreground="#FFD700")
        self.tv.tag_configure("fail", foreground="#888888")

    # ── Controls ──────────────────────────────────────────────────────────────

    def _start_serial(self):
        port = self.port_var.get().strip()
        if not port:
            messagebox.showerror("No Port", "Please select a serial port.")
            return
        try:
            baud = int(self.baud_var.get())
        except ValueError:
            messagebox.showerror("Bad Baud", "Enter a valid baud rate.")
            return
        self._clear()
        running[0] = True
        threading.Thread(target=serial_reader,  args=(port, baud), daemon=True).start()
        threading.Thread(target=processing_worker, daemon=True).start()
        self._set_running_ui(True)
        self.status_var.set(f"Streaming from {port} @ {baud} baud …")

    def _start_demo(self):
        self._clear()
        running[0] = True
        threading.Thread(target=demo_reader,       daemon=True).start()
        threading.Thread(target=processing_worker, daemon=True).start()
        self._set_running_ui(True)
        self.status_var.set("Demo mode — cycling through 6 configs (30 s each) …")

    def _stop(self):
        running[0] = False
        self._set_running_ui(False)
        self.status_var.set("Stopped.")

    def _clear(self):
        with data_lock:
            live_ts.clear(); live_bz.clear()
            for i in range(1, NUM_CONFIGS + 1):
                proc_buf[i].clear()
                buf_done[i] = False
            results.clear()
            proc_queue.clear()
            active_cfg[0]  = None
            cfg_changed[0] = False
        self.sig_line.set_data([], [])
        self.sig_wm.set_text("")
        self.ax_sig.set_title("Waiting for data …", color="#E0E0E0", fontsize=10, pad=8)
        for rect in self.bar_rects:
            rect.set_height(0); rect.set_alpha(0.25)
        for lbl in self.bar_labels:
            lbl.set_text("")
        self.best_ann.set_text("")
        for iid in self._table_rows.values():
            self.tv.delete(iid)
        self._table_rows.clear()
        for i, lbl in self.badges.items():
            lbl.config(bg="#1A1A2E", fg=CONFIG_COLORS[i],
                       font=("Segoe UI", 9), text=f" C{i} ")

    def _set_running_ui(self, state):
        self.start_btn.config(state="disabled" if state else "normal")
        self.demo_btn.config(state="disabled"  if state else "normal")
        self.stop_btn.config(state="normal"    if state else "disabled")

    def _on_close(self):
        running[0] = False
        time.sleep(0.1)
        self.destroy()
        sys.exit(0)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = BioZPlotter()
    app.mainloop()
