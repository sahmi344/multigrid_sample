
import sys
import serial
import serial.tools.list_ports
import threading
import collections
import time
import tkinter as tk
from tkinter import ttk, messagebox

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# ── Configuration ─────────────────────────────────────────────────────────────

NUM_CONFIGS  = 6
WINDOW_SIZE  = 500
BAUD_DEFAULT = 115200
UPDATE_MS    = 50

CONFIG_COLORS = {
    1: "#E63946",
    2: "#2196F3",
    3: "#4CAF50",
    4: "#FF9800",
    5: "#9C27B0",
    6: "#00BCD4",
}

# ── Shared state ──────────────────────────────────────────────────────────────

data_lock    = threading.Lock()
timestamps   = collections.deque(maxlen=WINDOW_SIZE)
bioz_values  = collections.deque(maxlen=WINDOW_SIZE)
active_cfg   = [None]
prev_cfg     = [None]
cfg_changed  = [False]
running      = [False]

# ── Serial reader thread ──────────────────────────────────────────────────────

def serial_reader(port_name: str, baud: int):
    try:
        ser = serial.Serial(port_name, baud, timeout=1)
    except serial.SerialException as exc:
        print(f"[ERROR] Cannot open {port_name}: {exc}")
        running[0] = False
        return

    print(f"[INFO] Listening on {port_name} @ {baud} baud …")
    buffer = ""

    while running[0]:
        try:
            raw = ser.read(ser.in_waiting or 1)
            if raw:
                buffer += raw.decode("ascii", errors="ignore")
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(",")
                    if len(parts) != 3:
                        continue
                    try:
                        cfg  = int(parts[0])
                        ts   = int(parts[1])
                        bval = float(parts[2])
                    except ValueError:
                        continue
                    if cfg not in range(1, NUM_CONFIGS + 1):
                        continue
                    with data_lock:
                        if cfg != active_cfg[0]:
                            timestamps.clear()
                            bioz_values.clear()
                            prev_cfg[0]    = active_cfg[0]
                            active_cfg[0]  = cfg
                            cfg_changed[0] = True
                        timestamps.append(ts)
                        bioz_values.append(bval)
        except serial.SerialException:
            break

    ser.close()
    print("[INFO] Serial port closed.")

# ── Demo / simulation thread ──────────────────────────────────────────────────

def demo_reader():
    import math, random
    t       = 0
    cfg     = 1
    seg     = 0
    SEG_LEN = 100

    while running[0]:
        bval = 2048 + 800 * math.sin(2 * math.pi * t / 60) + random.gauss(0, 30)
        ts   = t * 10

        with data_lock:
            if cfg != active_cfg[0]:
                timestamps.clear()
                bioz_values.clear()
                prev_cfg[0]    = active_cfg[0]
                active_cfg[0]  = cfg
                cfg_changed[0] = True
            timestamps.append(ts)
            bioz_values.append(bval)

        t   += 1
        seg += 1
        if seg >= SEG_LEN:
            seg  = 0
            cfg  = (cfg % NUM_CONFIGS) + 1

        time.sleep(0.02)

# ── GUI ───────────────────────────────────────────────────────────────────────

class BioZPlotter(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Real-Time BioZ Signal Plotter")
        self.configure(bg="#1A1A2E")
        self.resizable(True, True)
        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("Dark.TFrame",   background="#1A1A2E")
        style.configure("Dark.TLabel",   background="#1A1A2E", foreground="#E0E0E0",
                        font=("Segoe UI", 10))
        style.configure("Active.TLabel", background="#1A1A2E", foreground="#4FC3F7",
                        font=("Segoe UI", 12, "bold"))
        style.configure("Dark.TButton",  background="#0F3460", foreground="#E0E0E0",
                        font=("Segoe UI", 10, "bold"), borderwidth=0)
        style.configure("Stop.TButton",  background="#7B0000", foreground="#E0E0E0",
                        font=("Segoe UI", 10, "bold"), borderwidth=0)
        style.configure("Dark.TCombobox", fieldbackground="#0F3460", foreground="#E0E0E0",
                        background="#0F3460")
        style.configure("Dark.TEntry",   fieldbackground="#0F3460", foreground="#E0E0E0")

        # ── Control bar ───────────────────────────────────────────────────────
        ctrl = ttk.Frame(self, style="Dark.TFrame", padding=(10, 8))
        ctrl.pack(fill="x", side="top")

        ttk.Label(ctrl, text="Port:", style="Dark.TLabel").pack(side="left", padx=(0, 4))
        self.port_var = tk.StringVar()
        ports = [p.device for p in serial.tools.list_ports.comports()]
        self.port_cb = ttk.Combobox(ctrl, textvariable=self.port_var, values=ports,
                                    width=14, style="Dark.TCombobox")
        self.port_cb.pack(side="left", padx=(0, 10))
        if ports:
            self.port_cb.current(0)

        ttk.Label(ctrl, text="Baud:", style="Dark.TLabel").pack(side="left", padx=(0, 4))
        self.baud_var = tk.StringVar(value=str(BAUD_DEFAULT))
        ttk.Entry(ctrl, textvariable=self.baud_var, width=8,
                  style="Dark.TEntry").pack(side="left", padx=(0, 10))

        self.start_btn = ttk.Button(ctrl, text="▶  Start", style="Dark.TButton",
                                    command=self._start_serial)
        self.start_btn.pack(side="left", padx=4)

        self.demo_btn = ttk.Button(ctrl, text="⚡  Demo", style="Dark.TButton",
                                   command=self._start_demo)
        self.demo_btn.pack(side="left", padx=4)

        self.stop_btn = ttk.Button(ctrl, text="■  Stop", style="Stop.TButton",
                                   command=self._stop, state="disabled")
        self.stop_btn.pack(side="left", padx=4)

        ttk.Button(ctrl, text="🗑  Clear", style="Dark.TButton",
                   command=self._clear).pack(side="left", padx=4)

        # Config indicator badges (right side)
        badge_frame = ttk.Frame(ctrl, style="Dark.TFrame")
        badge_frame.pack(side="right", padx=10)
        self.badges = {}
        for i in range(1, NUM_CONFIGS + 1):
            lbl = tk.Label(badge_frame, text=f" C{i} ", bg="#1A1A2E",
                           fg=CONFIG_COLORS[i], font=("Segoe UI", 9, "bold"),
                           relief="flat", padx=4, pady=2)
            lbl.pack(side="left", padx=2)
            self.badges[i] = lbl

        # ── Matplotlib figure ─────────────────────────────────────────────────
        self.fig, self.ax = plt.subplots(figsize=(13, 6),
                                          facecolor="#0D0D1A", constrained_layout=True)
        self.ax.set_facecolor("#0D0D1A")
        self.ax.tick_params(colors="#9E9E9E")
        for spine in self.ax.spines.values():
            spine.set_color("#333366")
        self.ax.set_xlabel("Timestamp (ms)", color="#9E9E9E", fontsize=10)
        self.ax.set_ylabel("BioZ Value",     color="#9E9E9E", fontsize=10)
        self.ax.grid(True, color="#1E1E3A", linewidth=0.8, linestyle="--")

        (self.line,) = self.ax.plot([], [], color="#E0E0E0", linewidth=1.8)

        # Config label watermark inside plot
        self.cfg_text = self.ax.text(
            0.01, 0.96, "", transform=self.ax.transAxes,
            fontsize=28, fontweight="bold", alpha=0.15,
            color="#FFFFFF", va="top", ha="left"
        )

        # Title
        self.title_text = self.ax.set_title(
            "Waiting for data …",
            color="#E0E0E0", fontsize=12, pad=10
        )

        canvas = FigureCanvasTkAgg(self.fig, master=self)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=(0, 4))
        self.canvas = canvas

        toolbar_frame = ttk.Frame(self, style="Dark.TFrame")
        toolbar_frame.pack(fill="x", side="bottom")
        NavigationToolbar2Tk(canvas, toolbar_frame)

        # Status bar
        self.status_var = tk.StringVar(value="Ready — connect a port or run the demo.")
        ttk.Label(self, textvariable=self.status_var, style="Dark.TLabel",
                  anchor="w", padding=(10, 3)).pack(fill="x", side="bottom")

        self._ani = animation.FuncAnimation(
            self.fig, self._animate, interval=UPDATE_MS,
            blit=False, cache_frame_data=False
        )

    # ── Animation ─────────────────────────────────────────────────────────────

    def _animate(self, _frame):
        with data_lock:
            xs      = list(timestamps)
            ys      = list(bioz_values)
            cfg     = active_cfg[0]
            changed = cfg_changed[0]
            if changed:
                cfg_changed[0] = False

        if cfg is None:
            return

        color = CONFIG_COLORS.get(cfg, "#E0E0E0")

        # Update line data & color
        self.line.set_data(xs, ys)
        self.line.set_color(color)

        # Update title & watermark
        self.ax.set_title(f"BioZ Signal — Configuration {cfg}",
                          color=color, fontsize=12, pad=10)
        self.cfg_text.set_text(f"CONFIG {cfg}")
        self.cfg_text.set_color(color)

        # Highlight active badge, dim others
        for i, lbl in self.badges.items():
            if i == cfg:
                lbl.config(bg=color, fg="#0D0D1A",
                           font=("Segoe UI", 9, "bold"), relief="flat")
            else:
                lbl.config(bg="#1A1A2E", fg=CONFIG_COLORS[i],
                           font=("Segoe UI", 9), relief="flat")

        if xs:
            self.ax.relim()
            self.ax.autoscale_view()

        self.status_var.set(
            f"Config {cfg}  |  Samples: {len(xs)}  |  "
            + (f"Latest BioZ: {ys[-1]:.1f}" if ys else "")
        )

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
        threading.Thread(target=serial_reader, args=(port, baud), daemon=True).start()
        self._set_running_ui(True)
        self.status_var.set(f"Streaming from {port} @ {baud} baud …")

    def _start_demo(self):
        self._clear()
        running[0] = True
        threading.Thread(target=demo_reader, daemon=True).start()
        self._set_running_ui(True)
        self.status_var.set("Demo mode — simulated BioZ cycling through 6 configs.")

    def _stop(self):
        running[0] = False
        self._set_running_ui(False)
        self.status_var.set("Stopped.")

    def _clear(self):
        with data_lock:
            timestamps.clear()
            bioz_values.clear()
            active_cfg[0]  = None
            prev_cfg[0]    = None
            cfg_changed[0] = False
        self.line.set_data([], [])
        self.ax.set_title("Waiting for data …", color="#E0E0E0", fontsize=12, pad=10)
        self.cfg_text.set_text("")
        for i, lbl in self.badges.items():
            lbl.config(bg="#1A1A2E", fg=CONFIG_COLORS[i], font=("Segoe UI", 9))

    def _set_running_ui(self, state: bool):
        self.start_btn.config(state="disabled" if state else "normal")
        self.demo_btn.config(state="disabled" if state else "normal")
        self.stop_btn.config(state="normal"   if state else "disabled")

    def _on_close(self):
        running[0] = False
        time.sleep(0.1)
        self.destroy()
        sys.exit(0)

# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = BioZPlotter()
    app.mainloop()
