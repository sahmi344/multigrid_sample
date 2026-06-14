import serial
import matplotlib.pyplot as plt
from collections import deque
import time

# =====================================
# TIMING & THROTTLING CONFIGURATION
# =====================================
last_plot_time = time.time()
PLOT_INTERVAL = 0.05  # Restricts UI updates to 20 FPS (prevents CPU lockup)

# =====================================
# UART SETTINGS
# =====================================
ser = serial.Serial(
    port='COM5',
    baudrate=115200,
    timeout=1
)

# =====================================
# DISPLAY SETTINGS
# =====================================
DISPLAY_SAMPLES = 1000

# =====================================
# CONFIGURATION BUFFERS (Historical Data)
# =====================================
config_buffers = {
    1: [],
    2: [],
    3: [],
    4: [],
    5: [],
    6: []
}

# =====================================
# OSCILLOSCOPE BUFFER (Rolling View)
# =====================================
y_data = deque(
    [0] * DISPLAY_SAMPLES,
    maxlen=DISPLAY_SAMPLES
)

# =====================================
# PLOT SETUP
# =====================================
plt.style.use('dark_background')
plt.ion()  # Turn on interactive mode

fig, ax = plt.subplots(figsize=(14, 6))

line, = ax.plot(
    range(DISPLAY_SAMPLES),
    list(y_data),
    lw=2,
    color='#00FFCC'  # Clear, bright color for dark background
)

ax.set_xlim(0, DISPLAY_SAMPLES)
ax.set_ylim(0, 4095)

ax.set_title("Waiting For Data...", fontsize=12)
ax.set_xlabel("Samples")
ax.set_ylabel("BIOZI")
ax.grid(True, alpha=0.2)

# =====================================
# STATE VARIABLES
# =====================================
current_config = None

print("Listening... (Close the GUI window to stop safely)")

try:
    # Safely loops only while the Matplotlib window is actively open
    while plt.fignum_exists(fig.number):
        
        raw = ser.readline()
        if not raw:
            continue

        try:
            # Parse incoming text string
            line_text = raw.decode('utf-8', errors='ignore').strip()
            fields = line_text.split(',')
            
            if len(fields) != 3:
                continue

            config = int(fields[0])
            timestamp = int(fields[1])
            biozi = int(fields[2])
            
        except (ValueError, IndexError):
            # Gracefully ignore poorly formatted packets, don't crash
            continue

        # =====================================
        # CONFIGURATION CHANGE DETECTION
        # =====================================
        if current_config is None:
            current_config = config
            print(f"\nStarted Config {config}")

        elif config != current_config:
            print(f"\nConfig {current_config} Completed")
            print(f"Samples Stored = {len(config_buffers[current_config])}")
            print(f"Started Config {config}")

            # Clear oscilloscope display buffer cleanly back to baseline
            y_data.clear()
            y_data.extend([0] * DISPLAY_SAMPLES)
            
            current_config = config

        # =====================================
        # STORE DATA IN STATIC ARRAYS
        # =====================================
        if config in config_buffers:
            config_buffers[config].append(biozi)

        # Always feed the live rolling real-time buffer
        y_data.append(biozi)

        # =====================================
        # REAL-TIME DISPLAY (RATE LIMITED)
        # =====================================
        current_plot_time = time.time()
        
        if current_plot_time - last_plot_time > PLOT_INTERVAL:
            
            # 1. Convert to list once per frame update
            y_list = list(y_data)
            line.set_ydata(y_list)
            
            # 2. Heavy text/font engine manipulations happen only 20 times/sec
            ax.set_title(f"Bioimpedance Signal | Config {config}", fontsize=12)

            # 3. Dynamic axis limit calculations
            ymin, ymax = min(y_list), max(y_list)
            margin = max((ymax - ymin) * 0.1, 1)
            ax.set_ylim(ymin - margin, ymax + margin)

            # 4. Canvas interaction & rendering safely isolated
            try:
                fig.canvas.draw_idle()
                fig.canvas.flush_events()
            except Exception:
                # Catch rare window manipulation errors (like dragging/minimizing)
                pass
                
            last_plot_time = current_plot_time

except KeyboardInterrupt:
    print("\nStopped By User via Terminal Interrupt")

finally:
    # Cleanup resources perfectly on exit
    ser.close()
    plt.close('all')

    print("\n========== SUMMARY ==========")
    for cfg in config_buffers:
        print(f"Config {cfg}: {len(config_buffers[cfg])} samples")
    print("=============================")
