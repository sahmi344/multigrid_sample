import serial
import matplotlib.pyplot as plt
from collections import deque

# ==========================================
# SERIAL SETTINGS
# ==========================================
ser = serial.Serial(
    port='COM5',      # Change to your COM port
    baudrate=115200,
    timeout=1
)

# ==========================================
# PLOT SETTINGS
# ==========================================
WINDOW_SECONDS = 10
# Fixed display window
DISPLAY_SAMPLES = 1000

x_data = deque(range(DISPLAY_SAMPLES), maxlen=DISPLAY_SAMPLES)
y_data = deque([0]*DISPLAY_SAMPLES, maxlen=DISPLAY_SAMPLES)
# Dark theme
plt.style.use('dark_background')

plt.ion()

fig, ax = plt.subplots(figsize=(14, 6))

line, = ax.plot(x_data, y_data, lw=2)

ax.set_xlim(0, DISPLAY_SAMPLES)
ax.set_ylim(0, 4095)  # ESP32 ADC range

ax.set_title("Real-Time Bioimpedance Signal")
ax.set_xlabel("Time (s)")
ax.set_ylabel("BIOZI")
ax.grid(True, alpha=0.3)

print("Listening to ESP32...")

try:

    while True:

        raw = ser.readline()

        if not raw:
            continue

        line_text = raw.decode(
            'utf-8',
            errors='ignore'
        ).strip()

        # ----------------------------------
        # Skip Configuration Messages
        # ----------------------------------
        if line_text.startswith("Config"):
            print(line_text)
            continue

        # ----------------------------------
        # Expected format:
        # Config,Timestamp,BIOZI
        # Example:
        # 1,12345,2048
        # ----------------------------------
        fields = line_text.split(',')

        if len(fields) != 3:
            continue

        try:

            config = int(fields[0])
            timestamp = int(fields[1])
            biozi = int(fields[2])

        except ValueError:
            continue

        print(
            f"Cfg={config} | "
            f"Time={timestamp} | "
            f"BIOZI={biozi}"
        )

        # ----------------------------------
        # Add New Sample
        # ----------------------------------
        y_data.append(biozi)

        line.set_ydata(list(y_data))

        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        if len(x_data) > 1:

            current_time = x_data[-1]

            line.set_xdata(list(x_data))
            line.set_ydata(list(y_data))

            # ----------------------------------
            # Scrolling Window
            # ----------------------------------
            ax.set_xlim(
                max(0, current_time - WINDOW_SECONDS),
                current_time
            )

            # ----------------------------------
            # Auto Scale Y Axis
            # ----------------------------------
            ymin = min(y_data)
            ymax = max(y_data)

            margin = (ymax - ymin) * 0.1

            if margin < 1:
                margin = 1

            ax.set_ylim(
                ymin - margin,
                ymax + margin
            )

            ax.set_title(
                f"Real-Time Bioimpedance Signal | Config {config}"
            )

            fig.canvas.draw_idle()
            fig.canvas.flush_events()

except KeyboardInterrupt:

    print("\nStopped by user.")

finally:

    ser.close()
    plt.close('all')

    print("Serial Port Closed.")