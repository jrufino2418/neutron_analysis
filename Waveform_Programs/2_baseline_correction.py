import os
import struct
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

# Channel number to read
run_number = 33
channel_number = 8

# Filename
filename = f"/home/jrufino/PhD_Research/DAQ_Compass/run_{run_number}/RAW/DataR_CH{channel_number}@V1730S_26980_run_{run_number}.BIN"

# Quick look at the file to estimate number of events
with open(filename, "rb") as f:
    # ignore 
    f.read(2+ 2 + 2 + 8 + 2 + 2 + 4 + 1)  # header + board + timestamp + energy_long (max) + energy_short + flags + waveform_code
    waveform_length_event0 = int.from_bytes(f.read(4), "little", signed=False)
    print(f"Waveform length of first event: {waveform_length_event0} samples")
size = os.path.getsize(filename)
num_events_read = (size - 2) / (2 + 2 + 8 + 2 + 2 + 4 + 1 + 4 + waveform_length_event0*2)  # approx estimate
print(f"File size: {size} bytes")
print(f"Estimated number of events: {num_events_read}")
print("--------------------------------------------------------")

#

# CMA baseline correction function
def CMA_Filter(waveform, length, half_width, preload_Value, rejectThreshold):
    """CMA baseline correction algorithm."""
    CMA_trace = np.array([])
    movingBaselineFilter = deque()
    movingBaselineValue = 0.0
    # Initialize the moving baseline filter with preload value if provided
    if preload_Value > -7777:
        for i in range(half_width):
            movingBaselineFilter.append(preload_Value)
            movingBaselineValue += preload_Value
    # Fill the initial part of the filter with the first few samples
    for i in range(half_width):
        if preload_Value > -7777:
            if abs(waveform[i] - movingBaselineValue / len(movingBaselineFilter)) >= rejectThreshold:
                continue
        movingBaselineFilter.append(waveform[i])
        movingBaselineValue += waveform[i]
    # Main loop for CMA calculation
    for i in range(length):
        if abs(waveform[i] - movingBaselineValue / len(movingBaselineFilter)) < rejectThreshold:
            if i + half_width < length:
                # We're still in valid lengths
                if len(movingBaselineFilter) >= half_width * 2 + 1:
                    # Filter is fully occupied, pop as we move
                    movingBaselineValue -= movingBaselineFilter.popleft()
                movingBaselineValue += waveform[i]
                movingBaselineFilter.append(waveform[i])
            else:
                movingBaselineValue -= movingBaselineFilter.popleft()
        CMA_trace = np.append(CMA_trace, movingBaselineValue / len(movingBaselineFilter))
    return CMA_trace

# Helper function to read exact bytes
def read_exact(f, n):
    """Read exactly n bytes or raise EOFError."""
    b = f.read(n)
    if len(b) < n:
        raise EOFError(f"Unexpected EOF (needed {n} bytes, got {len(b)})")
    return b

# Loading data based on event index
def load_event(idx):
    with open(filename, "rb") as file:
        # Reading data header (2 bytes)
        header16 = file.read(2)
        header = struct.unpack('<H', header16)[0]
        
        # Reading energy long (Channel (2 bytes), MeV (8 bytes), or both) based on header flags
        check1 = bool(header & 0x1)
        check2 = bool((header & 0x2) >> 1)
        check3 = bool((header & 0x3) == 0x3)
        if check1:
            energy_long_size = 2
        elif check2:
            energy_long_size = 8
        elif check3:
            energy_long_size = 10  # 2 + 8 bytes
        else:
            energy_long_size = 0

        # Determine if waveform data is present
        has_waveform = bool((header & 0x8) >> 3)

        for event in range(idx + 1):
            # Board number (2 bytes)
            board_bytes = file.read(2)
            board = int.from_bytes(board_bytes, "little", signed=False)

            # Channel (2 bytes)
            channel = int.from_bytes(read_exact(file, 2), "little", signed=False)

            # Timestamp (8 bytes)
            timestamp = int.from_bytes(read_exact(file, 8), "little", signed=False)

            # Energy long (variable bytes)
            energy_long = int.from_bytes(read_exact(file, energy_long_size), "little", signed=False)

            # Energy short (2 bytes)
            energy_short = int.from_bytes(read_exact(file, 2), "little", signed=False)

            # Flags (4 bytes)
            flags = int.from_bytes(read_exact(file, 4), "little", signed=False)

            # Waveform code (1 byte)
            waveform_code = int.from_bytes(read_exact(file, 1), "little", signed=False)

            # Waveform length (4 bytes, uint32)
            waveform_length = int.from_bytes(read_exact(file, 4), "little", signed=False)

            # Reading waveform samples/traces (2 bytes each) and storing in array
            for i in range(waveform_length):
                ADC = - int.from_bytes(read_exact(file,2), "little", signed=False) # invert signal
                if event == idx:
                    if i == 0:
                        x = np.array([i])
                        waveform = np.array(ADC)
                    else:
                        x = np.append(x, i)
                        waveform = np.append(waveform, ADC)
        # Baseline correction using CMA filter
        half_width = 50
        rejectThreshold = 50
        CMA_trace = CMA_Filter(waveform, waveform_length, half_width, preload_Value=waveform[0], rejectThreshold=rejectThreshold)
        print(f"Loaded event {idx}")
        # print(f"Board number: {board}")
        # print(f"Channel number: {channel}")
        # print(f"Timestamp: {timestamp}")
        # print(f"Energy long: {energy_long}")
        # print(f"Energy short: {energy_short}")
        # print(f"Flags: {flags}")
        # print(f"Waveform code: {waveform_code}")
        print(f"Waveform length: {waveform_length} samples")
        print(f"First 10 waveform samples (raw 16-bit): {waveform[:10]}")
        print("--------------------------------------------------------")
        return x, waveform, CMA_trace

# State
current_event = 0
num_events = num_events_read  # replace with len(event_offsets)

# Initial plot
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.2)
x, waveform, CMA_trace = load_event(current_event)
line, = ax.plot(x, waveform, color='blue', label='Inverted Waveform')
line_baseline, = ax.plot(x, CMA_trace, color='red', linestyle='--', label='CMA Baseline')
line_corrected, = ax.plot(x, waveform - CMA_trace, color='green', linestyle=':', label='Corrected Waveform')
plt.legend()

# Button callbacks
def next_event(event):
    global current_event
    if current_event < num_events - 1:
        current_event += 1
        x, waveform, CMA_trace = load_event(current_event)
        line.set_ydata(waveform)
        line_baseline.set_ydata(CMA_trace)
        line_corrected.set_ydata(waveform - CMA_trace)
        ax.set_title(f"Event {current_event}")
        plt.draw()

def prev_event(event):
    global current_event
    if current_event > 0:
        current_event -= 1
        x, waveform, CMA_trace = load_event(current_event)
        line.set_ydata(waveform)
        line_baseline.set_ydata(CMA_trace)
        line_corrected.set_ydata(waveform - CMA_trace)
        ax.set_title(f"Event {current_event}")
        plt.draw()

# Buttons
axprev = plt.axes([0.7, 0.05, 0.1, 0.075])
axnext = plt.axes([0.81, 0.05, 0.1, 0.075])
bnext = Button(axnext, 'Next')
bprev = Button(axprev, 'Previous')
bnext.on_clicked(next_event)
bprev.on_clicked(prev_event)

ax.set_title(f"Event {current_event}")
plt.show()