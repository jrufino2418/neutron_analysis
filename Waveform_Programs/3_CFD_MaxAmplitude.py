import os
import struct
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

# Channel number to read
run_number = 97
channel_number = 9

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

# CFD and amplitude extraction function
def CFD_and_Amplitude(waveform, length):
    "Compute CFD and Amplitude from waveform."
    # Find maximum amplitude and its index
    Amax = np.max(waveform)
    imax = np.argmax(waveform)
    
    # Locate 75% of Amax on the rising edge
    for i in range(imax - 50, imax):
        if waveform[i] < 0.75 * Amax:
            i_75percent = i


    # Linear interpolation to find 50% crossing point for CFD
    m = (waveform[i_75percent + 1] - waveform[i_75percent]) 
    b = waveform[i_75percent] - m * i_75percent
    CFD = (0.5 * Amax - b) / m
    CFD_amplitude = m*CFD + b

    return CFD, CFD_amplitude, Amax, imax

# Continuous moving average (CMA) baseline correction function
def CMA_Filter(waveform, length, half_width, preload_Value, rejectThreshold):
    "CMA baseline correction algorithm."
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
    "Read exactly n bytes or raise EOFError."
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

        # Loop to the desired event
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

        # Baseline correction using continuous moving average (CMA) filter
        half_width = 50
        rejectThreshold = 50
        CMA_trace = CMA_Filter(waveform, waveform_length, half_width, preload_Value=waveform[0], rejectThreshold=rejectThreshold)
        # Compute CFD and Amplitude
        CFD, CFD_amplitude, Amax, imax = CFD_and_Amplitude(waveform - CMA_trace, waveform_length)
        print(f"Loaded event {idx}")
        # print(f"Board number: {board}")
        # print(f"Channel number: {channel}")
        # print(f"Timestamp: {timestamp}")
        # print(f"Energy long: {energy_long}")
        # print(f"Energy short: {energy_short}")
        # print(f"Flags: {flags}")
        # print(f"Waveform code: {waveform_code}")
        print(f"Waveform length: {waveform_length} samples")
        print(f"First 10 corrected waveform samples: {waveform[:10] - CMA_trace[:10]}")
        print(f"Amax: {Amax} at index {imax}")
        print(f"CFD (50% crossing point): {CFD_amplitude} at index {CFD}")
        print("--------------------------------------------------------")
        return x, waveform, CMA_trace, CFD, CFD_amplitude, Amax, imax

# State
current_event = 0
num_events = num_events_read  # replace with len(event_offsets)

# Initial plot
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.2)
x, waveform, CMA_trace, CFD, CFD_amplitude, Amax, imax = load_event(current_event)
corrected_waveform, = ax.plot(x, waveform - CMA_trace, color='blue', linestyle='-', label='Corrected Waveform')
imax_marker_x =  ax.axvline(imax, color='orange', linestyle='--', label='Amax Point')
Amax_marker_y = ax.axhline(Amax, color='orange', linestyle='--')
CFD_marker = ax.axvline(CFD, color='purple', linestyle='-.', label='CFD Point')
CFD_amplitude_marker = ax.axhline(CFD_amplitude, color='purple', linestyle='-.')
plt.legend()

# Update plot function
def update_plot(current_event):
    x, waveform, CMA_trace, CFD, CFD_amplitude, Amax, imax = load_event(current_event)
    corrected_waveform.set_xdata(x)
    corrected_waveform.set_ydata(waveform - CMA_trace)
    imax_marker_x.set_xdata([imax, imax])
    Amax_marker_y.set_ydata([Amax, Amax])
    CFD_marker.set_xdata([CFD, CFD])
    CFD_amplitude_marker.set_ydata([CFD_amplitude, CFD_amplitude])
    ax.set_title(f"Event {current_event}")
    ax.relim()
    ax.autoscale_view()
    plt.draw()

# Button callbacks
def next_event(event):
    global current_event
    if current_event < num_events - 1:
        current_event += 1
        update_plot(current_event)

def prev_event(event):
    global current_event
    if current_event > 0:
        current_event -= 1
        update_plot(current_event)

# Buttons
axprev = plt.axes([0.7, 0.05, 0.1, 0.075])
axnext = plt.axes([0.81, 0.05, 0.1, 0.075])
bnext = Button(axnext, 'Next')
bprev = Button(axprev, 'Previous')
bnext.on_clicked(next_event)
bprev.on_clicked(prev_event)

ax.set_title(f"Event {current_event}")
plt.show()