import os
import struct
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox

# Channel number to read
run_number = 97
channel_number = 9
integration_start = 7
integration_offset = 12
integration_stop = 100
integration_method = 3

# Filename
filename = f"/home/jrufino/PhD_Research/DAQ_Compass/run_{run_number}/RAW/DataR_CH{channel_number}@V1730S_26980_run_{run_number}.BIN"
print(f"Loading {filename}...")

# Check header + 1st event data to: know how many events there are,  
# know the data structure, and if waveform are present
size = os.path.getsize(filename)
with open(filename, "rb") as file:
    # Reading data header (2 bytes)
    header16 = file.read(2)
    header = struct.unpack('<H', header16)[0]
    # Determining energy long (Channel (2 bytes), MeV (8 bytes), or both Channel and MeV (10 bytes)) based on header flags
    check1 = bool(header & 0x1)
    check2 = bool((header & 0x2) >> 1)
    check3 = bool((header & 0x3) == 0x3)
    if check1:
        energy_long_size = 2; print("energy long size = 2")
    elif check2:
        energy_long_size = 8; print("energy long size = 8")
    elif check3:
        energy_long_size = 10; print("energy long size = 10")  # 2 + 8 bytes
    else:
        energy_long_size = 0
    # Determine if waveform data is present
    has_waveform = bool((header & 0x8) >> 3)
    # If waveform present, calculate how many events are in the file by waveform length of first event
    if has_waveform:
        file.read(2 + 2 + 8 + energy_long_size + 2 + 4 + 1)  # board + channel + timestamp + energy_long + energy_short + flags + waveform_code
        waveform_length = int.from_bytes(file.read(4), "little", signed=False)
        print(f"Waveform length of first event: {waveform_length} samples")

num_events_read = (size - 2) / (2 + 2 + 8 + energy_long_size + 2 + 4 + 1 + 4 + waveform_length*2)  # approx estimate
print(f"File size: {size} bytes")
print(f"Number of events in file: {num_events_read}")
print("--------------------------------------------------------")

# PSD Integration Script
def PSD_Integration(waveform, start, stop, offset, method):
    # Find maximum amplitude and its index
    Amax = np.max(waveform)
    imax = np.argmax(waveform)
    print(f"imax - start: {imax - start}")
    print(f"imax + stop: {imax + stop}")
    # Integration calculations
    if ((imax - start) > 0) and ((imax + stop) < len(waveform)):
        short_integral = 0; long_integral = 0
        waveform_short = waveform[imax + offset : imax + stop]; time_index_short = np.arange(0, len(waveform_short))
        waveform_long = waveform[imax - start : imax + stop]; time_index_long = np.arange(0, len(waveform_long))
        print(f"{len(time_index_long)} {len(waveform_long)}")
        # Trapezoidal Integration
        if method == 1:
            short_integral = np.trapezoid(time_index_short, waveform_short)
            long_integral = np.trapezoid(time_index_long, waveform_long)
        # Composite Simpson's Rule
        if method == 2:
            nothing = 0  # Placeholder for future implementation
        # Rectangular Integration
        if method == 3:
            for i in waveform_short:
                short_integral += i
            for i in waveform_long:
                long_integral += i

    else:
        short_integral = -1
        long_integral = -1

    return short_integral, long_integral, imax

# CFD and amplitude extraction function
def CFD_and_Amplitude(waveform):
    "Compute CFD and Amplitude from waveform."
    # Find maximum amplitude and its index
    Amax = np.max(waveform)
    imax = np.argmax(waveform)
    
    # Locate 75% of Amax on the rising edge
    for i in range(imax - 50, imax):
        if waveform[i] < 0.75 * Amax:
            i_75percent = i
        else:
            i_75percent = 0


    # Linear interpolation to find 50% crossing point for CFD
    m = (waveform[i_75percent + 1] - waveform[i_75percent]) 
    b = waveform[i_75percent] - m * i_75percent
    CFD = (0.5 * Amax - b) / m
    CFD_amplitude = m*CFD + b

    return CFD, CFD_amplitude, Amax, imax

# Continuous moving average (CMA) baseline correction function
def CMA_Filter(waveform, length, half_width, preload_Value, rejectThreshold):
    "CMA baseline correction algorithm."
    CMA_trace = np.zeros(length)
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
        CMA_trace[i] = movingBaselineValue / len(movingBaselineFilter)
        #print(f"{i}\t{waveform[i]}\t{CMA_trace[i]}" )
    return CMA_trace

# Helper function to read exact bytes
def read_exact(f, n):
    "Read exactly n bytes or raise EOFError."
    b = f.read(n)
    if len(b) < n:
        raise EOFError(f"Unexpected EOF (needed {n} bytes, got {len(b)})")
    return b

# Loading data based on event index
def load_event(idx, energy_long_size, waveform_length):
    with open(filename, "rb") as file:
        # Reading data header (2 bytes)
        header16 = file.read(2)

        # Jump to event skipping by event 
        read_bytes = 2 + 2 + 8 + energy_long_size + 2 + 4 + 1 + + 4 + 2*waveform_length
        bytes_read = 0
        events_read = 0
        for i in range(idx):
            junk = file.read(read_bytes)
            bytes_read += read_bytes
            events_read += 1
            #print(f"{i}: Read {bytes_read} bytes")        

        # Read current event
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

        # Waveform length (4 bytes)
        waveform_length = int.from_bytes(read_exact(file, 4), "little", signed=False)

        # Reading waveform samples/traces (2 bytes each) and storing in array
        for i in range(waveform_length):
            ADC = 15370 - int.from_bytes(read_exact(file,2), "little", signed=False) # invert signal
            if i == 0:
                x = np.array([i])
                waveform = np.array(ADC)
            else:
                x = np.append(x, i)
                waveform = np.append(waveform, ADC)

        # Baseline correction using continuous moving average (CMA) filter
        half_width = 10
        rejectThreshold = 1
        CMA_trace = CMA_Filter(waveform, waveform_length, half_width, preload_Value=waveform[0], rejectThreshold=rejectThreshold)
        corrected_waveform = waveform - CMA_trace

        # Compute CFD and Amplitude
        CFD, CFD_amplitude, Amax, imax = CFD_and_Amplitude(corrected_waveform)
        print(f"Loaded event {idx}")

        # PSD Integration parameters
        psd_start = integration_start
        psd_stop = integration_stop
        psd_offset = integration_offset
        psd_method = integration_method
        short_integral, long_integral, imax = PSD_Integration(corrected_waveform, psd_start, psd_stop, psd_offset, psd_method)
        if long_integral != 0:
            print(f"PSD:\t {short_integral/long_integral}")
        # print(f"Board number: {board}")
        # print(f"Channel number: {channel}")
        # print(f"Timestamp: {timestamp}")
        # print(f"Energy long: {energy_long}")
        # print(f"Energy short: {energy_short}")
        # print(f"Flags: {flags}")
        # print(f"Waveform code: {waveform_code}")
        print(f"Waveform length: {waveform_length} samples")
        print(f"First 10 corrected waveform samples: {corrected_waveform[:10]}")
        #print(f"Amax: {Amax} at index {imax}")
        #print(f"CFD (50% crossing point): {CFD_amplitude} at index {CFD}")
        print("--------------------------------------------------------")
        return x, corrected_waveform, imax, psd_start, psd_stop, psd_offset, long_integral, short_integral

# State
current_event = 0
num_events = num_events_read
# Initial plot
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.2)
x, corrected_waveform, imax, psd_start, psd_stop, psd_offset, long_integral, short_integral = load_event(current_event, energy_long_size, waveform_length)
corrected_waveform_plot, = ax.plot(x, corrected_waveform, color='black', linestyle=':', label='Corrected Waveform')

# Highlight integration areas
mask_long = (x >= (imax - psd_start)) & (x <= (imax + psd_stop))
mask_short = (x >= (imax + psd_offset)) & (x <= (imax + psd_stop))

highlight_long = ax.fill_between(x[mask_long], corrected_waveform[mask_long], color='red', alpha=0.3, label=f'Long Integration Area: {long_integral}')
highlight_short = ax.fill_between(x[mask_short], corrected_waveform[mask_short], color='blue', alpha=0.3, label=f'Short Integration Area: {short_integral}')

plt.legend()

# Update plot function
def update_plot(current_event):
    global highlight_long, highlight_short
    x, corrected_waveform, imax, psd_start, psd_stop, psd_offset, long_integral, short_integral = load_event(current_event, energy_long_size, waveform_length)
    # Update waveform data
    corrected_waveform_plot.set_xdata(x)
    corrected_waveform_plot.set_ydata(corrected_waveform)
    # Update highlighted areas
    mask_long = (x >= (imax - psd_start)) & (x <= (imax + psd_stop))
    mask_short = (x >= (imax + psd_offset)) & (x <= (imax + psd_stop))
    highlight_long.remove()
    highlight_short.remove()
    highlight_long = ax.fill_between(x[mask_long], corrected_waveform[mask_long], color='red', alpha=0.3, label=f'Long Integration Area: {long_integral}')
    highlight_short = ax.fill_between(x[mask_short], corrected_waveform[mask_short], color='blue', alpha=0.3, label=f'Short Integration Area: {short_integral}')
    
    ax.set_title(f"Event {current_event} (PSD = {short_integral/long_integral})")
    ax.relim()
    ax.autoscale_view()
    ax.legend()
    plt.draw()
    return highlight_long, highlight_short

# Button callbacks
def next_event(event):
    global current_event
    global num_events
    if current_event < num_events - 1:
        current_event += 1
        highlight_long, highlight_short = update_plot(current_event)

def prev_event(event):
    global current_event
    global num_events
    if current_event > 0:
        current_event -= 1
        highlight_long, highlight_short = update_plot(current_event)

def submit_method(text):
    global current_event
    global num_events
    try:
        method_value = int(text)
        if ((method_value >= 0) and (method_value < num_events)):
            current_event = method_value
            highlight_long, highlight_short = update_plot(current_event)
        else:
            print(f"Input value ({method_value})is out of bounds")
    except ValueError:
        pass

# Buttons
axprev = plt.axes([0.125, 0.05, 0.1, 0.075])
axnext = plt.axes([0.8, 0.05, 0.1, 0.075])
bnext = Button(axnext, 'Next')
bprev = Button(axprev, 'Previous')
bnext.on_clicked(next_event)
bprev.on_clicked(prev_event)

axbox_method = plt.axes([0.5, 0.05, 0.1, 0.075])
textbox = TextBox(axbox_method, 'Event input: ')
textbox.on_submit(submit_method)


ax.set_title(f"Event {current_event} (PSD = {short_integral/long_integral})")
plt.show()