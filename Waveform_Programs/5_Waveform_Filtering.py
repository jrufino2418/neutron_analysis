import os
import struct
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox

# Channel number to read  & integration parameters
run_number = 33
channel_number = 0
integration_start = 12
integration_offset = 12
integration_stop = 100
integration_method = 3
CFD_pileup_threshold = 0.45  # Threshold for pileup detection
CFD_thinpulse_threshold = 0.8  # Threshold for thin pulse detection

# Loading binary data file
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

# Thin pulse filtering
def Thin_Pulse_Filtering(waveform):
    rectangular_pulse = np.zeros(30)
    rectangular_pulse[10:] = 10.0  # Rectangular pulse of width 10 samples
    convolution_result = np.convolve(waveform, rectangular_pulse, mode='same')
    Amax_convolution = np.max(convolution_result)
    threshold_thinpulse = CFD_thinpulse_threshold * Amax_convolution
    count_crossings = 0
    length_convolution = len(convolution_result)
    for i in range(0, length_convolution - 1):
        if (convolution_result[i] > threshold_thinpulse):
            count_crossings += 1
    # Check convolution values for thin pulse detection 
    if (count_crossings > 16):
        thin_pulse_event = True
    else:
        thin_pulse_event = False
    return thin_pulse_event, count_crossings, convolution_result

# Pileup filtering 
def Pileup_Filtering(second_derivative_waveform):
    # Find maximum amplitude of second derivative
    #Amax_2nd_derivative = np.max(second_derivative_waveform)
    Amin_2nd_derivative = np.min(second_derivative_waveform)
    imin = np.argmin(second_derivative_waveform)
    count_crossings = 1
    
    # Check if 50 % of Amin is crossed more than twice
    Threshold_pileup = CFD_pileup_threshold * Amin_2nd_derivative
    length_2nd_derivative = len(second_derivative_waveform)
    for i in range(0, length_2nd_derivative - 1):
        if ((second_derivative_waveform[i] < Threshold_pileup) and (second_derivative_waveform[i+1] > Threshold_pileup)):
            if ((i < imin - 15) or (i > imin + 15)):
                count_crossings += 1
    # Check second derivative values before and after the peak for pileup and thin pulse detection
    if (count_crossings > 1):
        pile_up_event = True
    else:
        pile_up_event = False

    return pile_up_event, count_crossings

# PSD Integration Script
def PSD_Integration(waveform, start, stop, offset, method):
    # Find maximum amplitude and its index
    Amax = np.max(waveform)
    imax = np.argmax(waveform)
    # Integration calculations
    if ((imax - start) > 0) and ((imax + stop) < len(waveform)):
        short_integral = 0; long_integral = 0
        waveform_short = waveform[imax + offset : imax + stop]; time_index_short = np.arange(0, len(waveform_short))
        waveform_long = waveform[imax - start : imax + stop]; time_index_long = np.arange(0, len(waveform_long))
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
        baseline = 0
        for i in range(waveform_length):
            ADC = 15200 - int.from_bytes(read_exact(file,2), "little", signed=False) # invert signal
            if i == 0:
                x = np.array([i])
                waveform = np.array(ADC)
            else:
                x = np.append(x, i)
                waveform = np.append(waveform, ADC)
            if i < 40: # baseline calculation using first 40 samples
                baseline += ADC
        baseline = baseline / 40
        waveform = waveform - baseline  # baseline subtraction

        # Baseline correction using continuous moving average (CMA) filter
        #half_width = 10
        #rejectThreshold = 1
        #CMA_trace = CMA_Filter(waveform, waveform_length, half_width, preload_Value=waveform[0], rejectThreshold=rejectThreshold)
        corrected_waveform = waveform #- CMA_trace

        # Compute CFD and Amplitude
        #CFD, CFD_amplitude, Amax, imax = CFD_and_Amplitude(corrected_waveform)
        print(f"Loaded event {idx}")

        # print(f"Board number: {board}")
        # print(f"Channel number: {channel}")
        # print(f"Timestamp: {timestamp}")
        # print(f"Energy long: {energy_long}")
        # print(f"Energy short: {energy_short}")
        # print(f"Flags: {flags}")
        # print(f"Waveform code: {waveform_code}")
        # print(f"Waveform length: {waveform_length} samples")
        return x, corrected_waveform

# State
current_event = 0
num_events = num_events_read
x, corrected_waveform = load_event(current_event, energy_long_size, waveform_length)  # Waveform data

# Initialization of figures and plots
fig, ((ax1, ax2), (ax4, ax3)) = plt.subplots(2,2, sharex = False, figsize=(16, 10))
plt.subplots_adjust(bottom=0.2)

# Plot 1: Waveform
corrected_waveform_plot, = ax1.plot(x, corrected_waveform, color='black', linestyle=':', label='Corrected Waveform')

# Plot 1: PSD integration + plotting integration windows
short_integral, long_integral, imax = PSD_Integration(corrected_waveform, integration_start, integration_stop, integration_offset, integration_method)
if long_integral != 0:
    print(f"PSD:\t {short_integral/long_integral}")
mask_long = (x >= (imax - integration_start)) & (x <= (imax + integration_stop))
mask_short = (x >= (imax + integration_offset)) & (x <= (imax + integration_stop))
highlight_long = ax1.fill_between(x[mask_long], corrected_waveform[mask_long], color='red', alpha=0.3, label=f'Long Integration Area: {long_integral}')
highlight_short = ax1.fill_between(x[mask_short], corrected_waveform[mask_short], color='blue', alpha=0.3, label=f'Short Integration Area: {short_integral}')

ax1.set_title(f"Event {current_event} (PSD = {short_integral/long_integral})")
ax1.set_xlabel('Time (a.u.)')
ax1.set_ylabel('Waveform Amplitude (a.u.)')
ax1.legend()

# Plot 2: Waveform derivative
dx = 1  # Assuming uniform sampling with a step of 1 unit
derivative_waveform = np.gradient(corrected_waveform, dx) # Derivative of waveform 
derivative_waveform_plot, = ax2.plot(x, derivative_waveform, color='black', linestyle=':', marker = 'o', label='Derivative Waveform')

ax2.set_xlabel('Time (a.u.)')
ax2.set_ylabel('1st Derivative (a.u.)')

# Plot 3: Waveform 2nd derivative
second_derivative_waveform = np.gradient(derivative_waveform, dx)
second_derivative_waveform_plot, = ax3.plot(x, second_derivative_waveform, color='black', marker = 'o', label='2nd Derivative Waveform')
 

# Plot 3: CFD threshold line
pile_up_event, count_crossings = Pileup_Filtering(second_derivative_waveform)
zeros_array = np.zeros(len(second_derivative_waveform))
threshold_line = zeros_array + CFD_pileup_threshold * np.min(second_derivative_waveform)
threshold_line_plot = ax3.plot(x, threshold_line, color='red', linestyle='--', label=f'Pileup event?: {pile_up_event}\n Count crossings: {count_crossings}')

ax3.set_xlabel('Time (a.u.)')
ax3.set_ylabel('2nd Derivative (a.u.)')
ax3.legend()

# Plot 4: Convolution of rectangular pulse + waveform 
thin_pulse_event, conv_count_crossings, convolution_result = Thin_Pulse_Filtering(corrected_waveform)
convolution_plot, = ax4.plot(x, convolution_result, color='black', marker = 'o', label='')

# Plot 4: CFD threshold line
zeros_array = np.zeros(len(second_derivative_waveform))
conv_threshold_line = zeros_array + CFD_thinpulse_threshold * np.max(convolution_result)
conv_threshold_line_plot = ax4.plot(x, conv_threshold_line, color='red', linestyle='--', label=f'Thin Pulse event?: {thin_pulse_event} \n Count crossings: {conv_count_crossings}')

ax4.set_xlabel('Time (a.u.)')
ax4.set_ylabel('Convolution Amplitude (a.u.)')
ax4.legend()
print("--------------------------------------------------------")

plt.legend()

# Update plot function
def update_plot(current_event):
    global highlight_long, highlight_short
    x, corrected_waveform = load_event(current_event, energy_long_size, waveform_length)

    # Update plot 1: Waveform 
    corrected_waveform_plot.set_xdata(x)
    corrected_waveform_plot.set_ydata(corrected_waveform)
    # Update PSD and highlighted areas
    short_integral, long_integral, imax = PSD_Integration(corrected_waveform, integration_start, integration_stop, integration_offset, integration_method)
    if long_integral != 0:
        print(f"PSD:\t {short_integral/long_integral}")
    mask_long = (x >= (imax - integration_start)) & (x <= (imax + integration_stop))
    mask_short = (x >= (imax + integration_offset)) & (x <= (imax + integration_stop))
    highlight_long.remove()
    highlight_short.remove()
    highlight_long = ax1.fill_between(x[mask_long], corrected_waveform[mask_long], color='red', alpha=0.3, label=f'Long Integration Area: {long_integral}')
    highlight_short = ax1.fill_between(x[mask_short], corrected_waveform[mask_short], color='blue', alpha=0.3, label=f'Short Integration Area: {short_integral}')

    ax1.set_title(f"Event {current_event} (PSD = {short_integral/long_integral})")
    ax1.relim()
    ax1.autoscale_view()
    ax1.legend()

    # Update plot 2: waveform derivative
    derivative_waveform = np.gradient(corrected_waveform, dx) # Derivative of waveform 
    derivative_waveform_plot.set_xdata(x)
    derivative_waveform_plot.set_ydata(derivative_waveform)
    
    ax2.relim()
    ax2.autoscale_view()

    # Update plot 3: waveform 2nd derivative 
    second_derivative_waveform = np.gradient(derivative_waveform, dx) # 2nd Derivative of waveform
    second_derivative_waveform_plot.set_xdata(x)
    second_derivative_waveform_plot.set_ydata(second_derivative_waveform)
    # Update CFD threshold line
    pile_up_event, count_crossings = Pileup_Filtering(second_derivative_waveform)
    zeros_array = np.zeros(len(second_derivative_waveform))
    threshold_line = zeros_array + CFD_pileup_threshold * np.min(second_derivative_waveform)
    threshold_line_plot[0].set_xdata(x)
    threshold_line_plot[0].set_ydata(threshold_line)
    threshold_line_plot[0].set_label(f'Pileup event?: {pile_up_event} \nCount crossings: {count_crossings}')

    ax3.relim()
    ax3.autoscale_view()
    ax3.legend()

    # Update plot 4: Convolution of rectangular pulse + waveform
    thin_pulse_event, conv_count_crossings, convolution_result = Thin_Pulse_Filtering(corrected_waveform)
    convolution_plot.set_xdata(x)
    convolution_plot.set_ydata(convolution_result)
    # Update CFD threshold line
    zeros_array = np.zeros(len(second_derivative_waveform))
    conv_threshold_line = zeros_array + CFD_thinpulse_threshold * np.max(convolution_result)
    conv_threshold_line_plot[0].set_xdata(x)
    conv_threshold_line_plot[0].set_ydata(conv_threshold_line)
    conv_threshold_line_plot[0].set_label(f'Thin Pulse event?: {thin_pulse_event} \n Count crossings: {conv_count_crossings}')

    ax4.relim()
    ax4.autoscale_view()
    ax4.legend()
    print("--------------------------------------------------------")
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


plt.show()