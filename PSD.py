from calibration_and_setup import calibration as cal
from calibration_and_setup import cal_setup 
from calibration_and_setup import setup 
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from lmfit.models import GaussianModel
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import argparse
import struct
import mmap
import gc
import os


# Gaussian fitting function using lmfit
def fit_gaussians_lmfit(x, y, npeaks):
    """Fit 2 Gaussians using lmfit with peak guesses from find_peaks"""
    # Detect peaks for initial guesses
    peaks, props = find_peaks(y, height=max(y)*0.05, distance=15)  # tune thresholds
    if len(peaks) < npeaks: # not enough peaks found
        return None        

    # Limit to npeaks strongest
    peaks = peaks[np.argsort(y[peaks])[-npeaks:]]

    # Build lmfit model
    model = None
    params = None
    for i, p in enumerate(peaks):
        prefix = f"g{i}_"
        g = GaussianModel(prefix=prefix)
        if model is None:
            model = g
            params = g.make_params()
        else:
            model += g
            params.update(g.make_params())

        # Initial guesses from data
        params[f"{prefix}center"].set(x[p], min=x.min(), max=x.max())
        params[f"{prefix}amplitude"].set(y[p], min=0)
        params[f"{prefix}sigma"].set(0.01*(x.max()-x.min()), min=0.001)

    # Do fit
    result = model.fit(y, params, x=x)
    return result

# Function for the light curve 
def light_curve_function(x, A, B, C):
    return A/np.sqrt(x) + B * np.exp(C * x)

# Thin pulse filtering; only needed for channels 0 - 6 (small stilbene detectors)
def Thin_Pulse_Filtering(waveform, channel_number, CFD_thinpulse_threshold):
    thin_pulse_event = False
    if ((channel_number > -1) and (channel_number < 7)):
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
        return thin_pulse_event
    else:
        return thin_pulse_event

# Pileup filtering 
def Pileup_Filtering(second_derivative_waveform, CFD_pileup_threshold):
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

    return pile_up_event

# PSD Integration Script
def PSD_Integration(waveform, start, offset, stop, method):
    # Find maximum amplitude and its index
    imax = np.argmax(waveform)
    # Integration calculations
    if ((imax - start) > 0) and ((imax + stop) < len(waveform)):
        short_integral = 0; long_integral = 0
        waveform_short = waveform[imax + offset : imax + stop]; time_index_short = np.arange(0, len(waveform_short))
        waveform_long = waveform[imax - start : imax + stop]; time_index_long = np.arange(0, len(waveform_long))
        # Trapezoidal Integration
        if method == 1:
            short_integral = integrate.trapezoid(waveform_short, x = time_index_short)
            long_integral = integrate.trapezoid(waveform_long, x = time_index_long)
        # Composite Simpson's Rule
        if method == 2:
            short_integral = integrate.simpson(waveform_short, x = time_index_short)
            long_integral = integrate.simpson(waveform_long, x = time_index_long)
        # Rectangular Integration
        if method == 3:
            for i in waveform_short:
                short_integral += i
            for i in waveform_long:
                long_integral += i

    else:
        short_integral = -1
        long_integral = -1

    return short_integral, long_integral

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
        #print(f"{i}\t{waveform[i]}\t{CMA_trace[i]}" )
    return CMA_trace

# Helper function to read exact bytes
def read_exact(f, n):
    "Read exactly n bytes or raise EOFError."
    b = f.read(n)
    if len(b) < n:
        raise EOFError(f"Unexpected EOF (needed {n} bytes, got {len(b)})")
    return b

# This is the main function of the program
def load_file(run_number, channel_number, data_filepath):
    # Integration parameters, histogram ranges, gating parameters, and light curve fit parameters for each channel
    integration_start, integration_offset, integration_stop, integration_method, x_min, x_max, x_bins, y_min, y_max, y_bins, upper_gate, lower_gate, slices_in_region1, slices_in_region2, total_slices, a0, b0, c0, a1, b1, c1, a2, b2, c2 = setup(channel_number)
    calibrate = True

    # Constant Fraction Discrimmination Thresholds
    CFD_pileup_threshold = 0.45  # Threshold for pileup detection
    CFD_thinpulse_threshold = 0.8  # Threshold for thin pulse detection

    # Open binary file, and load the data from the binary file into a memory-map (mmap)
    filename = f"{data_filepath}DAQ_Compass/run_{run_number}/RAW/DataR_CH{channel_number}@V1730S_26980_run_{run_number}.BIN"
    file = open(filename,"rb")
    mm = mmap.mmap(file.fileno(), 0, access = mmap.ACCESS_READ)

    # Check header + 1st event data to 
    # ------------------------------------------
    # 1.) Know the data structure
    # 2.) Know if waveform are present  
    # 3.) Know how many events there are
    # ------------------------------------------
    mm_offset = 0 # Header is the first two bytes of the binary file
    header_bytes = mm[mm_offset:mm_offset+2]
    header = int.from_bytes(header_bytes, "little")

    # 1.) Checking energy long (Channel (2 bytes), MeV (8 bytes), or both) based on header flag
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

    # 2.) Checking if waveform data is present, and if it is, what is the length of the waveform based on the first event
    has_waveform = bool((header & 0x8) >> 3)
    if has_waveform:
        mm_offset = 2 + 2 + 2 + 8 + energy_long_size + 2 + 4 + 1  # header + board + channel + timestamp + energy_long + energy_short + flags + waveform_code
        waveform_length_bytes = mm[mm_offset:mm_offset+4] 
        waveform_length = int.from_bytes(waveform_length_bytes, "little")
        print(f"Waveform length of first event: {waveform_length} samples")

    # 3.) Calculating number of events in file
    size = os.path.getsize(filename)
    num_events_read = (size - 2) / (2 + 2 + 8 + 2 + 2 + 4 + 1 + 4 + waveform_length*2)  # approx estimate
    print(f"File size: {size} bytes")
    print(f"Number of events in file: {num_events_read}")
    events_int = int(num_events_read)
    print("--------------------------------------------------------")

    # Preallocate results
    TEvt = 0; BEvt = 0
    waveform = np.zeros(waveform_length); baseline = np.zeros(40)

    thin_pulse = np.zeros(events_int, dtype=bool)
    long_integral_list = np.zeros(events_int, dtype=np.float32)
    PSD = np.zeros(events_int, dtype=np.float32)
    index = 0

    if has_waveform:
        # Calculate PSD values for all events and store into arrays
        for k in range(events_int):
            PSD[k] = -1; long_integral_list[k] = -1 # Resetting PSD and long integral elements 
            # Collect waveform data for event 'k'
            waveform_start = 2 + (k + 1) * (23 + energy_long_size) + k*2*waveform_length
            waveform_end = waveform_start + waveform_length * 2
            waveform[:] = np.frombuffer(mm[waveform_start:waveform_end], dtype=np.uint16).astype(np.float32)
            waveform[:] = 15200 - waveform # this inverts the signal and estimates a baseline
            # Calculate a baseline and subtract it from the waveform
            baseline = waveform[:40].mean()
            waveform -= baseline
            # Compute first and second derivative of the waveform
            dx = 1  # Assuming uniform spacing
            derivative_waveform = np.gradient(waveform, dx)
            second_derivative_waveform = np.gradient(derivative_waveform, dx)
            # Check for pileup or thin pulse 
            pile_up_event = Pileup_Filtering(second_derivative_waveform, CFD_pileup_threshold)
            thin_pulse_event = Thin_Pulse_Filtering(waveform, channel_number, CFD_thinpulse_threshold)
            # PSD Integration 
            short_integral, long_integral = PSD_Integration(waveform, integration_start, integration_offset, integration_stop, integration_method)

            if ((long_integral > 0) and (short_integral > 0) and (not thin_pulse_event)):
                long_integral_list[k] = long_integral
                PSD[k] = short_integral / long_integral
                TEvt += 1
                percent_complete = ((TEvt + BEvt) / events_int) * 100
                Total_events_parsed = bool (percent_complete % 10)
                if not Total_events_parsed:
                    print(f"Percent complete: {percent_complete:.2f}%, good events parsed: {TEvt}")
                    pass
                    
            if ((long_integral <= 0) or (short_integral <= 0)):
                PSD[k] = -1; long_integral_list[k] = -1 
                BEvt += 1
        print(f"Finished calculating PSD for run={run_number}, channel={channel_number}. Good events: {TEvt}, Bad events: {BEvt}")
        print(f"Total events parsed: {TEvt + BEvt}")
    # Plotting Results
    xx = np.linspace(x_min+1, x_max, 400)
    fig = plt.figure()
    plt.hist2d(long_integral_list, PSD, bins = [x_bins, y_bins], range = [[x_min,x_max],[y_min,y_max]], cmap='viridis', norm = mcolors.LogNorm())
    plt.plot(xx, light_curve_function(xx, a0, b0, c0), label=f'Gamma upper fit (A = {a0:.2f}, B = {b0:.2f}, C = {c0:.2f})', lw=2, linestyle = '-', color = 'red')
    plt.plot(xx, light_curve_function(xx, a1, b1, c1), label=f'Neutron upper fit (A = {a1:.2f}, B = {b1:.2f}, C = {c1:.2f})', lw=2, linestyle = '--', color = 'red')
    plt.plot(xx, light_curve_function(xx, a2, b2, c2), label=f'Neutron lower fit (A = {a2:.2f}, B = {b2:.2f}, C = {c2:.2f})', lw=2, linestyle = '-.', color = 'red')
    plt.colorbar(location = 'right', label = 'counts', pad=0.01)
    plt.xlim(x_min, x_max)
    if calibrate:
        plt.xlabel('Light response (keVee)')
    else:
        plt.xlabel('Channel (a.u.)')
    plt.ylabel('PSD') 
    xticks = plt.xticks()[0]
    xlabels = [cal(x, channel_number, calibrate) for x in xticks]
    plt.xticks(xticks, [f"{lbl:.0f}" for lbl in xlabels])
    plt.legend()
    plt.savefig(f'{data_filepath}root/ch{channel_number}/run{run_number}.png', dpi=300)
    # # --- CRITICAL CLEANUP to avoid memory accumulation ---
    plt.close(fig)        # close the figure and free GUI memory
    gc.collect()          # force Python garbage collection of temporaries
    print("--------------------------------------------------------")

    return PSD, long_integral_list, events_int