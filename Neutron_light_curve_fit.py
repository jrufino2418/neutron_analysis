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
import os

# Run and channel number
parser = argparse.ArgumentParser()
parser.add_argument("--run", type=int, required=True)
parser.add_argument("--channel", type=int, required=True)
args = parser.parse_args()
channel_number = args.channel
run_number = args.run
# Integration parameters, histogram ranges, gating parameters, and light curve fit parameters for each channel
integration_start, integration_offset, integration_stop, integration_method, x_min, x_max, x_bins, y_min, y_max, y_bins, upper_gate, lower_gate, slices_in_region1, slices_in_region2, total_slices, a0, b0, c0, a1, b1, c1, a2, b2, c2 = setup(channel_number)
calibrate = True

# Constant Fraction Discrimmination Thresholds
CFD_pileup_threshold = 0.45  # Threshold for pileup detection
CFD_thinpulse_threshold = 0.8  # Threshold for thin pulse detection

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

def light_curve_function(x, A, B, C):
    return A/np.sqrt(x) + B * np.exp(C * x)

# Light response 
def Light_response(PSD, long_integral_list):
    # Gate slices, projection of PSD -> Gaussian fit -> 
    x_slice = [0]*total_slices
    sigma_upper_n = [0]*total_slices; sigma_lower_n = [0]*total_slices
    index = 0; x_f1 = 0; x_mid = 0.4 * (upper_gate-lower_gate) + lower_gate
    for s in range(0, total_slices,1):
        if (s < slices_in_region1):
            x_low = s*(x_mid - lower_gate)/slices_in_region1 + lower_gate
            x_center = x_low + (x_mid - lower_gate)/(slices_in_region1*2)
            x_high = x_low + (x_mid - lower_gate)/slices_in_region1 - 150
            x_slice[index] = x_center
            x_f1 = x_high + 150
        if (s >= slices_in_region1 and s < total_slices):
            x_low = (s-slices_in_region1)*(upper_gate - x_mid)/slices_in_region2 + x_f1 
            x_center = x_low + (upper_gate - x_mid)/(slices_in_region2*2) 
            x_high = x_low + (upper_gate - x_mid)/slices_in_region2 - 150
            x_slice[index] = x_center     
        
        # Creating the mask to project the gates onto a 1D histogram (projecting onto PSD-axis)
        mask = (long_integral_list > x_low) & (long_integral_list < x_high)
        num_bins = 100
        hist_slice, bin_edges = np.histogram(PSD[mask], bins=num_bins, range=(0,1))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Checking to see if the gate has any counts on it
        if np.sum(hist_slice) > 0:
            # This function will perform the Gaussian fit on one peak
            result = fit_gaussians_lmfit(bin_centers, hist_slice, npeaks=2)

            # If the fit is successful the results will be plotted
            if result is not None:
                comps = result.eval_components(x=bin_centers)
                data0 = comps['g0_']; data1 = comps['g1_']
                g0_center = result.params['g0_center']; g0_sigma = result.params['g0_sigma']
                g1_center = result.params['g1_center']; g1_sigma = result.params['g1_sigma']
                if (g0_center > g1_center):
                    sigma_upper_n[index] = g0_center + 3*g0_sigma
                    sigma_lower_n[index] = g0_center - 3*g0_sigma
                else:
                    sigma_upper_n[index] = g1_center + 3*g1_sigma
                    sigma_lower_n[index] = g1_center - 3*g1_sigma
            else:
                sigma_upper_n[index] = 0; sigma_lower_n[index] = 0
        else:
            sigma_upper_n[index] = 0; sigma_lower_n[index] = 0
        index += 1

    # Fit
    p0 = [0.5, 0.5, 0]              # Initial guesses (important)
    popt, pcov = curve_fit(light_curve_function, x_slice, sigma_upper_n, p0=p0)
    A_upper, B_upper, C_upper = popt
    print(f'{A_upper}, {B_upper}, {C_upper}')
    popt1, pcov1 = curve_fit(light_curve_function, x_slice, sigma_lower_n, p0=p0)
    A_lower, B_lower, C_lower = popt1
    print(f'{A_lower}, {B_lower}, {C_lower}')
    xx = np.linspace(x_min+1, x_max, 400)

    # Plotting results
    plt.figure(1)
    plt.hist2d(long_integral_list, PSD, bins = [x_bins, y_bins], range = [[x_min,x_max],[y_min,y_max]], cmap='viridis', norm = mcolors.LogNorm())
    plt.plot(x_slice, sigma_upper_n, 'o', color = 'tab:red')
    plt.plot(xx, light_curve_function(xx, *popt), label=f'Neutron upper fit (A = {A_upper:.2f}, B = {B_upper:.2f}, C = {C_upper:.2f})', lw=2, color = 'tab:red')
    plt.plot(x_slice, sigma_lower_n, 'o', color = 'tab:orange')
    plt.plot(xx, light_curve_function(xx, *popt1), label=f'Neutron lower fit (A = {A_lower:.2f}, B = {B_lower:.2f}, C = {C_lower:.2f})', lw=2, color = 'tab:orange')
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
    plt.savefig(f'ch{channel_number}/PSD_light_curve_calibration.png', dpi=300)
    return 

# Calculating a Gaussian fit on the Compton Edge, and finding the calibration point
def Compton_edge_calibration(long_integral_list):
    # Gaussian fit on the Compton edge
    num_bins = x_bins
    Compton_edge_threshold = 35000
    bin_size = (x_max - x_min) / num_bins
    num_bins_compton_edge = int((x_max-Compton_edge_threshold)/bin_size)

    long_integral_4fit = [long_integral_list[i] for i in range(len(long_integral_list)) if ((long_integral_list[i] > Compton_edge_threshold) and (long_integral_list[i] < x_max))]
    hist_slice, bin_edges = np.histogram(long_integral_4fit, bins=num_bins_compton_edge, range=[Compton_edge_threshold, x_max])
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    if np.sum(hist_slice) > 0:
        result = fit_gaussians_lmfit(bin_centers, hist_slice, npeaks=1)
        if result is not None:
            # One way to get results from the fit (comps is the total component)
            comps = result.eval_components(x=bin_centers)
            data0 = comps['g0_']
            g0_center = result.params['g0_center']
            # Other way to get results from the fit
            A = result.params['g0_amplitude']
            mu = result.params['g0_center']
            sigma = result.params['g0_sigma']

            # 80% height point on high-energy side
            x_80 = mu + sigma * np.sqrt(-2 * np.log(0.8))

            print("80% height on high-energy side:", x_80)

    # Create the 1D Histogram of long integral
    plt.figure(1)
    #plt.title(rf'Cs-137 for Stilbene detector (channel {channel_number})', fontsize=16)
    plt.xlim(x_min, x_max)
    if calibrate:
        plt.xlabel('Light response (keVee)')
    else:
        plt.xlabel('Channel (a.u.)')
    plt.ylabel('Counts')
    plt.hist(long_integral_list, bins = num_bins, range = [x_min, x_max], histtype = 'step')
    if result is not None:
        plt.plot(bin_centers, result.best_fit, color = 'red', label='Gaussian fit')
        plt.axvline(x_80, color='red', linestyle='--', label='80% height')
        plt.legend()
    xticks = plt.xticks()[0]
    xlabels = [cal(x, channel_number, calibrate) for x in xticks]
    plt.xticks(xticks, [f"{lbl:.0f}" for lbl in xlabels])
    plt.savefig(f'ch{channel_number}/Cs137_calibration.png', dpi=300)
    return 

# Thin pulse filtering; only needed for channels 0 - 6 (small stilbene detectors)
def Thin_Pulse_Filtering(waveform, channel_number):
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

# Open binary file, and load the data from the binary file into a memory-map (mmap)
filename = f"/home/jrufino/PhD_Research/DAQ_Compass/run_{run_number}/RAW/DataR_CH{channel_number}@V1730S_26980_run_{run_number}.BIN"
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
        pile_up_event = Pileup_Filtering(second_derivative_waveform)
        thin_pulse_event = Thin_Pulse_Filtering(waveform, channel_number)
        # PSD Integration 
        short_integral, long_integral = PSD_Integration(waveform, integration_start, integration_offset, integration_stop, integration_method)

        if ((long_integral > 0) and (short_integral > 0) and (not thin_pulse_event)):
            long_integral_list[k] = long_integral
            PSD[k] = short_integral / long_integral
            TEvt += 1
            Total_events_parsed = bool (TEvt % 1000)
            if not Total_events_parsed:
                print(f"Total events parsed:\t{TEvt}")
        if ((long_integral <= 0) or (short_integral <= 0)):
            PSD[k] = -1; long_integral_list[k] = -1 
            BEvt += 1

#Light response fit 
Light_response(PSD, long_integral_list)

# plt.show()