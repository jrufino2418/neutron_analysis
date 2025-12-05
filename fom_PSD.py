import os
import struct
import mmap
import gc
import argparse
import numpy as np
from scipy import integrate
from collections import deque
from lmfit.models import GaussianModel
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable

from calibration_and_setup import calibration as cal
from calibration_and_setup import setup 

# Run and channel number
parser = argparse.ArgumentParser()
parser.add_argument("--run", type=int, required=True)
parser.add_argument("--channel", type=int, required=True)
args = parser.parse_args()
run_number = args.run
channel_number = args.channel
print(f"Running PSD with run={run_number} and channel={channel_number}")

# Integration parameters, histogram ranges, gating parameters, and light curve fit parameters for each channel
integration_start, integration_offset, integration_stop, integration_method, x_min, x_max, x_bins, y_min, y_max, y_bins, upper_gate, lower_gate, slices_in_region1, slices_in_region2, total_slices, a0, b0, c0, a1, b1, c1, a2, b2, c2 = setup(channel_number)
calibrate = False

offset_low = 5; offset_high = 20; offset_step = 5
offset_range = range(offset_low, offset_high+1, offset_step); offset_length = len(offset_range)

stop_low = 40; stop_high = 190; stop_step = 50; 
stop_range = range(stop_low, stop_high+1, stop_step); stop_length = len(stop_range)

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

# Figure of merit analysis
def FOM_Analysis(long_integral_list, PSD, offset, stop):
    # Initializing figure to plot the FOM data
    fig, ((ax1, ax2, ax3)) = plt.subplots(3,1,figsize=(8, 13.5))
    # fig.suptitle(rf'$E_n$ = 1 MeV - Run {run_number} - Channel {channel_number} - Offset {offset} - Stop {stop} - Integration Method {integration_method}', fontsize=16)
    fig.subplots_adjust(hspace=0)

    # Force ax1 and ax2 to touch exactly
    pos1 = ax1.get_position()
    pos4 = ax2.get_position()
    pos3 = ax3.get_position()
    # Make bottom of ax1 = top of ax2
    ax1.set_position([pos1.x0, pos4.y1, pos1.width, pos1.height])
    ax3.set_position([pos3.x0, pos3.y0 - 0.06, pos3.width, pos3.height])

    labels = ['(a)', '(b)', '(c)']
    axes = [ax1, ax2, ax3]

    for ax, label in zip(axes, labels):
        ax.text(
            # 0.925, 0.95, label,            # (x, y) in axes coords
            0.02, 0.95, label,            # (x, y) in axes coords
            transform=ax.transAxes,       
            fontsize=14,
            fontweight='bold',
            va='top', ha='left'
        )

    # (ax1) PSD Data and slices/gates are in the first plot
    h1 = ax1.hist2d(long_integral_list, PSD, bins = [x_bins, y_bins], range = [[x_min,x_max],[y_min,y_max]], cmap='viridis', norm = mcolors.LogNorm())
    #ax1.set_xlabel('long-integral')
    ax1.set_xlim(x_min, x_max)
    xticks = ax1.get_xticks()
    xlabels = [cal(x, channel_number, calibrate) for x in xticks]
    ax1.set_xticks(xticks)
    ax1.set_xticklabels([f"{lbl:.0f}" for lbl in xlabels])
    ax1.set_ylabel('PSD')
    fig.colorbar(h1[3], location = 'top', label = 'counts',ax=ax1, pad=0.01)

    # (ax3) Gate slices, projection of PSD -> Gaussian fit -> FOM
    x_fom = [0]*total_slices
    fom_slice = [0]*total_slices; index = 0; x_f1 = 0; x_mid = 0.4 * (upper_gate-lower_gate) + lower_gate
    for s in range(0, total_slices,1):
        if (s < slices_in_region1):
            x_low = s*(x_mid - lower_gate)/slices_in_region1 + lower_gate
            x_center = x_low + (x_mid - lower_gate)/(slices_in_region1*2)
            x_high = x_low + (x_mid - lower_gate)/slices_in_region1 - 150
            x_fom[index] = x_center
            x_f1 = x_high + 150
        if (s >= slices_in_region1 and s < total_slices):
            x_low = (s-slices_in_region1)*(upper_gate - x_mid)/slices_in_region2 + x_f1 
            x_center = x_low + (upper_gate - x_mid)/(slices_in_region2*2) 
            x_high = x_low + (upper_gate - x_mid)/slices_in_region2 - 150
            x_fom[index] = x_center     

        # Vertical/upright rectangular plots to represent the gates on the 2D Histogram
        ax1.axvspan(x_low, x_high, color='black', alpha=0.2, label="Full gate")
        if index == 0:
            ax1.axvspan(x_low, x_high, color='blue', alpha=0.5, label="Full gate")
        if index == (total_slices-1):
            ax1.axvspan(x_low, x_high, color='red', alpha=0.5, label="Full gate")
        
        # Creating the mask to project the gates onto a 1D histogram (projecting onto PSD-axis)
        mask = (long_integral_list > x_low) & (long_integral_list < x_high)
        num_bins = 100
        hist_slice, bin_edges = np.histogram(PSD[mask], bins=num_bins, range=(0,1))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Checking to see if the gate has any counts on it
        if np.sum(hist_slice) > 0:
            # This function will perform the Gaussian fit on the two peaks
            result = fit_gaussians_lmfit(bin_centers, hist_slice, npeaks=2)

            # If the fit is successful the results will be plotted and the figure of merit will be calculated
            if result is not None:
                comps = result.eval_components(x=bin_centers)
                data0 = comps['g0_']; data1 = comps['g1_']
                g0_center = result.params['g0_center']; g1_center = result.params['g1_center']
                if(g0_center.value > g1_center.value):
                    if index == 0:
                        ax3.hist(PSD[mask], bins = num_bins, range=(0,1), alpha = 0.5, color = 'blue')
                        ax3.plot(bin_centers, result.best_fit, color = 'blue', lw=2)
                    if index == (total_slices-1):
                        ax3.hist(PSD[mask], bins = num_bins, range=(0,1), alpha = 0.5, color = 'red')
                        ax3.plot(bin_centers, result.best_fit, color = 'red', lw=2)
                    Neutron_band = data0; n_center = result.params['g0_center']; n_fwhm = result.params['g0_fwhm']
                    Gamma_band = data1; g_center = result.params['g1_center']; g_fwhm = result.params['g1_fwhm']
                if(g1_center.value > g0_center.value):
                    if index == 0:
                        ax3.hist(PSD[mask], bins = num_bins, range=(0,1), alpha = 0.5, color = 'blue')
                        ax3.plot(bin_centers, result.best_fit, color = 'blue', lw=2)
                    if index == (total_slices-1):
                        ax3.hist(PSD[mask], bins = num_bins, range=(0,1), alpha = 0.5, color = 'red')
                        ax3.plot(bin_centers, result.best_fit, color = 'red', lw=2)
                    Neutron_band = data1; n_center = result.params['g1_center']; n_fwhm = result.params['g1_fwhm']
                    Gamma_band = data0; g_center = result.params['g0_center']; g_fwhm = result.params['g0_fwhm']
                
                # Calculating the FOM from the Gaussian fit results
                fom_s = abs(n_center - g_center) / (g_fwhm + n_fwhm)
            else:
                fom_s = 0
        else:
            fom_s = 0
        fom_slice[index] = fom_s
        index += 1

    # Calculating the FOM for the whole gated region
    mask = (long_integral_list > lower_gate) & (long_integral_list < upper_gate)
    hist, bin_edges = np.histogram(PSD[mask], bins=100, range=(0,1))
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    if np.sum(mask) > 0:
        # --- Fit Gaussians with lmfit ---
        result = fit_gaussians_lmfit(bin_centers, hist, npeaks=2)
        if result is not None:
            comps = result.eval_components(x=bin_centers)
            # print(result.fit_report())
            print("--------------------------------------------------------------------------------------------------------------------")
            data0 = comps['g0_']; data1 = comps['g1_']
            g0_center = result.params['g0_center']; g1_center = result.params['g1_center']
            if(g0_center.value > g1_center.value):
                Neutron_band = data0
                n_center = result.params['g0_center']
                n_fwhm = result.params['g0_fwhm']
                Gamma_band = data1
                g_center = result.params['g1_center']
                g_fwhm = result.params['g1_fwhm']
            if(g1_center.value > g0_center.value):
                Neutron_band = data1
                n_center = result.params['g1_center']
                n_fwhm = result.params['g1_fwhm']
                Gamma_band = data0
                g_center = result.params['g0_center']
                g_fwhm = result.params['g0_fwhm']
            fom = abs(n_center - g_center) / (g_fwhm + n_fwhm)
        else:
            fom = 0

    ax3.set_title("Projection of PSD gates (slices) and Gaussian fits")
    ax3.set_xlabel("PSD")
    ax3.set_ylabel("Counts")

    # (ax2) Scatter plot of the FOM calculations vs slice 
    ax2.scatter(x_fom, fom_slice, color = 'black')
    ax2.scatter(x_fom[0], fom_slice[0], color = 'blue')
    ax2.scatter(x_fom[total_slices-1], fom_slice[total_slices-1], color = 'red')

    ax2.set_xlim(x_min, x_max)
    # ax2.set_ylim(0, 2.5)
    xticks = ax2.get_xticks()
    xlabels = [cal(x, channel_number, calibrate) for x in xticks]
    ax2.set_xticks(xticks)
    ax2.set_xticklabels([f"{lbl:.0f}" for lbl in xlabels])
    ax2.set_xlabel("Light response (keVee)")
    ax2.set_ylabel("Figure-of-Merit")

    # Saving plot, to compare with FOM 2D histogram with different integration parameters
    plt.savefig(f'/home/jrufino/PhD_Research/data/ch{channel_number}/run{run_number}/FOM_offset{offset}_stop{stop}.png', dpi=300)

    # --- CRITICAL CLEANUP to avoid memory accumulation ---
    plt.close(fig)        # close the figure and free GUI memory
    gc.collect()          # force Python garbage collection of temporaries

    return fom

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
#filename = f"/afs/crc.nd.edu/group/nsl/ast/vol/astVol05/14Cxn/DAQ/run_{run_number}/RAW/DataR_CH{channel_number}@V1730S_26980_run_{run_number}.BIN"
filename = f"/home/jrufino/PhD_Research/DAQ_Compass/run_{run_number}/RAW/DataR_CH{channel_number}@V1730S_26980_run_{run_number}.BIN"; print(f"Loading {filename}...")
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

# Preallocate results
TEvt = 0; BEvt = 0
waveform = np.zeros(waveform_length); baseline = np.zeros(40)

thin_pulse = np.zeros(events_int, dtype=bool)
long_integral_list = np.zeros(events_int, dtype=np.float32)
PSD = np.zeros(events_int, dtype=np.float32)

integration_offset = np.zeros(offset_length * stop_length, dtype=np.int32)
integration_stop  = np.zeros(offset_length * stop_length, dtype=np.int32)
FOM = np.zeros(offset_length * stop_length, dtype=np.float32)
index = 0

if has_waveform:
    for i in offset_range:
        for j in stop_range:
            print("--------------------------------------------------------")
            TEvt = 0; BEvt = 0
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
                short_integral, long_integral = PSD_Integration(waveform, integration_start, i, j, integration_method)

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

            # Perform a Gaussian fit in a gated region of the PSD histogram to calculate FOM
            if np.all(long_integral_list < 1):
                print("All elements in array are empty")
                FOM[index] = 0
                integration_offset[index] = i
                integration_stop[index] = j
                index +=1
            else:
                print("PSD completed, moving onto Gaussian fitting ")
                fom = FOM_Analysis(long_integral_list, PSD, i , j)
                FOM[index] = fom
                integration_offset[index] = i
                integration_stop[index] = j
                index +=1

# Saving FOM parameter space in ascii columnar format
FOM = np.array(FOM)
with open(f"/home/jrufino/PhD_Research/data/ch{channel_number}/run{run_number}/FOM_parameter_space.txt",'w') as file_out:
    for i in range(len(FOM)):
        file_out.write(f'{integration_offset[i]}\t{integration_stop[i]}\t{FOM[i]}\n')
