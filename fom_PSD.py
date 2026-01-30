# -------------------------------------------------------------------------------------------------------------------- #
# In this program we will perform the figure of merit analysis (This part of the program takes a long time to run
# this is because it takes (offset * stop) combinations through all events for a given run and channel.
# To decrease computation time, you can decrease the range of offset and stop values, or increase the step size)
# -------------------------------------------------------------------------------------------------------------------- #
import os
import struct
import mmap
import gc
import argparse
import numpy as np
import Analysis 
import importlib
importlib.reload(Analysis)
from scipy import integrate
from collections import deque
from lmfit.models import GaussianModel
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable

from calibration_and_setup import calibration as cal
from calibration_and_setup import setup 

# Run and channel number from command line arguments
# e.g., python3 fom_PSD.py --run 1234 --channel 8
parser = argparse.ArgumentParser()
parser.add_argument("--run", type=int, required=True)
parser.add_argument("--channel", type=int, required=True)
args = parser.parse_args()
run_number = args.run
channel_number = args.channel
print(f"Running PSD with run={run_number} and channel={channel_number}")

# -------------------------------------------------------------------------------------------------------------------- #
# 1) In this block we will load the binary data file, check its header and first event to understand its structure
# -------------------------------------------------------------------------------------------------------------------- #
# Open binary file, and load the data from the binary file into a memory-map (mmap)
data_filepath = f'/home/jrufino/PhD_Research/data/'
filename = f"{data_filepath}/DAQ_Compass/run_{run_number}/RAW/DataR_CH{channel_number}@V1730S_26980_run_{run_number}.BIN"
file = open(filename,"rb")
mm = mmap.mmap(file.fileno(), 0, access = mmap.ACCESS_READ)

# -------------------------------------------------------------------------------------------------------------------- #
# Check header + 1st event data to 
# ------------------------------------------
# 1.) Know the data structure
# 2.) Know if waveform are present  
# 3.) Know how many events there are
# ------------------------------------------
mm_offset = 0 # Header is the first two bytes of the binary file
header_bytes = mm[mm_offset:mm_offset+2]
header = int.from_bytes(header_bytes, "little")

# Checking energy long (Channel (2 bytes), MeV (8 bytes), or both) based on header flag
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

# Checking if waveform data is present, and if it is, what is the length of the waveform based on the first event
has_waveform = bool((header & 0x8) >> 3)
if has_waveform:
    mm_offset = 2 + 2 + 2 + 8 + energy_long_size + 2 + 4 + 1  # header + board + channel + timestamp + energy_long + energy_short + flags + waveform_code
    waveform_length_bytes = mm[mm_offset:mm_offset+4] 
    waveform_length = int.from_bytes(waveform_length_bytes, "little")
    print(f"Waveform length of first event: {waveform_length} samples")

# Calculating number of events in file
size = os.path.getsize(filename)
num_events_read = (size - 2) / (2 + 2 + 8 + 2 + 2 + 4 + 1 + 4 + waveform_length*2)  # approx estimate
print(f"File size: {size} bytes")
print(f"Number of events in file: {num_events_read}")
events_int = int(num_events_read)

# -------------------------------------------------------------------------------------------------------------------- #
# 2) In this block we will setup the integration parameters, histogram ranges, gating parameters for each channel
# -------------------------------------------------------------------------------------------------------------------- #    
# Integration parameter space to sweep through
offset_low = 5; offset_high = 20; offset_step = 5
offset_range = range(offset_low, offset_high+1, offset_step); offset_length = len(offset_range)

stop_low = 40; stop_high = 190; stop_step = 50; 
stop_range = range(stop_low, stop_high+1, stop_step); stop_length = len(stop_range)

# Integration parameters 
integration_start = 10
# integration_offset = 17   
# integration_stop = 190
integration_method = 3

# Histogram parameters
x_min = 0; x_max = 20000
y_min = 0; y_max = 1
x_bins = 200; y_bins = 200

# PSD gate parameters
lower_gate = 0; upper_gate = x_max - 1000
slices_in_region1 = 10; slices_in_region2 = 5
total_slices = slices_in_region1 + slices_in_region2

# CFD thresholds and calibration flag
CFD_pileup_threshold = 0.45  # Threshold for pileup detection
CFD_thinpulse_threshold = 0.8  # Threshold for thin pulse detection
calibrate = False  # Whether to apply energy calibration or not

# -------------------------------------------------------------------------------------------------------------------- #
# 3) In this block we will perform the figure of merit analysis by sweeping through offset and stop combinations
# -------------------------------------------------------------------------------------------------------------------- #
# Filepath to save FOM results
fom_filepath = f'{data_filepath}/analyzed/ch{channel_number}/fom_analysis/'
if not os.path.exists(fom_filepath):
    os.makedirs(fom_filepath)

# Preallocate results during FOM analysis
TEvt = 0; BEvt = 0
waveform = np.zeros(waveform_length); baseline = np.zeros(40)
thin_pulse = np.zeros(events_int, dtype=bool)
long_integral_list = np.zeros(events_int, dtype=np.float32)
PSD = np.zeros(events_int, dtype=np.float32)

# Results to be saved
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
                pile_up_event = Analysis.Pileup_Filtering(second_derivative_waveform, CFD_pileup_threshold)
                thin_pulse_event = Analysis.Thin_Pulse_Filtering(waveform, channel_number, CFD_thinpulse_threshold)
                # PSD Integration 
                short_integral, long_integral = Analysis.PSD_Integration(waveform, integration_start, i, j, integration_method)

                if ((long_integral > 0) and (short_integral > 0) and (not thin_pulse_event)):
                    long_integral_list[k] = long_integral
                    PSD[k] = short_integral / long_integral

                    TEvt += 1
                    percent_complete = ((TEvt + BEvt) / events_int) * 100
                    Total_events_parsed = bool (percent_complete % 10)
                    if not Total_events_parsed:
                        print(f"Percent complete: {percent_complete:.2f}%, good events parsed: {TEvt}")
                if ((long_integral <= 0) or (short_integral <= 0)):
                    PSD[k] = -1; long_integral_list[k] = -1 
                    BEvt += 1
            print(f"Finished calculating PSD for offset={i}, stop={j}. Good events: {TEvt}, Bad events: {BEvt}")
            print(f"Total events parsed: {TEvt + BEvt}")
            
            # Perform a Gaussian fit in a gated region of the PSD histogram to calculate FOM
            if np.all(long_integral_list < 1):
                print("All elements in array are empty")
                FOM[index] = 0
                integration_offset[index] = i
                integration_stop[index] = j
                index +=1
            else:
                print("PSD completed, moving onto Gaussian fitting ")
                fom = Analysis.FOM_Analysis(long_integral_list, PSD, i , j, x_min, x_max, x_bins, y_min, y_max, y_bins, lower_gate, upper_gate, total_slices, slices_in_region1, slices_in_region2, channel_number, calibrate = False, save_filepath=fom_filepath)
                FOM[index] = fom
                integration_offset[index] = i
                integration_stop[index] = j
                index +=1
            # Free up memory
            gc.collect()

# Saving FOM parameter space in ascii columnar format
FOM = np.array(FOM)
with open(f"{fom_filepath}FOM_parameter_space.txt",'w') as file_out:
    for i in range(len(FOM)):
        file_out.write(f'{integration_offset[i]}\t{integration_stop[i]}\t{FOM[i]}\n')
