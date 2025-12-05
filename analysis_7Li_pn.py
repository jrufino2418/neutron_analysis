from calibration_and_setup import setup 
import matplotlib.pyplot as plt
from PSD import load_file
import numpy as np
import csv
import os
import ROOT

# Function for the light curve 
def light_curve_function(x, A, B, C):
    return A/np.sqrt(x) + B * np.exp(C * x)

# --------------------------------------------------------------------------------------------------------------------#
# Channel number to analyze
channel_number = 9
data_filepath = f'/home/jrufino/PhD_Research/data/'

# Integration parameters, histogram ranges, gating parameters, and light curve fit parameters for individual channel
integration_start, integration_offset, integration_stop, integration_method, x_min, x_max, x_bins, y_min, y_max, y_bins, upper_gate, lower_gate, slices_in_region1, slices_in_region2, total_slices, a0, b0, c0, a1, b1, c1, a2, b2, c2 = setup(channel_number)
calibrate = True

# --------------------------------------------------------------------------------------------------------------------#
# Create output root file for histograms and tree
f = ROOT.TFile(f"{data_filepath}root/ch{channel_number}/analysis_7Li_pn_ch{channel_number}.root", "RECREATE")
tree = ROOT.TTree("RunTree", "Tree storing histograms per run")

# Create branches for the ROOT histograms and scalars for each run
from array import array
Run = array('i', [0])
E_proton = array('f', [0.])
E_neutron = array('f', [0.])
PSD_vs_Light = ROOT.TH2F()   # placeholder; overwritten each event
LightProj    = ROOT.TH1F()   # placeholder; overwritten each event

tree.Branch("Run", Run, "Run/I")
tree.Branch("E_proton", E_proton, "E_proton/F")
tree.Branch("E_neutron", E_neutron, "E_neutron/F")
tree.Branch("PSD_vs_Light", PSD_vs_Light)
tree.Branch("LightProj", LightProj)


# --------------------------------------------------------------------------------------------------------------------#
# Scalars
total_events = 409
run_arr, time_arr, target_charge, suppression_charge = [0]* total_events, [0]* total_events, [0]* total_events, [0]* total_events
for run_num in range(0, total_events):  # Adjust range as needed
    filename = f'{data_filepath}DAQ_Scalars/14C_xn_Sept2025_run{run_num:03d}.dat'
    if not os.path.exists(filename):
        continue
    try:
        # Use genfromtxt to handle trailing commas and missing values
        data = np.genfromtxt(filename, delimiter=',')
        # Remove any empty columns caused by trailing commas
        if data.ndim == 1:
            data = data.reshape(-1, 3)
        elif data.shape[1] > 3:
            data = data[:, :3]
        total_time = data[-1, 0]  # Use last time value
        sum_charge1 = np.sum(data[:, 1])
        sum_charge2 = np.sum(data[:, 2])
        run_arr[run_num] = run_num
        time_arr[run_num] = total_time
        target_charge[run_num] = sum_charge1
        suppression_charge[run_num] = sum_charge2
        print(f'{run_num}, {total_time}, {sum_charge1}, {sum_charge2}')
    except Exception as e:
        print(f"Error reading {filename}: {e}")

# --------------------------------------------------------------------------------------------------------------------#
# elog file: 14C(x,n) - September 8, 2025 - 7Li(p,n).csv
# lower_run_number = 95
# upper_run_number = 141
lower_run_number = 114
upper_run_number = 118
num_runs = upper_run_number - lower_run_number - 1
run = [0]*num_runs; NMR = [0]*num_runs; E_beam = [0]*num_runs; neutron_yield_analyzed = [0]*num_runs

# Create 3D histogram for LightProj vs Run (can be visualized in 3D)
LightProj_vs_Run = ROOT.TH2F("Light response projection vs Run", "LightProj vs Run;Run;LightProj;Counts", num_runs, lower_run_number, upper_run_number, x_bins, x_min, x_max)

with open(f'{data_filepath}DAQ_Scalars/14C(x,n) - September 8, 2025 - 7Li(p,n).csv', mode='r') as csv_file:
    csv_reader = csv.DictReader(csv_file) # Uses the first row as fieldnames
    for row in csv_reader:
        # Access columns by name:
        run_number_csv = row['Run'].strip() # Remove any leading/trailing whitespace
        NMR_csv = row['NMR']
        E_beam_csv = row['Beam Energy (MeV)']

        if not NMR_csv or NMR_csv.lower() == 'x':
            continue  # Skip empty or invalid run numbers
        
        else:
            run_num = int(run_number_csv)
            target_scalars = target_charge[run_num]
            if ((target_scalars > 0) and (lower_run_number < run_num < upper_run_number)):
                # Data from e-log
                index = run_num - lower_run_number - 1
                run[index] = run_num
                NMR[index] = float(NMR_csv)
                E = float(E_beam_csv)
                E_beam[index] = E - 0.0

                # Data from analysis and scalars
                PSD, Light_Response, events_int = load_file(run_num, channel_number, data_filepath)

                Run[0] = run_num
                E_proton[0] = E

                # Create new TH2F for this run
                PSD_vs_Light = ROOT.TH2F(f"PSD_run{run_num}", "PSD vs Light", x_bins, x_min, x_max, y_bins, y_min, y_max)
                LightProj = ROOT.TH1F(f"LightProj_run{run_num}", "Gated Light Projection", x_bins, x_min, x_max)

                # Obtaining neutron yield from the gates and neutron band projection onto light response axis
                neutron_yield = 0
                for k in range(events_int):
                    # Fill PSD vs Light histogram
                    psd = PSD[k]
                    light_response = Light_Response[k]
                    PSD_vs_Light.Fill(light_response, psd)
                    # Check if the event falls within the neutron region gate and fill projection
                    gamma_upper_band = light_curve_function(light_response, a0, b0, c0)
                    neutron_upper_band = light_curve_function(light_response, a1, b1, c1)
                    neutron_lower_band = light_curve_function(light_response, a2, b2, c2)
                    if ((light_response < upper_gate) and (psd > gamma_upper_band) and (psd > neutron_lower_band) and (psd < neutron_upper_band)):
                        neutron_yield +=1
                        LightProj.Fill(light_response)
                        LightProj_vs_Run.Fill(run_num, light_response)

                neutron_yield_analyzed[index] = neutron_yield / target_scalars if target_scalars > 0 else 0

                # Set the branch addresses to the new histograms
                tree.SetBranchAddress("PSD_vs_Light", PSD_vs_Light)
                tree.SetBranchAddress("LightProj", LightProj)
                tree.Fill() # Fill the tree with current run data
                # Write histograms to file so they appear in TBrowser
                PSD_vs_Light.Write()
                LightProj.Write()

                # Clean up to avoid memory leak
                del PSD_vs_Light
                del LightProj
                del PSD
                del Light_Response
                del events_int
                print(f"Index: {index}, Run: {run[index]}, NMR: {NMR[index]}, E_beam: {E_beam[index]}, neutron_yield: {neutron_yield_analyzed[index]}")

# Normalize each row (each run) independently
for run_bin in range(1, LightProj_vs_Run.GetNbinsX() + 1):
    # Get the integral for this run (row)
    row_integral = 0.0
    for lightproj_bin in range(1, LightProj_vs_Run.GetNbinsY() + 1):
        row_integral += LightProj_vs_Run.GetBinContent(run_bin, lightproj_bin)
    
    # Normalize each bin in this row
    if row_integral > 0:
        for lightproj_bin in range(1, LightProj_vs_Run.GetNbinsY() + 1):
            normalized_value = LightProj_vs_Run.GetBinContent(run_bin, lightproj_bin) / row_integral
            LightProj_vs_Run.SetBinContent(run_bin, lightproj_bin, normalized_value)

# Write the 2D histogram and tree to file
LightProj_vs_Run.Write()
tree.Write()
f.Close()
# --------------------------------------------------------------------------------------------------------------------#
# Burke et al. data points for comparison 
with open(f'{data_filepath}DAQ_Scalars/7Li_pn_Burke.txt', mode='r') as file:
    lines = file.readlines()
    burke_Ebeam = []
    burke_yield = []
    burke_yield_err = []
    for line in lines:
        parts = line.split()
        if len(parts) >= 3:
            burke_Ebeam.append(float(parts[0]))
            burke_yield.append(float(parts[1]))
            burke_yield_err.append(float(parts[2]))

# Find max neutron yield
max_yield_analyzed = max(neutron_yield_analyzed)
max_burke_yield = max(burke_yield)

# Normalize neutron yield to Burke et al.
for i in range(num_runs):
    # make yield normalized to max
    if neutron_yield_analyzed[i] != 0:
        normalized_yield = neutron_yield_analyzed[i] / max_yield_analyzed * max_burke_yield
        neutron_yield_analyzed[i] = normalized_yield


plt.figure(1, figsize=(10, 6))
plt.scatter(E_beam, neutron_yield_analyzed, color='blue', label='This work (normalized to Burke)')
plt.scatter(burke_Ebeam, burke_yield, color='red', label='Burke et al. (1974)', marker='x')
#plt.title('Neutron Yield vs Beam Energy for $^{7}$Li$(p,n)$ Reaction', fontsize=16)
plt.xlabel(r'$E_p$ (MeV)', fontsize=14); plt.xlim(1.8, 2.8)
plt.ylabel(r'$d\sigma/d\Omega$ (mb/sr)', fontsize=14)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(f'{data_filepath}root/ch{channel_number}/Neutron_Yield_vs_Beam_Energy_7Li_pn.png', dpi=300)
# plt.show()