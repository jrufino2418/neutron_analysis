
from calibration_and_setup import calibration as cal
from calibration_and_setup import cal_setup 
from calibration_and_setup import setup
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from lmfit.models import GaussianModel
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from scipy import integrate
from collections import deque
import numpy as np
import gc

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

# Light response 
def Neutron_light_response_fit(PSD, long_integral_list, x_min, x_max, x_bins, y_min, y_max, y_bins, lower_gate, upper_gate, total_slices, slices_in_region1, slices_in_region2, N_sigma, channel_number, calibrate, save_filepath):

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
                    sigma_upper_n[index] = g0_center + N_sigma*g0_sigma
                    sigma_lower_n[index] = g0_center - N_sigma*g0_sigma
                else:
                    sigma_upper_n[index] = g1_center + N_sigma*g1_sigma
                    sigma_lower_n[index] = g1_center - N_sigma*g1_sigma
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
    plt.savefig(f'{save_filepath}PSD_light_curve_calibration.png', dpi=300)
    return A_upper, B_upper, C_upper, A_lower, B_lower, C_lower

# Light response 
def Gamma_light_response_fit(PSD, long_integral_list, x_min, x_max, x_bins, y_min, y_max, y_bins, lower_gate, upper_gate, total_slices, slices_in_region1, slices_in_region2, N_sigma, channel_number, calibrate, save_filepath):
    # Gate slices, projection of PSD -> Gaussian fit -> 
    x_slice = [0]*total_slices; sigma_upper = [0]*total_slices; sigma_lower = [0]*total_slices
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
            result = fit_gaussians_lmfit(bin_centers, hist_slice, npeaks=1)

            # If the fit is successful the results will be plotted
            if result is not None:
                comps = result.eval_components(x=bin_centers)
                data0 = comps['g0_']
                g0_center = result.params['g0_center']; g0_sigma = result.params['g0_sigma']
                sigma_upper[index] = g0_center + N_sigma*g0_sigma
                sigma_lower[index] = g0_center - N_sigma*g0_sigma
            else:
                sigma_upper[index] = 0; sigma_lower[index] = 0
        else:
            sigma_upper[index] = 0; sigma_lower[index] = 0
        index += 1

    # Fit
    p0 = [0.5, 0.5, 0]              # Initial guesses (important)
    popt, pcov = curve_fit(light_curve_function, x_slice, sigma_upper, p0=p0)
    A_upper, B_upper, C_upper = popt
    print(f'{A_upper}, {B_upper}, {C_upper}')
    popt1, pcov1 = curve_fit(light_curve_function, x_slice, sigma_lower, p0=p0)
    A_lower, B_lower, C_lower = popt1
    print(f'{A_lower}, {B_lower}, {C_lower}')
    xx = np.linspace(x_min+1, x_max, 400)

    # Plotting results
    plt.figure(3)
    plt.hist2d(long_integral_list, PSD, bins = [x_bins, y_bins], range = [[x_min,x_max],[y_min,y_max]], cmap='viridis', norm = mcolors.LogNorm())
    plt.plot(x_slice, sigma_upper, 'o', color = 'tab:red')
    plt.plot(xx, light_curve_function(xx, *popt), label=f'Gamma upper fit (A = {A_upper:.2f}, B = {B_upper:.2f}, C = {C_upper:.2f})', lw=2, color = 'tab:red')
    plt.plot(x_slice, sigma_lower, 'o', color = 'tab:orange')
    plt.plot(xx, light_curve_function(xx, *popt1), label=f'Gamma lower fit (A = {A_lower:.2f}, B = {B_lower:.2f}, C = {C_lower:.2f})', lw=2, color = 'tab:orange')
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
    plt.savefig(f'{save_filepath}Cs137_PSD_light_response.png', dpi=300)
    return A_upper, B_upper, C_upper

# Calculating a Gaussian fit on the Compton Edge, and finding the calibration point
def Compton_edge_calibration(long_integral_list, x_min, x_max, x_bins, channel_number, calibrate, save_filepath):
    # Gaussian fit on the Compton edge
    num_bins = x_bins
    Compton_edge_threshold = 32000
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
    plt.figure(2)
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
    plt.savefig(f'{save_filepath}Cs137_calibration.png', dpi=300)
    return 

# Figure of merit analysis
def FOM_Analysis(long_integral_list, PSD, offset, stop, x_min, x_max, x_bins, y_min, y_max, y_bins, lower_gate, upper_gate, total_slices, slices_in_region1, slices_in_region2, channel_number, calibrate, save_filepath):
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
    plt.savefig(f"{save_filepath}FOM_offset{offset}_stop{stop}.png", dpi=300)

    # --- CRITICAL CLEANUP to avoid memory accumulation ---
    plt.close(fig)        # close the figure and free GUI memory
    gc.collect()          # force Python garbage collection of temporaries

    return fom

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
    elif ((imax - start) > 0) and ((imax + stop) > len(waveform)):
        short_integral = 0; long_integral = 0
        waveform_short = waveform[imax + offset : ]; time_index_short = np.arange(0, len(waveform_short))
        waveform_long = waveform[imax - start : ]; time_index_long = np.arange(0, len(waveform_long))
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
