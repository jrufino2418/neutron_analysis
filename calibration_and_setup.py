
# FOM parameters for each channel used in analysis
def fom(channel):
    if(channel == 8):
        # Integration parameters 
        integration_start = 10
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
        return integration_start, integration_method, x_min, x_max, x_bins, y_min, y_max, y_bins, upper_gate, lower_gate, slices_in_region1, slices_in_region2, total_slices, CFD_pileup_threshold, CFD_thinpulse_threshold, calibrate

# Setup parameters for each channel used in analysis
def setup(channel):
    if(channel == 0):
        # Integration parameters
        integration_start = 7
        integration_offset = 9
        integration_stop = 90
        integration_method = 3
        # Histogram parameters
        x_min = 0; x_max = 12000 
        y_min = 0; y_max = 1
        x_bins = 100; y_bins = 100
        # PSD gate parameters
        lower_gate = 2000; upper_gate = 10000
        slices_in_region1 = 20; slices_in_region2 = 3
        total_slices = slices_in_region1 + slices_in_region2
        # Light curve fitting parameters
        a, b, c = 15, 0.08, 0
        return integration_start, integration_offset, integration_stop, integration_method, x_min, x_max, x_bins, y_min, y_max, y_bins, upper_gate, lower_gate, slices_in_region1, slices_in_region2, total_slices, a, b, c

    if(channel == 8):
        # Integration parameters
        integration_start = 7
        integration_offset = 11
        integration_stop = 155
        integration_method = 3
        # Histogram parameters
        x_min = 0; x_max = 24000 
        y_min = 0; y_max = 1
        x_bins = 200; y_bins = 200
        # PSD gate parameters
        lower_gate = 4500; upper_gate = 21000
        slices_in_region1 = 10; slices_in_region2 = 5
        total_slices = slices_in_region1 + slices_in_region2
        # Light curve fitting parameters
        a0, b0, c0 = 7.1275791726727356, 0.16074051452650842, 1.599915006571205e-06 # Upper Gamma fit
        a1, b1, c1 = 11.326015596859731, 0.42350249026482845, -2.6271324558887407e-06
        a2, b2, c2 = -13.614293967877837, 0.42398668492222036, -2.8077234307179065e-06
        return integration_start, integration_offset, integration_stop, integration_method, x_min, x_max, x_bins, y_min, y_max, y_bins, upper_gate, lower_gate, slices_in_region1, slices_in_region2, total_slices, a0, b0, c0, a1, b1, c1, a2, b2, c2

    if(channel == 9):
        # Integration parameters
        integration_start = 7
        integration_offset = 17
        integration_stop = 190
        integration_method = 3
        # Histogram parameters
        x_min = 0; x_max = 22000 
        y_min = 0; y_max = 1
        x_bins = 200; y_bins = 200
        # PSD gate parameters
        lower_gate = 3000; upper_gate = 16500
        slices_in_region1 = 14; slices_in_region2 = 5
        total_slices = slices_in_region1 + slices_in_region2
        # Light curve fitting parameters
        a0, b0, c0 = 7.06086869252511, 0.14554205961636357, 2.324878746421244e-06 # Upper Gamma fit
        a1, b1, c1 =  9.643124673316269, 0.3903843858562569, -3.283568256243246e-06 
        a2, b2, c2 = -14.308438310834488, 0.4383946514319766, -6.639502580106621e-06
        return integration_start, integration_offset, integration_stop, integration_method, x_min, x_max, x_bins, y_min, y_max, y_bins, upper_gate, lower_gate, slices_in_region1, slices_in_region2, total_slices, a0, b0, c0, a1, b1, c1, a2, b2, c2

def calibration(long_integral_value, channel, calibration):
    if (channel == 0 and calibration == True):
        # E1 = 180; adc1 = 1.85e+3
        E1 = 0; adc1 = 0
        E2 = 478; adc2 = 3.156e+4
        slope = (E2-E1) / (adc2 - adc1)
        # b = E1 - slope * adc1
        b = 0
        return slope * long_integral_value + b
    if (channel == 8 and calibration == True):
        # E1 = 180; adc1 = 2.16e+3
        E1 = 0; adc1 = 0
        E2 = 478; adc2 = 41830.53687507425
        slope = (E2-E1) / (adc2 - adc1)
        #b = E1 - slope * adc1
        b = 0
        return slope * long_integral_value + b
    if (channel == 9 and calibration == True):
        # E1 = 180; adc1 = 2.16e+3
        E1 = 0; adc1 = 0
        E2 = 478; adc2 = 38037.77278721143
        slope = (E2-E1) / (adc2 - adc1)
        #b = E1 - slope * adc1
        b = 0
        return slope * long_integral_value + b

    else:
        return long_integral_value