import struct
import numpy as np
import matplotlib.pyplot as plt

# Channel number to read
run_number = 1
channel_number = 0

# Filename
filename = f"/home/jrufino/PhD_Research/DAQ_Compass/run_{run_number}/RAW/DataR_CH{channel_number}@V1730S_26980_run_{run_number}.BIN"

# Read first event and dump all info
traces = np.array([])   # to store waveform samples
with open(filename, "rb") as file:
    # Reading header (2 bytes) (first event only)
    header16 = file.read(2)
    header = struct.unpack('<H', header16)[0]
    print(f"Header (16-bit): 0x{header:04X} (bin: {header:016b}, decimal: {header})")

    # Reading board number (2 bytes), channel number (2 bytes), timestamp (8 bytes)
    board16 = file.read(2)
    print(f"check 0: {not (board16)}") # True if EOF
    if not board16:
        print("Hit EOF")
        exit(1)
    channel16 = file.read(2)
    timestamp64 = file.read(8)

    # Reading energy long (Channel (2 bytes), MeV (8 bytes), or both) based on header flags
    check1 = (header & 0x3) == 0x3
    check2 = (header & 0x2) >> 1
    check3 = (header & 0x1)
    print(f"check 1:  {check1}")
    print(f"check 2:  {check2}")
    print(f"check 3:  {check3}")
    if (header & 0x3) == 0x3:
        print("Both energy long Ch and MeV present")
        energy_long_Ch16 = file.read(2)
        energy_long_MeV64 = file.read(8)
    elif (header & 0x2) >> 1:
        print("Only energy long MeV present")
        energy_long_MeV64 = file.read(8)
        energy_long_Ch16 = None
    elif (header & 0x1):
        print("Only energy long Ch present")
        energy_long_MeV64 = None
        energy_long_Ch16 = file.read(2)

    # Reading energy short (2 bytes) and flags (4 bytes)
    energy_short_ADC16 = file.read(2)
    flags32 = file.read(4)
    
    # Reading waveform if present
    print(f"check 4: {header & 0x8 >> 3}")
    if header & 0x8 >> 3:
        waveform_code8 = file.read(1)
        waveform_length32 = file.read(4)
        waveform_length = struct.unpack('<I', waveform_length32)[0]
    
    # Reading waveform samples/traces (2 bytes each) and storing in numpy array
    for i in range(waveform_length):  # Read up to 1000 samples or until EOF
        trace16 = file.read(2)
        if len(trace16) < 2:
            break
#        trace = 16383 - struct.unpack('<H', trace16)[0]
        trace = struct.unpack('<H', trace16)[0]
        traces = np.append(traces, trace)

    board = struct.unpack('<H', board16)[0]
    channel = struct.unpack('<H', channel16)[0]
    timestamp = struct.unpack('<Q', timestamp64)[0]
    energy_long_Ch = struct.unpack('<H', energy_long_Ch16)[0] if energy_long_Ch16 else None
    energy_long_MeV = struct.unpack('<Q', energy_long_MeV64)[0] if energy_long_MeV64 else None
    energy_short_ADC = struct.unpack('<H', energy_short_ADC16)[0]
    flags = struct.unpack('<I', flags32)[0] 
    waveform_code = struct.unpack('<B', waveform_code8)[0] 
    
    print(f"Board number: {board}")
    print(f"Channel number: {channel}")
    print(f"Timestamp: {timestamp}")
    print(f"Energy long (Ch): {energy_long_Ch}")
    print(f"Energy long (MeV): {energy_long_MeV}")
    print(f"Energy short (ADC): {energy_short_ADC}")
    print(f"Flags: {flags}")
    print(f"Waveform code: {waveform_code}")
    print(f"Waveform length: {waveform_length} samples")
    print(f"First 10 waveform samples (raw 16-bit): {traces[:10]}")
    print(f"Total waveform samples read: {len(traces)}")
    print("Done.")
    # Now 'traces' contains all waveform samples as a numpy array
    # Plot the waveform samples
    if len(traces) > 0:
        plt.figure(figsize=(8,4))
        plt.plot(range(len(traces)), traces, drawstyle='steps-mid', marker='o')
        plt.title(f'First Raw Waveform')
        plt.xlabel('Sample #')
        plt.ylabel('ADC value (raw 16-bit)')
        plt.grid(True)
        plt.tight_layout()
        plt.show()