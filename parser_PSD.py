import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
from matplotlib.widgets import Button
import matplotlib.image as mpimg
import numpy as np
import argparse
import os

# Run and channel number
parser = argparse.ArgumentParser()
parser.add_argument("--run", type=int, required=True)
parser.add_argument("--channel", type=int, required=True)
args = parser.parse_args()
run_number = args.run
channel_number = args.channel
print(f"Running PSD with run={run_number} and channel={channel_number}")

offset_low = 5; offset_high = 20; offset_step = 5
offset_range = range(offset_low, offset_high+1, offset_step); offset_length = len(offset_range)

stop_low = 40; stop_high = 190; stop_step = 50; 
stop_range = range(stop_low, stop_high+1, stop_step); stop_length = len(stop_range)

offset_values = list(offset_range)   # horizontal next/prev
stop_values = list(stop_range)    # vertical next/prev
method_value = 3                    # start method

current_offset_idx = 0
current_stop_idx = 0

# --- DATA LOADING FUNCTION ---
def load_data(offset, stop):
    # Loading FOM 2D histogram data
    filename = f'/home/jrufino/PhD_Research/data/ch{channel_number}/run{run_number}/FOM_parameter_space.txt'
    if not os.path.exists(filename):
        print(f"File not found: {filename}")
        FOM_data, offset_index, stop_index = [], [], []
    else:
        FOM_data = []
        offset_index = []
        stop_index = []
        with open(filename, 'r') as file:
            for line in file:
                if line.strip():
                    parts = line.split()
                    try:
                        offset_index.append(float(parts[0]))
                        stop_index.append(float(parts[1]))
                        FOM_data.append(float(parts[2]))
                    except (ValueError, IndexError):
                        continue
        
    # Loading the PSD data from images
    file_png = f'/home/jrufino/PhD_Research/data/ch{channel_number}/run{run_number}/FOM_offset{offset}_stop{stop}.png'
    if not os.path.exists(file_png):
        img = False
        print(f"File not found: {file_png}")
    else:
        img = mpimg.imread(file_png)

    return np.array(FOM_data), np.array(offset_index), np.array(stop_index), img

# --- PLOT SETUP ---
fig, ax1 = plt.subplots(1,1, sharex = False, figsize=(8, 6))
# plt.subplots_adjust(bottom=0.11)  # leave space for widgets
# plt.subplots_adjust(left=0.15)
# --- INITIAL PLOT ---
# Load initial data
FOM_data, offset_data, stop_data, image = load_data(
    offset_values[current_offset_idx], 
    stop_values[current_stop_idx])
# Plot 2D histogram
h = ax1.hist2d(
    offset_data, stop_data, 
    bins=[offset_length, stop_length],
    weights = FOM_data,
    range=[[offset_low - offset_step/2, offset_high + offset_step/2],[stop_low - stop_step/2, stop_high + stop_step/2]],
    cmap='cividis')
cbar = fig.colorbar(h[3], label='Count')  # <== create once
rect = patches.Rectangle((offset_values[current_offset_idx] - offset_step/2, stop_values[current_stop_idx] - stop_step/2),
              offset_step, stop_step, angle = 0, facecolor = 'none', edgecolor = 'red', linewidth = 2)
ax1.add_patch(rect)
ax1.set_xlabel('Integration Offset')
ax1.set_ylabel('Integration Stop')
ax1.set_title(f'FOM Heatmap (offset = {offset_values[current_offset_idx]}, stop = {stop_values[current_stop_idx]})')

def update_plot():
    global image_fig, image_ax  # keep track of image figure

    ax1.clear()
    # Update data
    FOM_data, offset_data, stop_data, image = load_data(
        offset_values[current_offset_idx], 
        stop_values[current_stop_idx])
    # Update 2D histogram
    h = ax1.hist2d(
        offset_data, stop_data, 
        bins=[offset_length, stop_length],
        weights = FOM_data,
        range=[[offset_low - offset_step/2, offset_high + offset_step/2],[stop_low - stop_step/2, stop_high + stop_step/2]],
        cmap='cividis')
    rect = patches.Rectangle((offset_values[current_offset_idx] - offset_step/2, stop_values[current_stop_idx] - stop_step/2),
              offset_step, stop_step, angle = 0, facecolor = 'none', edgecolor = 'red', linewidth = 2)
    ax1.add_patch(rect)
    ax1.set_xlabel('Integration Offset')
    ax1.set_ylabel('Integration Stop')
    ax1.set_title(f'FOM Heatmap (offset = {offset_values[current_offset_idx]}, stop = {stop_values[current_stop_idx]})')
    fig.canvas.draw_idle()
    
    if image is not None:
        # Create new figure the first time
        if 'image_fig' not in globals() or image_fig is None:
            image_fig, image_ax = plt.subplots()
        else:
            image_ax.clear()

        image_ax.imshow(image)
        image_ax.axis('off')
        image_ax.set_position([0, 0, 1, 1])  # [left, bottom, width, height] — full figure
        image_ax.set_title(rf'PSD n/$\gamma$ discrimmination (offset={offset_values[current_offset_idx]}, stop={stop_values[current_stop_idx]})')
        image_fig.canvas.draw_idle()

# --- BUTTON CALLBACKS ---
def next_offset(event):
    global current_offset_idx
    current_offset_idx = (current_offset_idx + 1) % len(offset_values)
    update_plot()

def prev_offset(event):
    global current_offset_idx
    current_offset_idx = (current_offset_idx - 1) % len(offset_values)
    update_plot()

def next_stop(event):
    global current_stop_idx
    current_stop_idx = (current_stop_idx + 1) % len(stop_values)
    update_plot()

def prev_stop(event):
    global current_stop_idx
    current_stop_idx = (current_stop_idx - 1) % len(stop_values)
    update_plot()

# --- WIDGETS ---
# plt.axes([[left, bottom, width, height]])
axprev_off = plt.axes([0.1, 0.0, 0.1, 0.05])
axnext_off = plt.axes([0.2, 0.0, 0.1, 0.05])

axprev_stop = plt.axes([0.0, 0.05, 0.1, 0.05])
axnext_stop = plt.axes([0.0, 0.1, 0.1, 0.05])

bprev_off = Button(axprev_off, 'Prev Offset')
bnext_off = Button(axnext_off, 'Next Offset')

bprev_stop = Button(axprev_stop, 'Prev Stop')
bnext_stop = Button(axnext_stop, 'Next Stop')

bprev_off.on_clicked(prev_offset)
bnext_off.on_clicked(next_offset)

bprev_stop.on_clicked(prev_stop)
bnext_stop.on_clicked(next_stop)

# --- Separate image figure ---
if image is not None:
    image_fig, image_ax = plt.subplots(figsize=(10, 8))
    image_ax.imshow(image)
    image_ax.axis('off')
    image_ax.set_position([0, 0, 1, 1])  # [left, bottom, width, height] — full figure
    image_ax.set_title(rf'PSD n/$\gamma$ discrimmination (offset={offset_values[current_offset_idx]}, stop={stop_values[current_stop_idx]})')
else:
    image_fig, image_ax = None, None

plt.show()
