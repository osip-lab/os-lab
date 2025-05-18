import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Circle
import matplotlib
from utilities.media_tools.utils import wait_for_path_from_clipboard
from basler_cam.mode_position_capture_gui import fit_gaussian, rebin_image

matplotlib.use('Qt5Agg')  # Or 'TkAgg' if Qt5Agg doesn't work
from matplotlib import gridspec
from matplotlib.widgets import Slider, TextBox

# ---- Set video/image path ----
path = wait_for_path_from_clipboard(filetype='media')  # <--- changed from 'video' to 'media'

if not os.path.exists(path):
    raise FileNotFoundError(f"File not found: {path}")

# ---- Try to load as video ----
frames = []
is_video = False

cap = cv2.VideoCapture(path)
if cap.isOpened():
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray)

    cap.release()

    if len(frames) > 0:
        frames = np.array(frames)
        is_video = True
    else:
        print("⚠ Detected empty video file. Trying to load as image instead.")
else:
    print("⚠ Failed to open as video. Trying to load as image instead.")

# ---- If not a video, try to load as an image ----
if not is_video:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise IOError(f"Cannot open file as video or image: {path}")
    frames = np.array([img])  # Add an extra dimension: shape (1, H, W)

PIXEL_SIZE_MM = 0.0055
rebin_factor = 64
selected_frames = []
selected_frames_indices = []

# Precompute intensity sum per frame
frame_sums = [np.sum(f) for f in frames]

# ---- Initialize plot ----
vmax_default = np.max(frames)

# Create the main figure window
fig = plt.figure(figsize=(10, 8))

# Define a grid layout with 3 rows and 3 columns
# Row heights: [top projection, main image, bottom plot]
# Column widths: [left projection, main image, colorbar]
gs = gridspec.GridSpec(
    3, 3,
    height_ratios=[1, 6, 1],
    width_ratios=[1, 6, 0.25]
)


# Create axes for each panel
ax_top = fig.add_subplot(gs[0, 1])     # Top projection (column mean)
ax_left = fig.add_subplot(gs[1, 0])    # Left projection (row mean)
ax = fig.add_subplot(gs[1, 1])         # Main image display
ax_cb = fig.add_subplot(gs[1, 2])      # Colorbar
ax_plot = fig.add_subplot(gs[2, 1])    # Intensity vs. frame plot

# Adjust layout to make room for sliders and help text below
plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.25)

# Display the first frame in grayscale
img_disp = ax.imshow(frames[0], cmap='gray', vmin=0, vmax=vmax_default)

# Add a colorbar for intensity scaling
cb = plt.colorbar(img_disp, cax=ax_cb)

# === Line objects for projections ===

# These lines show the row/column means of the raw image
col_mean_line, = ax_top.plot([], [], lw=1)       # Top projection
row_mean_line, = ax_left.plot([], [], lw=1)      # Left projection

# These lines show the Gaussian fit projections (dashed red)
gauss_col_line, = ax_top.plot([], [], 'r--', lw=1)
gauss_row_line, = ax_left.plot([], [], 'r--', lw=1)

# Lines for fitted Gaussian mean positions (vertical/horizontal markers)
gauss_xline = ax_top.axvline(0, color='r', linestyle='--', visible=False)
gauss_yline = ax_left.axhline(0, color='r', linestyle='--', visible=False)

# === Styling for side projection plots ===

# Remove ticks to declutter side plots
ax_top.set_xticks([])
ax_top.set_yticks([])
ax_left.set_xticks([])
ax_left.set_yticks([])

# Invert the y-axis on the left projection so it aligns with image
ax_left.invert_yaxis()

# === Bottom intensity plot (total image intensity per frame) ===

intensity_line, = ax_plot.plot(frame_sums, lw=1)  # Main intensity trace
selected_lines = []                               # To store markers for selected frames

ax_plot.set_xlim(0, len(frames) - 1)
ax_plot.set_ylabel("Total Intensity")
ax_plot.set_xlabel("Frame")

# === UI Controls ===

# Slider to select the current frame
ax_frame = plt.axes([0.25, 0.2, 0.65, 0.03])
frame_slider = Slider(ax_frame, 'Frame', 0, len(frames) - 1, valinit=0, valstep=1)

# Slider to adjust the display intensity max (vmax)
ax_vmax = plt.axes([0.25, 0.15, 0.65, 0.03])
vmax_slider = Slider(ax_vmax, 'vmax', 1, 255, valinit=vmax_default)

# Text box to set the rebinning factor
ax_rebin = plt.axes([0.25, 0.1, 0.2, 0.03])
rebin_textbox = TextBox(ax_rebin, "Rebin", initial=str(rebin_factor))

# === Help instructions at the bottom ===

# Add a fixed help text at the bottom center of the window
help_text = (
    "Press: space to add frame to frames for fitting, w to fit, "
    "arrows to navigate between frames, and the number box is for rebinningFrame"
)
fig.text(0.5, 0.02, help_text, ha='center', va='bottom', fontsize=9)


def on_rebin_text_submit(text):
    global rebin_factor
    try:
        val = int(text)
        if val >= 1:
            rebin_factor = val
            print(f"Rebinning factor set to: {rebin_factor}")
            update(None)
        else:
            print("Rebin factor must be >= 1")
    except ValueError:
        print("Invalid integer input for rebin factor")

rebin_textbox.on_submit(on_rebin_text_submit)

# State
clicked_points = []
circle_patch = None
gaussian_contour = None
current_frame_index = 0
fit_data = None


def calc_circle(x1, y1, x2, y2, x3, y3):
    temp = x2 ** 2 + y2 ** 2
    bc = (x1 ** 2 + y1 ** 2 - temp) / 2
    cd = (temp - x3 ** 2 - y3 ** 2) / 2
    det = (x1 - x2) * (y2 - y3) - (x2 - x3) * (y1 - y2)
    if abs(det) < 1e-10:
        raise ValueError("Points are colinear")
    cx = (bc * (y2 - y3) - cd * (y1 - y2)) / det
    cy = ((x1 - x2) * cd - (x2 - x3) * bc) / det
    r = np.sqrt((cx - x1) ** 2 + (cy - y1) ** 2)
    return cx, cy, r


def update_title(frame_idx, radius=None, status=""):
    global fit_data
    time = frame_idx / fps
    base_title = (f" {frame_idx} (Time: {time:.2f}s)")
    if radius is not None:
        radius_mm = radius * PIXEL_SIZE_MM
        base_title += f" | Radius: {radius:.2f}px = {radius_mm:.3f}mm"
    if frame_idx in selected_frames_indices:
        base_title += " [SELECTED]"
    if status:
        base_title += f" | {status}"
    if fit_data is not None:
        gauss, par = fit_data
        base_title += f" | Fit: w_x={par['w_x'] * 5.5e-3:.2f}mm, w_y={par['w_y'] * 5.5e-3:.2f}mm"
    ax.set_title(base_title)


def update_selected_lines():
    global selected_lines
    for line in selected_lines:
        line.remove()
    selected_lines = [ax_plot.axvline(x=idx, color='red', linestyle='--', alpha=0.5) for idx in selected_frames_indices]
    fig.canvas.draw_idle()


def update(val):
    global current_frame_index, circle_patch, clicked_points, gaussian_contour, fit_data
    current_frame_index = int(frame_slider.val)
    vmax_val = vmax_slider.val
    frame = frames[current_frame_index]
    frame = rebin_image(frame, rebin_factor) if rebin_factor > 1 else frame
    img_disp.set_data(frame)
    img_disp.set_clim(vmin=0, vmax=vmax_val)
    print("kaki 1")

    # Remove old overlays
    if circle_patch:
        circle_patch.remove()
        circle_patch = None
    if gaussian_contour:
        for coll in gaussian_contour.collections:
            coll.remove()
        gaussian_contour = None

    clicked_points = []

    # Update projections
    row_mean = np.mean(frame, axis=1)
    col_mean = np.mean(frame, axis=0)
    col_mean_line.set_data(np.arange(len(col_mean)), col_mean)
    row_mean_line.set_data(row_mean, np.arange(len(row_mean)))
    ax_top.set_xlim(0, len(col_mean))
    ax_top.set_ylim(0, np.max(col_mean) * 1.1)
    ax_left.set_xlim(0, np.max(row_mean) * 1.1)
    ax_left.set_ylim(0, len(row_mean))

    # Hide Gaussian projection lines
    # Hide Gaussian lines initially
    # gauss_xline.set_visible(False)
    # gauss_yline.set_visible(False)
    # gauss_col_line.set_data([], [])
    # gauss_row_line.set_data([], [])
    # If Gaussian fit is available, plot the contour and projections
    if fit_data is not None:
        gauss, par = fit_data
        gaussian_contour = ax.contour(gauss, levels=5, colors='r')

        # Gaussian mean position
        # x_mean, y_mean = par['x_0'], par['y_0']
        # gauss_xline.set_visible(True)
        # gauss_yline.set_visible(True)
        # gauss_xline.set_xdata([x_mean])
        # gauss_yline.set_ydata([y_mean])

        # Gaussian projections
        gauss_col_mean = np.mean(gauss, axis=0)
        gauss_row_mean = np.mean(gauss, axis=1)
        print(f"{col_mean.shape=}")
        print(f"{col_mean=}")
        print(f"{gauss_col_mean.shape=}")
        print(f"{gauss_col_mean=}")

        gauss_col_line.set_visible(True)
        gauss_row_line.set_visible(True)
        gauss_col_line.set_data(np.arange(len(gauss_col_mean)) / rebin_factor, gauss_col_mean)
        gauss_row_line.set_data(gauss_row_mean, np.arange(len(gauss_row_mean)) / rebin_factor)

        # Update axis limits to ensure lines are shown
        ax_top.relim()
        ax_top.autoscale_view()
        ax_left.relim()
        ax_left.autoscale_view()

    update_title(current_frame_index)
    fig.canvas.draw_idle()



frame_slider.on_changed(update)
vmax_slider.on_changed(update)


def on_key(event):
    global rebin_factor, fit_data, gaussian_contour

    current = int(frame_slider.val)

    if event.key == 'right' and current < len(frames) - 1:
        frame_slider.set_val(current + 1)

    elif event.key == 'left' and current > 0:
        frame_slider.set_val(current - 1)

    elif event.key == ' ':
        if current_frame_index not in selected_frames_indices:
            selected_frames_indices.append(current_frame_index)
            print(selected_frames_indices)
            selected_frames.append(frames[current_frame_index])
            update_selected_lines()
        update_title(current_frame_index)
        fig.canvas.draw_idle()

    # elif event.key.isdigit() and 1 <= int(event.key) <= 9:
    #     rebin_factor = int(event.key)
    #     print(f"Rebinning factor set to: {rebin_factor}")
    #     update(None)

    elif event.key == 'w':
        if not selected_frames_indices:
            print("No frames selected.")
            return

        update_title(current_frame_index, status="Calculating fit...")
        fig.canvas.draw_idle()
        plt.pause(0.01)  # allow GUI to refresh

        try:
            selected_array = frames[selected_frames_indices]
            averaged_frame = np.mean(selected_array, axis=0)

            # rebinned = rebin_image(averaged_frame, rebin_factor) if rebin_factor > 1 else averaged_frame
            # print ("rebinned_shape :", rebinned.shape)
            # gauss, par = fit_gaussian(rebinned, rebinning=1)  # already rebinned
            # print("gauss shape:", gauss.shape)
            gauss, par = fit_gaussian(averaged_frame, rebinning=rebin_factor)  # already rebinned
            print("Fit parameters:", par)
            fit_data = (gauss, par)
            print("updating title")
            update_title(current_frame_index, status="Fit finished")
            update(None)

        except Exception as e:
            print("Fit failed:", e)
            fit_data = None
            update_title(current_frame_index, status="Fit failed")
            update(None)


def on_click(event):
    global clicked_points, circle_patch

    if event.inaxes != ax or plt.get_current_fig_manager().toolbar.mode != '':
        return

    if len(clicked_points) >= 3:
        clicked_points = []
        ax.images[0].set_data(frames[current_frame_index])
        if circle_patch:
            circle_patch.remove()
            circle_patch = None

    x, y = event.xdata, event.ydata
    clicked_points.append((x, y))
    ax.plot(x, y, 'ro')

    if len(clicked_points) == 3:
        try:
            (x1, y1), (x2, y2), (x3, y3) = clicked_points
            cx, cy, radius = calc_circle(x1, y1, x2, y2, x3, y3)
            radius_mm = radius * PIXEL_SIZE_MM
            print(f"Radius: {radius:.2f} pixels, {radius_mm:.3f} mm")
            circle_patch = Circle((cx, cy), radius, color='cyan', linestyle='--', fill=False, linewidth=2, alpha=0.5)
            ax.add_patch(circle_patch)
            update_title(current_frame_index, radius)
            plt.draw()
        except ValueError as e:
            print("Error:", e)


fig.canvas.mpl_connect('key_press_event', on_key)
fig.canvas.mpl_connect('button_press_event', on_click)

update_title(0)
plt.show()
# %%