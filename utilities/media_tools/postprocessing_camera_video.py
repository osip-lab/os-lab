import time

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.widgets import SpanSelector
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from basler_cam.mode_position_capture_gui import fit_gaussian
from utilities.utils import append_numerical_result_line, wait_for_path_from_clipboard

matplotlib.use('Qt5Agg')  # Or 'TkAgg' if Qt5Agg doesn't work
PIXEL_SIZE_BASLER_CAMERA = 5.5e-6  # 5.5 microns
# If None, it is ignored. If a float, the extracted spot sizes are converted to
# an NA estimate via NA_x = NA_TO_SPOT_SIZE_RATIO * w_x (and likewise for y),
# with w in METERS - so the ratio has units of 1/m. The NAs are then shown in
# the fit plot title and recorded in numerical-results.txt.
NA_TO_SPOT_SIZE_RATIO = None
# If True, the user clicks the Gaussian center and a point ~1 sigma away on the
# selected frame to seed the fit. If False, the initial guess is estimated
# automatically (as before).
MANUAL_INITIAL_GUESS = True


def load_video_as_numpy(video_path):
    """Loads the video from `video_path` into a numpy array of shape (T, N, M)."""
    cap = cv2.VideoCapture(video_path)
    frames = []

    fps = cap.get(cv2.CAP_PROP_FPS)  # Frames per second
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Total frames
    print(f"FPS: {fps}, Total frames: {total_frames}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert to grayscale if needed
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        frames.append(gray_frame)

    cap.release()
    frames = np.array(frames)
    print(f"Video loaded with shape {frames.shape} (T, N, M)")
    return frames, fps


def plot_intensity_vs_time(intensity, fps):
    """Plots the sum of pixel intensities per frame over time."""
    times = np.arange(len(intensity)) / fps  # Convert frame indices to time
    plt.plot(times, intensity)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Intensity (sum of pixel values)")
    plt.title("Pixel Intensity Over Time")
    plt.show()
    return times


def get_time_range_from_user(times, intensity):
    """Allows the user to select a time range using SpanSelector."""
    fig, ax = plt.subplots()
    ax.plot(times, intensity)
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Intensity (sum of pixel values)")
    ax.set_title("Select Time Range")

    selected_range = [None, None]

    def onselect(xmin, xmax):
        selected_range[0], selected_range[1] = xmin, xmax
        print(f"Selected time range: {xmin:.2f} - {xmax:.2f} seconds")
        plt.close(fig)  # Close the plot after selection

    span = SpanSelector(ax, onselect, "horizontal", useblit=True, interactive=True,
                        props={'alpha': 0.3, 'color': 'red'})

    plt.show(block=True)
    return tuple(selected_range)


def get_manual_initial_guess(frame):
    """Presents `frame` and lets the user click the Gaussian center followed by a
    point ~1 sigma away. Returns a dict with 'x_0', 'y_0' and 'sigma' (pixels)."""
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(frame, cmap='gray')
    ax.set_title("Click the Gaussian center, then a point ~1 sigma away")
    fig.tight_layout()
    pts = plt.ginput(2, timeout=0)
    plt.close(fig)
    if len(pts) < 2:
        raise RuntimeError("Manual initial guess requires two clicks (center and 1-sigma point).")
    (cx, cy), (px, py) = pts
    sigma = float(np.hypot(px - cx, py - cy))
    print(f"Manual initial guess: center=({cx:.1f}, {cy:.1f}), sigma={sigma:.1f} px")
    return {'x_0': float(cx), 'y_0': float(cy), 'sigma': max(sigma, 1.0)}


def trim_video_by_time_range(video_array, time_range, fps):
    """Trims the video to the selected time range."""
    start_time, end_time = time_range
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    trimmed_video = video_array[start_frame:end_frame]
    print(f"Trimmed video shape: {trimmed_video.shape}")
    return trimmed_video

# %%
video_path = wait_for_path_from_clipboard(filetype='video')

video_array, fps = load_video_as_numpy(video_path)

intensity_t = video_array.sum(axis=(1, 2))  # Sum pixel intensities per frame

times = np.arange(len(intensity_t)) / fps  # Get time values for each frame

selected_time_range = get_time_range_from_user(times, intensity_t)

trimmed_video = trim_video_by_time_range(video_array, selected_time_range, fps)
timestamps = times[int(selected_time_range[0] * fps):int(selected_time_range[1] * fps)]
# %%
nrows = 2
ncols = (trimmed_video.shape[0] // nrows) + (trimmed_video.shape[0] % nrows)
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(7, nrows * 5))
fig.suptitle("Click a frame to select it and continue", fontsize=11)
selected_frame = None
selected_frame_time = None  # time [s] of the chosen frame within the video


def on_click(event):
    global selected_frame, selected_frame_time
    for i, ax in enumerate(axes.flat):
        if ax == event.inaxes and i < trimmed_video.shape[0]:
            selected_frame = trimmed_video[i].astype(float)
            selected_frame_time = timestamps[i]
            print(f"Selected frame {i} (time {selected_frame_time:.2f}s)")
            plt.close(fig)
            return


fig.canvas.mpl_connect('button_press_event', on_click)

for i, ax in enumerate(axes.flat):
    if i < trimmed_video.shape[0]:
        ax.set_title(f"Frame {i} (Time: {timestamps[i]:.2f}s)")
        ax.imshow(trimmed_video[i], cmap='gray')
        ax.axis('off')
    else:
        ax.axis('off')  # Hide any unused subplots

fig.tight_layout()
# plt.get_current_fig_manager().window.showMaximized()
plt.show()  # blocks until a frame is clicked (the click closes the window)

# %% Plot resulted fit on top of the image with ellipses:
if selected_frame is None:
    raise RuntimeError("No frame was selected — click one of the frames in the grid.")

manual_guess = get_manual_initial_guess(selected_frame) if MANUAL_INITIAL_GUESS else None
gauss, pars = fit_gaussian(selected_frame, rebinning=2, manual_guess=manual_guess)

sy, sx = selected_frame.shape
x0, y0 = int(pars['x_0']), int(pars['y_0'])

fig, ax = plt.subplots(figsize=(10, 10))
div = make_axes_locatable(ax)
hax = div.append_axes('top', size='20%', pad=0.2)
hax.sharex(ax)
hax.tick_params(bottom=False, top=True, labelbottom=False, labeltop=True)
vax = div.append_axes('right', size='20%', pad=0.2)
vax.sharey(ax)
vax.tick_params(left=False, right=True, labelleft=False, labelright=True)

ax.imshow(selected_frame, cmap='gray', origin='upper')
ax.contour(gauss, levels=5, colors='r')

hax.plot(np.arange(sx), selected_frame[y0, :])
hax.plot(np.arange(sx), gauss[y0, :])
vax.plot(selected_frame[:, x0], np.arange(sy))
vax.plot(gauss[:, x0], np.arange(sy))

w_x_m = pars['w_x'] * PIXEL_SIZE_BASLER_CAMERA
w_y_m = pars['w_y'] * PIXEL_SIZE_BASLER_CAMERA
w_x_mm = w_x_m * 1e3
w_y_mm = w_y_m * 1e3

title = f"w_x = {w_x_mm:.3f} mm,  w_y = {w_y_mm:.3f} mm"
results_text = (f"frame_time = {selected_frame_time:.2f} s, "
                f"(w_x, w_y) = ({w_x_mm:.4f} mm, {w_y_mm:.4f} mm)")
if NA_TO_SPOT_SIZE_RATIO is not None:
    NA_x = NA_TO_SPOT_SIZE_RATIO * w_x_m
    NA_y = NA_TO_SPOT_SIZE_RATIO * w_y_m
    title += f"\nNA_x = {NA_x:.4f},  NA_y = {NA_y:.4f}"
    results_text += (f", NA_x = {NA_x:.4f}, NA_y = {NA_y:.4f}, "
                     f"NA_to_spot_size_ratio = {NA_TO_SPOT_SIZE_RATIO:.6g} 1/m")
fig.suptitle(title, fontsize=14)

fig.tight_layout()
fig.subplots_adjust(top=0.90 if NA_TO_SPOT_SIZE_RATIO is not None else 0.93)

append_numerical_result_line(video_path, results_text)

plt.show()
