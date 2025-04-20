import time

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.widgets import SpanSelector
from scipy.optimize import curve_fit
from scipy.ndimage import center_of_mass, median_filter
from basler_cam.mode_position_capture_gui import rebin, fit_gaussian, gaussian2d
from utilities.video_tools.utils import wait_for_path_from_clipboard

matplotlib.use('Qt5Agg')  # Or 'TkAgg' if Qt5Agg doesn't work
PIXEL_SIZE_BASLER_CAMERA = 5.5e-6  # 5.5 microns
from matplotlib.patches import Circle


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

nrows = 2
ncols = (trimmed_video.shape[0] // nrows) + (trimmed_video.shape[0] % nrows)
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(7, nrows * 5))
clicked_axes = []
averaged_frame = np.zeros((video_array.shape[1], video_array.shape[2]))  # Initialize averaged_frame to None


def on_click(event):
    for i, ax in enumerate(axes.flat):
        if ax == event.inaxes:
            if i < trimmed_video.shape[0]:
                ax.set_title(f"Frame {i} (Time: {timestamps[i]:.2f}s) Clicked!")
                clicked_axes.append(i)
                plt.draw()


fig.canvas.mpl_connect('button_press_event', on_click)


def on_key(event):
    if event.key == 'enter':
        if clicked_axes:
            global averaged_frame
            averaged_frame = np.mean(trimmed_video[clicked_axes], axis=0)
            plt.figure()
            plt.title("Summed Frame of Selected Axes")
            plt.imshow(averaged_frame, cmap='gray')
            plt.axis('off')
            plt.show()

fig.canvas.mpl_connect('key_press_event', on_key)

for i, ax in enumerate(axes.flat):
    if i < trimmed_video.shape[0]:
        ax.set_title(f"Frame {i} (Time: {timestamps[i]:.2f}s)")
        ax.imshow(trimmed_video[i], cmap='gray')
        ax.axis('off')
    else:
        ax.axis('off')  # Hide any unused subplots

fig.tight_layout()
plt.get_current_fig_manager().window.showMaximized()
plt.show()

# %% Plot resulted fit on top of the image with ellipses:
fig, ax = plt.subplots()
ax.imshow(averaged_frame, cmap='gray')
gauss, pars = fit_gaussian(averaged_frame, rebinning=1)
ax.contour(gauss, levels=5, colors='r')
plt.title(f"s_x = {pars['s_x']:.0f}, s_y = {pars['s_y']:.0f}")
plt.show()
