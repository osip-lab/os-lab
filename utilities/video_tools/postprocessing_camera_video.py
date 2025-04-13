import time

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.widgets import SpanSelector
from scipy.optimize import curve_fit
from scipy.ndimage import center_of_mass, median_filter
from basler_cam.mode_position_capture_gui import rebin, fit_gaussian, gaussian2d

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



def gaussian2d(xy, ampl, xo, yo, sigma_x, sigma_y, theta, offset):
    x = xy[0]
    y = xy[1]
    a = np.cos(theta) ** 2 / (2 * sigma_x ** 2) + np.sin(theta) ** 2 / (2 * sigma_y ** 2)
    b = -np.sin(2 * theta) / (4 * sigma_x ** 2) + np.sin(2 * theta) / (4 * sigma_y ** 2)
    c = np.sin(theta) ** 2 / (2 * sigma_x ** 2) + np.cos(theta) ** 2 / (2 * sigma_y ** 2)
    g = offset + ampl * np.exp(-(a * (x - xo) ** 2 + 2 * b * (x - xo) * (y - yo) + c * (y - yo) ** 2))
    return np.ravel(g)


def fit_gaussian(arr, rebinning=1):
    sy0, sx0 = np.shape(arr)
    sy, sx = np.shape(arr)

    xx = np.linspace(0, sx - 1, sx)
    yy = np.linspace(0, sy - 1, sy)
    xx, yy = np.meshgrid(xx, yy)

    background = np.percentile(arr, 15)
    mh = arr > np.percentile(arr - background, (1 - 100 / sx0 / sy0) * 100)
    amplitude = np.mean(arr[mh]) - background
    y0, x0 = center_of_mass(np.array(mh, dtype=np.float64))
    mc = arr > amplitude / np.e ** 0.5
    radius = max((np.sum(mc) / np.pi) ** 0.5, 1)
    initial_guess = (amplitude, x0, y0, radius, radius, 0.0, background)
    tic = time.time()
    try:
        p = curve_fit(gaussian2d, np.array((xx, yy)), arr.ravel(), p0=initial_guess, full_output=True,
                      bounds=(
                          (0.0, 0.0, 0.0, 0.0, 0.0, -np.pi / 4, 0.0), (4095, sx, sy, np.inf, np.inf, np.pi / 4, 4095)),
                      ftol=1e-3, xtol=1e-3)
        # p = curve_fit(gaussian2d, np.array((xx, yy)), arr.ravel(), p0=initial_guess, full_output=True)
    except RuntimeError:
        p = (initial_guess, np.zeros_like(initial_guess), 'fitting unsuccessful')
    dt = time.time() - tic

    pars = p[0]
    pars = (pars[0], pars[1] * rebinning, pars[2] * rebinning, pars[3] * rebinning, pars[4] * rebinning,
            pars[5], pars[6])

    xx = np.linspace(0, sx0 - 1, sx0)
    yy = np.linspace(0, sy0 - 1, sy0)
    xx, yy = np.meshgrid(xx, yy)

    gauss = np.reshape(gaussian2d(np.array((xx, yy)), *pars), (sy0, sx0))
    # gauss = zoom(gauss, rebinning, order=0)
    pars = {'amplitude': pars[0], 'offset': pars[6], 'angle': pars[5], 'time': dt,
            'x_0': pars[1], 'y_0': pars[2], 's_x': pars[3], 's_y': pars[4],
            'w_x': pars[3] * 2 ** 0.5, 'w_y': pars[4] * 2 ** 0.5}

    return gauss, pars

# %%
video_path = r"C:\Users\OsipLab\Weizmann Institute Dropbox\Michael Kali\Lab's Dropbox\Laser Phase Plate\Experiments\Results\20250413\after first alignment - 50000.avi"  # Change to your video file

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
summed_frame = np.zeros((video_array.shape[1], video_array.shape[2]))  # Initialize summed_frame to None


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
            global summed_frame
            summed_frame = np.sum(trimmed_video[clicked_axes], axis=0)
            plt.figure()
            plt.title("Summed Frame of Selected Axes")
            plt.imshow(summed_frame, cmap='gray')
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

# %%
# Store clicked points
clicked_points = []
fig, ax = plt.subplots()
img = ax.imshow(summed_frame, cmap='gray')
circle_patch = None  # Will hold the circle once created

PIXEL_SIZE_MM = 0.0055  # 5.5 microns in mm

def calc_circle(x1, y1, x2, y2, x3, y3):
    temp = x2**2 + y2**2
    bc = (x1**2 + y1**2 - temp) / 2
    cd = (temp - x3**2 - y3**2) / 2
    det = (x1 - x2)*(y2 - y3) - (x2 - x3)*(y1 - y2)
    if abs(det) < 1e-10:
        raise ValueError("Points are colinear")
    cx = (bc*(y2 - y3) - cd*(y1 - y2)) / det
    cy = ((x1 - x2)*cd - (x2 - x3)*bc) / det
    r = np.sqrt((cx - x1)**2 + (cy - y1)**2)
    return cx, cy, r

def on_click(event):
    global clicked_points, circle_patch

    # Ignore clicks outside the image
    if event.inaxes != ax:
        return

    # Ignore clicks when zoom/pan tool is active
    if plt.get_current_fig_manager().toolbar.mode != '':
        return

    # Start a new round after 3 points
    if len(clicked_points) >= 3:
        clicked_points = []
        ax.cla()
        ax.imshow(summed_frame, cmap='gray')
        circle_patch = None

    # Record click
    x, y = event.xdata, event.ydata
    clicked_points.append((x, y))
    ax.plot(x, y, 'ro')  # mark the point

    if len(clicked_points) == 3:
        x1, y1 = clicked_points[0]
        x2, y2 = clicked_points[1]
        x3, y3 = clicked_points[2]

        try:
            cx, cy, radius = calc_circle(x1, y1, x2, y2, x3, y3)
            radius_mm = radius * PIXEL_SIZE_MM
            print(f"Radius: {radius:.2f} pixels, {radius_mm:.3f} mm")

            # Remove previous circle if it exists
            if circle_patch:
                circle_patch.remove()

            # Draw new circle
            circle_patch = Circle((cx, cy), radius, color='cyan', fill=False, linewidth=2)
            ax.add_patch(circle_patch)

            ax.set_title(f"Radius: {radius:.2f} px | {radius_mm:.3f} mm")
            plt.draw()

        except ValueError as e:
            print("Error:", e)

fig.canvas.mpl_connect('button_press_event', on_click)
plt.title("Click 3 points (zoom tool ignored)")
plt.show()

# %% Plot resulted fit on top of the image with ellipses:
fig, ax = plt.subplots()
ax.imshow(summed_frame, cmap='gray')
gauss, pars = fit_gaussian(summed_frame, rebinning=1)
ax.contour(gauss, levels=5, colors='r')
plt.title(f"w_x = {pars['w_x' ] * 5.5e-6:.2e}, w_x = {pars['w_y' ] * 5.5e-6:.2e}")
plt.show()


