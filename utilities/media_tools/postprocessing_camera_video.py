"""postprocessing_camera_video.py

Extract the cavity mode's spot size from a camera video, and map it to the
numerical aperture (NA) of the cavity's short arm.

Pipeline
--------
1. Load a video (file path taken from the clipboard) and enter the long arm
   length in cm (it changes between measurements, so it is asked for on every
   run rather than read from a config constant).
2. Drag a horizontal window over the intensity-vs-time trace to pick the part
   of the video worth looking at.
3. Click one frame out of the grid of the selected frames.
4. Drag out the mode's diameter to seed the fit (MANUAL_INITIAL_GUESS) - press at
   one side of it, release at the other, and the circle follows the mouse. In
   the same window, CROP_TO_CIRCLE_KEY makes the next drag mark a circle to
   keep: everything outside it is replaced by its CROP_FILL_PERCENTILE-th
   percentile, which takes a second spot or a bright edge out of the fit.
   SKIP_MANUAL_GUESS_KEY leaves the guess to the fit. Then fit a 2D Gaussian to
   that frame (utilities.utils.fit_gaussian_beam): spot sizes w_minor, w_major
   in pixels -> metres via PIXEL_SIZE_BASLER_CAMERA. They are named after the
   beam's own principal axes, not the image's - a tilted mode has no width
   "along x", and which of the two is the wider one is what the measurement
   is about.
5. Map each spot size to the short arm's NA, by whichever route NA_FROM_SPOT_SIZE
   selects:
   - 'simulation': the cavity-design simulation
     (simple_analysis_scripts.camera_spot_size_per_cavity_NA). The optical
     system it simulates - the two element lists, the arm lengths and the
     camera distance - is defined in the configuration block below; nothing has
     to be edited in the cavity-design project. The measured spot sizes go in
     with it, so the dependency plot comes back with them marked, and the
     system is drawn with the mode that was measured.
   - 'ratio': the fixed linear ratio NA = NA_TO_SPOT_SIZE_RATIO * w, which was
     calibrated once against a measured spectrum. No simulation is run.
   - None: no NA at all, only the spot sizes.
6. With SHOW_SHORT_ARM_LENGTH, run the cavity's lens-position scan
   (pico_scope/mode_analysis.py, the one the mode-spacing scripts use) and read
   off where the lens has to be for the NA just measured. It answers with TWO
   positions per NA: the mode spacing falls monotonically as the lens moves out,
   but the NA falls to a minimum near collimation and climbs again past it, so an
   NA on its own names a lens position on each branch. A mode-spacing measurement
   - or knowing which side of collimation the lens came from - tells them apart.
7. Print the spot sizes, NAs and lens positions, and append a one-line record to
   numerical-results.txt in the folder of the video.
"""

import time

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.backend_bases import MouseButton
from matplotlib.patches import Circle, Ellipse
from matplotlib.widgets import SpanSelector
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from utilities.utils import (append_numerical_result_line, ask_long_arm_length,
                             fit_gaussian_beam, gaussian_beam_image,
                             wait_for_path_from_clipboard)
# The spot size -> NA mapping comes from the cavity-design project; this module
# forwards the geometry below to it and caches the scan.
from utilities.media_tools.spot_size_analysis import get_spot_size_to_na, na_from_spot_size
# ... and the lens-position scan that says where the lens has to be for the NA
# that comes out. The same one the PicoScope mode-spacing scripts run.
from pico_scope.mode_analysis import get_na_interpolators

matplotlib.use('Qt5Agg')  # Or 'TkAgg' if Qt5Agg doesn't work
PIXEL_SIZE_BASLER_CAMERA = 5.5e-6  # 5.5 microns
# If True, the user drags the mode's diameter on the selected frame to seed the
# fit. If False, the initial guess is estimated automatically (as before).
# Either way the decision can be taken per frame: pressing SKIP_MANUAL_GUESS_KEY
# in the marking window falls back to the automatic guess for that frame, and
# CROP_TO_CIRCLE_KEY masks everything outside a dragged circle before the fit.
MANUAL_INITIAL_GUESS = True
# Keep away from matplotlib's own window shortcuts when changing these - s saves,
# q quits, p pans, o zooms, g grids, f goes fullscreen, k/l switch to log axes,
# c/v/left/right step the view history, h/r reset it.
SKIP_MANUAL_GUESS_KEY = 'escape'
CROP_TO_CIRCLE_KEY = 'm'

# --- how the spot sizes become an NA ---------------------------------------
# 'simulation' - propagate the mode through the real optical system (below);
# 'ratio'      - the fixed linear ratio just underneath, no simulation;
# None         - report the spot sizes only.
NA_FROM_SPOT_SIZE = 'simulation'
# Used by the 'ratio' route: NA_minor = NA_TO_SPOT_SIZE_RATIO * w_minor with w
# in METRES, so the ratio is in 1/m. This one was extracted from the
# video/spectrum of .\2026-07-13\25MHz\1 35 - it holds for that geometry only,
# which is why the simulation route exists.
NA_TO_SPOT_SIZE_RATIO = 0.0545 * 1000

# --- the system being measured, simulated by the 'simulation' route --------
# (edit this when the setup changes.)
# Element names come from the cavity-design catalog; list them in optical order.
# To see the available names:
#   python -c "from utilities.media_tools.spot_size_analysis import list_cavity_elements; print(*list_cavity_elements(), sep='\n')"
#
# What the outgoing beam meets on its way to the camera, starting with the
# cavity's end mirror in TRANSMISSION (the ..._REFRACTIVE entry - the mode is
# matched to it, and every distance below is measured from it):
OUTGOING_ELEMENTS = [
    'COASTLINE_20CM_REFRACTIVE',
    'NEWPORT_200MM_PLANO_CONVEX',
]
# What the mode meets going the other way, from the long arm into the short arm.
# One lens or two; with two, MID_ARM_LENGTH is the distance between them (with
# one it is ignored). They are turned around automatically - the catalog
# orientations belong to the real cavity, whose layout is mirrored w.r.t. this
# simulation's.
INTRACAVITY_ELEMENTS = [
    # 'THOLABS_200MM_PLANO_CONVEX_LENS',
    'EDMUND_4MM_ASPHERIC_16701',
]
# All lengths in metres, as everywhere in the cavity-design library.
MID_ARM_LENGTH = 1e-2     # between the two intracavity lenses (unused if there is one)
LENS_DISTANCE = 59e-3     # end mirror -> the next outgoing element (the collimating lens)
CAMERA_DISTANCE = 0.02    # last outgoing element -> camera sensor
N_points = 200            # long-arm NAs simulated across the scanned range
# (min, max) long-arm NA the simulation scans - it sets the range of camera spot
# sizes the mapping is defined over. None keeps the simulation's own range (its
# NA floor .. 0.01); widen it if a measured spot size comes back out of range.
NA_LONG_ARM_RANGE = None

# --- where the lens has to be for the NA that comes out --------------------
# A second, separate simulation: the lens-position scan of the whole cavity
# (pico_scope/mode_analysis.py, the one the mode-spacing scripts run), read from
# the NA rather than from a mode spacing. It costs a scan per run, so it can be
# turned off. Note that it answers with TWO lens positions - see the report.
SHOW_SHORT_ARM_LENGTH = True
# The cavity itself, in optical order, end mirror to end mirror - not the path
# to the camera that OUTGOING_ELEMENTS describes. Catalog names again:
#   python -c "from pico_scope.mode_analysis import list_cavity_elements; print(*list_cavity_elements(), sep='\n')"
CAVITY_ELEMENTS = [
    'LASER_OPTIK_MIRROR',
    'EDMUND_4MM_ASPHERIC_16701',
    'COASTLINE_20CM_MIRROR',
]
SHORT_ARM_LENGTHS = (0.5e-4, 2e-4)  # [m] lens-scan span around the collimation point
LENS_SCAN_N_POINTS = 300            # lens positions simulated across it


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


CROP_FILL_PERCENTILE = 10  # what the pixels outside the kept circle are replaced with


def key_label(key):
    """How a hotkey is written in a window title."""
    return {'escape': 'Esc', 'enter': 'Enter', ' ': 'Space'}.get(key, f"'{key}'")


def crop_outside_circle(frame, center, radius, percentile=CROP_FILL_PERCENTILE):
    """Replace every pixel outside the circle with the `percentile`-th percentile of those pixels.

    Everything the circle does not keep - a second spot, a reflection, a bright edge - is flattened
    to the low end of its own background level, so it can neither pull the 2D Gaussian fit nor lift
    its offset. Returns a new frame; `center` is (x, y) and `radius` is in pixels.
    """
    yy, xx = np.ogrid[:frame.shape[0], :frame.shape[1]]
    outside = (xx - center[0]) ** 2 + (yy - center[1]) ** 2 > radius ** 2
    if not outside.any():  # a circle around the whole frame removes nothing
        return frame.copy()
    cropped = frame.copy()
    cropped[outside] = np.percentile(frame[outside], percentile)
    return cropped


def mark_gaussian_on_frame(frame, skip_key=SKIP_MANUAL_GUESS_KEY, crop_key=CROP_TO_CIRCLE_KEY):
    """Present `frame` and let the user drag out circles on it, in one window.

    A drag runs from one side of a circle to the opposite one - press at one end of the diameter,
    release at the other - and the circle is drawn live while the mouse moves. What the drag means
    depends on the mode:
      - by default it marks the mode itself, and seeds the fit with center = the drag's midpoint and
        sigma = half its length;
      - after `crop_key` is pressed the next drag marks the circle to KEEP: everything outside it is
        replaced by crop_outside_circle(), the image is redrawn, and the window returns to marking
        the mode. Any number of crops can be taken.
    `skip_key` (or closing the window) leaves the guess to fit_gaussian, which estimates it itself.

    Returns (frame, guess, crops): the frame to fit - cropped if the user cropped it - the initial
    guess dict ('x_0', 'y_0', 'sigma' in pixels) or None, and the list of ((x, y), radius) circles
    that were kept.
    """
    frame = np.asarray(frame, dtype=float)
    # Constrained layout, not tight_layout(): the title is two lines long and changes with the mode,
    # so the room it needs has to be re-made on every draw - tight_layout() runs once, before the
    # title exists, and the first line ends up above the top of the window.
    fig, ax = plt.subplots(figsize=(8, 8), layout='constrained')
    image = ax.imshow(frame, cmap='gray')
    preview = Circle((0, 0), 0, fill=False, color='tab:red', lw=1.4, ls='--', visible=False)
    ax.add_patch(preview)

    state = {'frame': frame, 'guess': None, 'crops': [], 'start': None, 'mode': 'mark',
             'background': None}

    def show_title():
        if state['mode'] == 'crop':
            title = (f"Drag the circle to KEEP: press one side, release the other\n"
                     f"outside it -> its {CROP_FILL_PERCENTILE}th percentile  |  "
                     f"{key_label(crop_key)}: cancel")
        else:
            title = (f"Drag the mode's diameter: press one side, release the other\n"
                     f"{key_label(crop_key)}: mask outside a circle  |  "
                     f"{key_label(skip_key)}: skip, the fit guesses")
        # Short lines, small font and wrap=True together: the title has to survive a window the
        # user narrowed, and a clipped instruction is one the user never reads.
        ax.set_title(title, fontsize=10, wrap=True)
        fig.canvas.draw_idle()

    def circle_of_drag(x, y):
        """The circle whose diameter runs from the press point to (x, y)."""
        (x0, y0) = state['start']
        return ((0.5 * (x0 + x), 0.5 * (y0 + y)), 0.5 * float(np.hypot(x - x0, y - y0)))

    def on_press(event):
        if event.inaxes is ax and event.button == MouseButton.LEFT:
            state['start'] = (event.xdata, event.ydata)
            preview.set_center(state['start'])
            preview.set_radius(0.0)
            preview.set_visible(False)
            # Snapshot the image once, so each mouse move only has to blit the circle back over it -
            # a full redraw of a multi-megapixel frame per motion event would make the circle lag
            # behind the mouse. Backends that cannot blit fall back to redrawing (see on_motion).
            try:
                fig.canvas.draw()
                state['background'] = fig.canvas.copy_from_bbox(ax.bbox)
            except Exception:
                state['background'] = None
            preview.set_visible(True)

    def on_motion(event):
        # Repainting on every motion event is what makes the circle follow the mouse.
        if state['start'] is None or event.inaxes is not ax:
            return
        center, radius = circle_of_drag(event.xdata, event.ydata)
        preview.set_center(center)
        preview.set_radius(radius)
        if state['background'] is None:
            fig.canvas.draw_idle()
            return
        fig.canvas.restore_region(state['background'])
        ax.draw_artist(preview)
        fig.canvas.blit(ax.bbox)

    def on_release(event):
        if state['start'] is None or event.button != MouseButton.LEFT:
            return
        if event.inaxes is not ax:  # released off the image: abandon the drag
            state['start'] = None
            preview.set_visible(False)
            fig.canvas.draw_idle()
            return
        center, radius = circle_of_drag(event.xdata, event.ydata)
        state['start'] = None
        if radius < 1.0:  # a stray click rather than a drag
            preview.set_visible(False)
            fig.canvas.draw_idle()
            return
        if state['mode'] == 'crop':
            state['frame'] = crop_outside_circle(state['frame'], center, radius)
            state['crops'].append((center, radius))
            image.set_data(state['frame'])  # the clim stays put, so the contrast does not jump
            preview.set_visible(False)
            print(f"Cropped to a circle at ({center[0]:.1f}, {center[1]:.1f}) px, "
                  f"radius {radius:.1f} px; outside -> {CROP_FILL_PERCENTILE}th percentile.")
            state['mode'] = 'mark'
            show_title()
        else:
            state['guess'] = {'x_0': float(center[0]), 'y_0': float(center[1]),
                              'sigma': max(radius, 1.0)}
            fig.canvas.stop_event_loop()

    def on_key(event):
        if event.key == skip_key:
            fig.canvas.stop_event_loop()
        elif event.key == crop_key:
            state['mode'] = 'mark' if state['mode'] == 'crop' else 'crop'
            state['start'] = None
            preview.set_visible(False)
            show_title()

    cids = [fig.canvas.mpl_connect(name, handler) for name, handler in
            (('button_press_event', on_press), ('motion_notify_event', on_motion),
             ('button_release_event', on_release), ('key_press_event', on_key),
             ('close_event', lambda event: fig.canvas.stop_event_loop()))]
    show_title()
    fig.show()  # realise the window before blocking on events
    fig.canvas.start_event_loop(timeout=0)  # runs until stop_event_loop()
    for cid in cids:
        fig.canvas.mpl_disconnect(cid)
    plt.close(fig)

    if state['guess'] is None:
        print("Manual initial guess skipped - fit_gaussian estimates it automatically.")
    else:
        print(f"Manual initial guess: center=({state['guess']['x_0']:.1f}, "
              f"{state['guess']['y_0']:.1f}), sigma={state['guess']['sigma']:.1f} px")
    return state['frame'], state['guess'], state['crops']


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

# Asked here, before the interactive windows take over the console.
long_arm_length = ask_long_arm_length()  # [m], prompted in cm

intensity_t = video_array.sum(axis=(1, 2))  # Sum pixel intensities per frame

times = np.arange(len(intensity_t)) / fps  # Get time values for each frame

selected_time_range = get_time_range_from_user(times, intensity_t)

trimmed_video = trim_video_by_time_range(video_array, selected_time_range, fps)
timestamps = times[int(selected_time_range[0] * fps):int(selected_time_range[1] * fps)]
# %%
n_frames = trimmed_video.shape[0]
frame_h, frame_w = trimmed_video.shape[1:3]
# Pick the column count that makes the thumbnails as large as possible on a wide (maximized)
# window, instead of a fixed number of rows that leaves most of the window empty.
screen_aspect = 16 / 9
frame_aspect = frame_w / frame_h
# Thumbnail width when the grid is fitted into a screen_aspect x 1 window, for each candidate.
ncols = max(range(1, n_frames + 1),
            key=lambda nc: min(screen_aspect / nc,
                               frame_aspect / int(np.ceil(n_frames / nc))))
nrows = int(np.ceil(n_frames / ncols))
fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                         figsize=(13, 13 * (nrows * frame_h) / (ncols * frame_w)),
                         squeeze=False)
fig.suptitle("Click a frame to select it and continue", fontsize=11)
# No per-frame titles and almost no padding, so the thumbnails fill the window.
fig.subplots_adjust(left=0.002, right=0.998, top=0.96, bottom=0.004, wspace=0.01, hspace=0.01)
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
    if i < n_frames:
        ax.imshow(trimmed_video[i], cmap='gray')
    ax.axis('off')  # Also hides any unused subplots
# plt.get_current_fig_manager().window.showMaximized()
plt.show()  # blocks until a frame is clicked (the click closes the window)

# %% Plot resulted fit on top of the image with ellipses:
if selected_frame is None:
    raise RuntimeError("No frame was selected — click one of the frames in the grid.")

crops = []
if MANUAL_INITIAL_GUESS:
    # The frame comes back cropped if the user cropped it, so the fit and the plot below both see
    # the image the user was looking at.
    selected_frame, manual_guess, crops = mark_gaussian_on_frame(selected_frame)
else:
    manual_guess = None
fit_ok, pars = fit_gaussian_beam(selected_frame, rebinning=2, manual_guess=manual_guess)
if not fit_ok:
    print("The 2D Gaussian fit did not converge - the numbers below are the "
          "initial guess, not a fit. Re-drag the mode's diameter, or crop away "
          "whatever else is in the frame.")
# the fitted model on the frame's own grid, for the cross-sections below
gauss = gaussian_beam_image(pars, selected_frame.shape)

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
# A single contour, at the beam radius w = 2 sigma - the stack of contours that
# used to be drawn at other widths hid the spot they were describing. Ellipse
# takes full axes, so each is twice the corresponding radius.
ax.add_patch(Ellipse((pars['x_0'], pars['y_0']),
                     width=2 * pars['w_major'], height=2 * pars['w_minor'],
                     # fit_gaussian_beam reports the major axis' direction in
                     # the convention Ellipse takes, so there is no sign to flip
                     angle=pars['major_axis_deg'],
                     edgecolor='r', facecolor='none', linewidth=1.5))

hax.plot(np.arange(sx), selected_frame[y0, :])
hax.plot(np.arange(sx), gauss[y0, :])
vax.plot(selected_frame[:, x0], np.arange(sy))
vax.plot(gauss[:, x0], np.arange(sy))

w_minor_m = pars['w_minor'] * PIXEL_SIZE_BASLER_CAMERA
w_major_m = pars['w_major'] * PIXEL_SIZE_BASLER_CAMERA
w_minor_mm = w_minor_m * 1e3
w_major_mm = w_major_m * 1e3

fig.suptitle(f"w_minor = {w_minor_mm:.3f} mm,  w_major = {w_major_mm:.3f} mm  "
             f"(major axis {pars['major_axis_deg']:+.1f}°)", fontsize=14)
fig.tight_layout()
fig.subplots_adjust(top=0.93)
# Without a running event loop, the window would stay blank until the final
# plt.show(); a short pause flushes the fit to screen before the (slower)
# simulation runs.
plt.pause(0.1)

# %% Map the spot sizes to the NA in the short arm --------------------------
# One NA per principal axis - both routes are rotationally symmetric, so each
# spot size is mapped on its own. The axes are the beam's own (minor and major),
# not the image's: a tilted mode's widths are not its extents along x and y. An
# NA left None is one the chosen route could not give (see the printed reason);
# the spot sizes are reported either way.
NAs = {'minor': None, 'major': None}
spot_sizes_m = {'minor': w_minor_m, 'major': w_major_m}

if NA_FROM_SPOT_SIZE == 'simulation':
    # The spot size <-> NA relation comes from the cavity-design project (path
    # in local_config.py); spot_size_analysis builds and caches the
    # interpolator, and the measured spot sizes are handed over with it so the
    # dependency plot comes back with them marked on it.
    spot_size_to_NA, na_error = get_spot_size_to_na(
        long_arm_length=long_arm_length,  # asked at the start, not the config default
        mid_arm_length=MID_ARM_LENGTH,
        lens_distance=LENS_DISTANCE,
        camera_distance=CAMERA_DISTANCE,
        outgoing_elements=OUTGOING_ELEMENTS,
        intracavity_elements=INTRACAVITY_ELEMENTS,
        N_points=N_points,
        NA_long_arm_range=NA_LONG_ARM_RANGE,
        measured_spot_sizes_m=(w_minor_m, w_major_m),
        measured_labels=('w_minor', 'w_major'),
        plot=True,  # always: the plot is what carries the measurement markers
    )
    if na_error is not None:
        print(f"NA mapping unavailable: {na_error}")
    else:
        for axis, spot_size_m in spot_sizes_m.items():
            NAs[axis], lookup_error = na_from_spot_size(spot_size_to_NA, spot_size_m)
            if lookup_error is not None:
                print(f"NA_{axis} unavailable: {lookup_error}")

elif NA_FROM_SPOT_SIZE == 'ratio':
    NAs = {axis: NA_TO_SPOT_SIZE_RATIO * spot_size_m
           for axis, spot_size_m in spot_sizes_m.items()}

elif NA_FROM_SPOT_SIZE is not None:
    raise ValueError(f"NA_FROM_SPOT_SIZE is {NA_FROM_SPOT_SIZE!r}; expected "
                     f"'simulation', 'ratio' or None.")

# %% Where the lens has to be for that NA ----------------------------------
# The NA on its own does not say where the lens is. Along the lens scan the mode
# spacing falls monotonically - one spacing, one lens position, which is what the
# PicoScope scripts report - but the NA falls to a minimum near collimation and
# climbs again past it, so it names one position on each side. Both are reported;
# which of them the cavity is on takes a mode-spacing measurement, or simply
# knowing which way the lens was walked in.
short_arms_m = {axis: None for axis in NAs}
na_minimum = None
if SHOW_SHORT_ARM_LENGTH and any(na is not None for na in NAs.values()):
    print("Running the lens-position scan for the small arm length...")
    na_interp, _, lens_scan_error = get_na_interpolators(
        elements=CAVITY_ELEMENTS, long_arm=long_arm_length,
        mid_arm=MID_ARM_LENGTH, short_arm_lengths=SHORT_ARM_LENGTHS,
        N_points=LENS_SCAN_N_POINTS, plot_system=False)
    if na_interp is None:
        print(f"Small arm length unavailable: {lens_scan_error}")
    else:
        na_minimum = na_interp.na_minimum
        for axis, na in NAs.items():
            if na is None:
                continue
            try:
                short_arms_m[axis] = na_interp.short_arms_for_na(na)
            except ValueError as error:
                # an NA the scan never reaches is a fact about the measurement,
                # not a reason to lose the spot sizes that are already in hand
                print(f"Small arm length for NA_{axis} unavailable: {error}")


def short_arm_text(arms_m):
    """The lens positions for one NA, in mm - or why there are none."""
    if not arms_m:
        return 'unavailable'
    return ' or '.join(f"{arm * 1e3:.4f} mm" for arm in arms_m)


# Which route produced the NAs, for the report and the record: the two disagree
# whenever the ratio's calibration geometry is not the one being measured, so a
# logged NA is only readable next to the route it came from.
if NA_FROM_SPOT_SIZE == 'ratio':
    na_route_text = f'ratio {NA_TO_SPOT_SIZE_RATIO:.6g} 1/m'
else:
    na_route_text = NA_FROM_SPOT_SIZE or 'none'

# %% Report and record ------------------------------------------------------
width = 64
print()
print('=' * width)
print("  CAMERA SPOT SIZE ANALYSIS".center(width))
print('=' * width)
print(f"  {'Frame time':<28}{selected_frame_time:>18.2f} s")
print(f"  {'Long arm length':<28}{long_arm_length * 1e2:>18.4g} cm")
print('-' * width)
print(f"  {'Spot size w_minor':<28}{w_minor_mm:>18.4f} mm")
print(f"  {'Spot size w_major':<28}{w_major_mm:>18.4f} mm")
print(f"  {'Major axis direction':<28}{pars['major_axis_deg']:>18.2f} deg")
if any(na is not None for na in NAs.values()):
    print('-' * width)
    for axis, na in NAs.items():
        if na is not None:
            print(f"  {'Short arm NA_' + axis:<28}{na:>18.4f}")
    print(f"  {'  (NA from)':<28}{na_route_text:>18}")
if any(arms for arms in short_arms_m.values()):
    print('-' * width)
    for axis, arms in short_arms_m.items():
        if arms:
            print(f"  {'Small arm length, NA_' + axis:<28}{short_arm_text(arms):>18}")
    if na_minimum is not None and any(len(arms or ()) > 1 for arms in short_arms_m.values()):
        # the reason there are two of them, said once rather than per axis
        print(f"    (two of them: the NA bottoms out at {na_minimum[1]:.4f}, "
              f"at {na_minimum[0] * 1e3:.4f} mm)")
print('=' * width)
print()

na_text = ', '.join(f"NA_{axis} = " + (f"{na:.4f}" if na is not None else "N/A")
                    for axis, na in NAs.items())
results_text = (f"frame_time = {selected_frame_time:.2f} s, "
                f"(w_minor, w_major) = ({w_minor_mm:.4f} mm, {w_major_mm:.4f} mm), "
                f"major_axis = {pars['major_axis_deg']:.2f} deg, "
                f"long_arm_length = {long_arm_length:.4g} m, {na_text}, "
                f"NA_from = {na_route_text}")
if any(arms for arms in short_arms_m.values()):
    results_text += ', ' + ', '.join(
        f"short_arm_length_NA_{axis} = {short_arm_text(arms)}"
        for axis, arms in short_arms_m.items() if arms)
if crops:
    # A crop changes the fit, so the record has to say the frame was not the raw one.
    results_text += ', kept_circles = ' + '; '.join(
        f"({center[0]:.0f}, {center[1]:.0f}) px r={radius:.0f} px" for center, radius in crops)
append_numerical_result_line(video_path, results_text)

na_title = ',  '.join(f"NA_{axis} = {na:.4f}" for axis, na in NAs.items() if na is not None)
# Each NA gives two lens positions, so the title names the axis each pair belongs
# to - four numbers on one line would otherwise say nothing about which is which.
arm_title = ',  '.join(f"small arm (NA_{axis}) = {short_arm_text(arms)}"
                       for axis, arms in short_arms_m.items() if arms)
if na_title:
    title = (f"w_minor = {w_minor_mm:.3f} mm,  w_major = {w_major_mm:.3f} mm  "
             f"(major axis {pars['major_axis_deg']:+.1f}°)\n{na_title}")
    if arm_title:
        title += f"\n{arm_title}"
    fig.suptitle(title, fontsize=14)
    fig.subplots_adjust(top=0.90 if not arm_title else 0.87)
    fig.canvas.draw_idle()

# Keep all windows open (and responsive) after the report has been printed.
plt.show(block=True)
