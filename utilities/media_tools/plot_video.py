"""Inspect a video (or a single image) frame by frame.

The path is taken from the clipboard. Controls:
    left / right : previous / next frame
    space        : add the current frame to the set used for the Gaussian fit
    w            : fit a 2D Gaussian to the average of the selected frames
    3 clicks     : fit a circle through three points on the image
    sliders      : frame index and display vmax; the text box sets the rebin factor

Run from the repository root: python -m utilities.media_tools.plot_video
"""
import os

import cv2
import matplotlib
import numpy as np

matplotlib.use('Qt5Agg')  # Or 'TkAgg' if Qt5Agg doesn't work

import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import Circle
from matplotlib.ticker import FuncFormatter
from matplotlib.widgets import Slider, TextBox
from mpl_toolkits.axes_grid1 import make_axes_locatable

# The GUI-free fitting module, the same one kalishlot's camera adapters use
from basler_cam.gaussian_fit import fit_gaussian, gaussian2d, rebin_image
from utilities.utils import wait_for_path_from_clipboard

PIXEL_SIZE_MM = 0.0055
DEFAULT_REBIN_FACTOR = 64
CONTOUR_LEVELS = 5
# Bottom of the axes area, in figure coordinates. Everything under it belongs to
# the rotated time labels and the controls. It is a starting point only - long
# videos have taller (rotated) time labels, so the margin is measured and grown
# once the figure knows how big they really are.
BOTTOM_MARGIN = 0.35
LABEL_CLEARANCE_PX = 8

HELP_TEXT = ("space: add the current frame to the fit set | w: fit a Gaussian | "
             "left/right: change frame | 3 clicks: fit a circle | box: rebin factor")


def load_frames(path):
    """Return (frames, fps) for a video, or a single-frame stack for an image.

    Frames are decoded once into memory as grayscale, so stepping backwards is
    instant. `fps` falls back to 1 for images and for containers that report no
    frame rate, so the time axis and the title stay usable either way.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    frames = []
    fps = 0.0
    cap = cv2.VideoCapture(path)
    try:
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            if not frames:
                print("⚠ Detected empty video file. Trying to load as image instead.")
        else:
            print("⚠ Failed to open as video. Trying to load as image instead.")
    finally:
        cap.release()

    if not frames:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise IOError(f"Cannot open file as video or image: {path}")
        frames = [img]

    if not fps or np.isnan(fps):
        fps = 1.0
    return np.array(frames), fps


def calc_circle(x1, y1, x2, y2, x3, y3):
    """Center and radius of the circle through three points."""
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


class VideoInspector:
    """The figure and all of the state that its callbacks share."""

    def __init__(self, frames, fps, rebin_factor=DEFAULT_REBIN_FACTOR):
        self.frames = frames
        self.fps = fps
        self.rebin_factor = rebin_factor
        self.frame_sums = [np.sum(f) for f in frames]  # Precomputed intensity per frame

        # --- Interaction state ---
        self.current_frame_index = 0
        self.selected_frames_indices = []
        self.fit_data = None           # (gauss, parameters) from the last successful fit
        self.clicked_points = []       # Image-coordinate clicks for the circle fit
        self.click_markers = []        # The red dots drawn for those clicks
        self.circle_patch = None
        self.gaussian_contour = None
        self.selected_lines = []

        self._build_figure()
        self._connect()
        self.update()
        self.reserve_room_for_time_labels()

    # ------------------------------------------------------------------ layout

    def _build_figure(self):
        self.fig = plt.figure(figsize=(10, 8))

        # Two rows: the image (with its projections) and the intensity trace.
        # Everything below the axes - the rotated time labels, the sliders and
        # the help text - lives in the bottom margin.
        self.gs = gridspec.GridSpec(2, 1, height_ratios=[6, 1],
                                    left=0.13, right=0.90, top=0.95, bottom=BOTTOM_MARGIN, hspace=0.7)
        self.ax = self.fig.add_subplot(self.gs[0])          # Main image
        self.ax_plot = self.fig.add_subplot(self.gs[1])     # Intensity vs. time

        vmax_default = max(int(np.max(self.frames)), 1)
        self.img_disp = self.ax.imshow(self.displayed_frame(), cmap='gray', vmin=0, vmax=vmax_default)

        # The image keeps a 1:1 aspect, so it does not fill its grid cell. Hanging
        # the projections and the colorbar off the image axes itself (rather than
        # off neighbouring grid cells) keeps them glued to the image as drawn, and
        # sharex/sharey puts their pixel columns/rows exactly above/beside it.
        divider = make_axes_locatable(self.ax)
        self.ax_top = divider.append_axes("top", size="18%", pad=0.06, sharex=self.ax)
        self.ax_left = divider.append_axes("left", size="18%", pad=0.06, sharey=self.ax)
        self.ax_cb = divider.append_axes("right", size="4%", pad=0.12)
        plt.colorbar(self.img_disp, cax=self.ax_cb)

        # Raw projections (solid) and the Gaussian-fit projections (dashed red)
        self.col_mean_line, = self.ax_top.plot([], [], lw=1)
        self.row_mean_line, = self.ax_left.plot([], [], lw=1)
        self.gauss_col_line, = self.ax_top.plot([], [], 'r--', lw=1)
        self.gauss_row_line, = self.ax_left.plot([], [], 'r--', lw=1)

        # Declutter: no ticks on the intensity axis of either projection, and the
        # shared pixel axes are labelled once - columns under the image, rows on
        # the left projection. Hiding the *labels* only, since clearing ticks on a
        # shared axis would clear the image's as well.
        self.ax_top.set_yticks([])
        self.ax_left.set_xticks([])
        self.ax_top.tick_params(labelbottom=False)
        self.ax.tick_params(labelleft=False)

        # --- Bottom plot: total intensity per frame ---
        self.ax_plot.plot(self.frame_sums, lw=1)
        self.current_frame_line = self.ax_plot.axvline(0, color='k', lw=1)  # Frame on screen
        self.ax_plot.set_xlim(0, max(len(self.frames) - 1, 1))
        self.ax_plot.set_ylabel("Total Intensity")
        # The data stays in frame units (so do all the axvline positions); only
        # the tick labels are converted to seconds.
        self.ax_plot.xaxis.set_major_formatter(FuncFormatter(lambda idx, pos: f"{idx / self.fps:.2f}"))
        self.ax_plot.tick_params(axis='x', labelrotation=90)  # Rotated, so labels can sit denser
        self.ax_plot.set_xlabel("Time [s]")

        # --- Controls, stacked below the axes area ---
        self.frame_slider = Slider(self.fig.add_axes([0.25, 0.22, 0.65, 0.03]), 'Frame',
                                   0, max(len(self.frames) - 1, 1), valinit=0, valstep=1)
        self.vmax_slider = Slider(self.fig.add_axes([0.25, 0.16, 0.65, 0.03]), 'vmax',
                                  1, 255, valinit=vmax_default)
        self.rebin_textbox = TextBox(self.fig.add_axes([0.25, 0.10, 0.2, 0.03]), "Rebin",
                                     initial=str(self.rebin_factor))
        self.fig.text(0.5, 0.02, HELP_TEXT, ha='center', va='bottom', fontsize=9)

    def reserve_room_for_time_labels(self):
        """Lift the axes area until the rotated time labels clear the sliders.

        How tall those labels are is only known once they are rendered - a long
        video labels its ticks '1234.56' where a short one gets '1.20' - so the
        bottom margin is measured here rather than guessed.
        """
        self.fig.canvas.draw()
        renderer = self.fig.canvas.get_renderer()
        labels = self.ax_plot.get_xticklabels() + [self.ax_plot.xaxis.label]
        labels_bottom = min(label.get_window_extent(renderer).y0 for label in labels)
        controls_top = self.frame_slider.ax.get_window_extent().y1

        deficit_px = controls_top + LABEL_CLEARANCE_PX - labels_bottom
        if deficit_px > 0:
            figure_height = self.fig.get_window_extent().height
            self.gs.update(bottom=min(self.gs.bottom + deficit_px / figure_height, 0.7))

    def _connect(self):
        self.frame_slider.on_changed(self.update)
        self.vmax_slider.on_changed(self.update)
        self.rebin_textbox.on_submit(self.on_rebin_text_submit)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)

    # ----------------------------------------------------------- coordinates

    def displayed_frame(self):
        frame = self.frames[self.current_frame_index]
        return rebin_image(frame, self.rebin_factor) if self.rebin_factor > 1 else frame

    def to_display_coords(self, full_res_coords):
        """Map full-resolution pixel coordinates onto the rebinned grid on screen.

        Rebinned bin k averages full-resolution pixels [k*r, (k+1)*r - 1], whose
        center sits at k*r + (r - 1)/2 - hence the shift before dividing.
        """
        return (np.asarray(full_res_coords) - (self.rebin_factor - 1) / 2) / self.rebin_factor

    # --------------------------------------------------------------- drawing

    def update(self, _=None):
        """Redraw everything that depends on the current frame / vmax / rebin."""
        # Clamped, because a single image gives the slider a wider range than
        # the one frame it has (matplotlib rejects valmin == valmax).
        self.current_frame_index = min(int(self.frame_slider.val), len(self.frames) - 1)
        frame = self.displayed_frame()
        self.img_disp.set_data(frame)
        self.img_disp.set_clim(vmin=0, vmax=self.vmax_slider.val)
        self.current_frame_line.set_xdata([self.current_frame_index])

        self.clear_click_overlay()
        self.draw_projections(frame)
        self.draw_fit_overlay(frame)
        self.autoscale_projections()
        self.update_title()
        self.fig.canvas.draw_idle()

    def draw_projections(self, frame):
        # Only the intensity axis is scaled here: the pixel axis of each
        # projection is shared with the image, which owns those limits.
        row_mean = np.mean(frame, axis=1)
        col_mean = np.mean(frame, axis=0)
        self.col_mean_line.set_data(np.arange(len(col_mean)), col_mean)
        self.row_mean_line.set_data(row_mean, np.arange(len(row_mean)))

    def autoscale_projections(self):
        """Fit the intensity axes around the data and the Gaussian fit together."""
        top_values = np.concatenate([self.col_mean_line.get_ydata(), self.gauss_col_line.get_ydata()])
        left_values = np.concatenate([self.row_mean_line.get_xdata(), self.gauss_row_line.get_xdata()])
        # max(..., 1): an all-black frame would give identical limits
        self.ax_top.set_ylim(0, max(np.max(top_values) * 1.1, 1))
        self.ax_left.set_xlim(0, max(np.max(left_values) * 1.1, 1))

    def fit_model_on_screen(self, shape):
        """Evaluate the fitted Gaussian on the same (rebinned) grid as the image.

        `fit_gaussian` reports its parameters in full-resolution sensor pixels,
        so they are converted to display coordinates before the model is built.
        """
        par = self.fit_data
        height, width = shape
        xx, yy = np.meshgrid(np.arange(width, dtype=np.float64),
                             np.arange(height, dtype=np.float64))
        model = gaussian2d(np.array((xx, yy)), par['amplitude'],
                           self.to_display_coords(par['x_0']), self.to_display_coords(par['y_0']),
                           par['s_x'] / self.rebin_factor, par['s_y'] / self.rebin_factor,
                           par['angle'], par['offset'])
        return model.reshape(height, width)

    def draw_fit_overlay(self, frame):
        """Draw the fitted Gaussian as a contour plus its two projections."""
        if self.gaussian_contour is not None:
            self.gaussian_contour.remove()
            self.gaussian_contour = None

        if self.fit_data is None:
            self.gauss_col_line.set_data([], [])
            self.gauss_row_line.set_data([], [])
            return

        gauss = self.fit_model_on_screen(frame.shape)
        self.gaussian_contour = self.ax.contour(gauss, levels=CONTOUR_LEVELS, colors='r')

        # On the display grid, so these overlay the raw projections directly
        self.gauss_col_line.set_data(np.arange(gauss.shape[1]), np.mean(gauss, axis=0))
        self.gauss_row_line.set_data(np.mean(gauss, axis=1), np.arange(gauss.shape[0]))

    def update_title(self, radius=None, status=""):
        time = self.current_frame_index / self.fps
        title = f" {self.current_frame_index} (Time: {time:.3f}s)"
        if radius is not None:
            title += f" | Radius: {radius:.2f}px = {radius * PIXEL_SIZE_MM:.3f}mm"
        if self.current_frame_index in self.selected_frames_indices:
            title += " [SELECTED]"
        if status:
            title += f" | {status}"
        if self.fit_data is not None:
            title += (f" | Fit: w_x={self.fit_data['w_x'] * PIXEL_SIZE_MM:.2f}mm,"
                      f" w_y={self.fit_data['w_y'] * PIXEL_SIZE_MM:.2f}mm")
        self.ax.set_title(title)

    def update_selected_lines(self):
        for line in self.selected_lines:
            line.remove()
        self.selected_lines = [self.ax_plot.axvline(x=idx, color='red', linestyle='--', alpha=0.5)
                               for idx in self.selected_frames_indices]
        self.fig.canvas.draw_idle()

    def clear_click_overlay(self):
        """Drop the circle-fit clicks and the artists drawn for them."""
        for marker in self.click_markers:
            marker.remove()
        self.click_markers = []
        self.clicked_points = []
        if self.circle_patch is not None:
            self.circle_patch.remove()
            self.circle_patch = None

    # -------------------------------------------------------------- callbacks

    def on_rebin_text_submit(self, text):
        try:
            value = int(text)
        except ValueError:
            print("Invalid integer input for rebin factor")
            return
        if value < 1:
            print("Rebin factor must be >= 1")
            return
        self.rebin_factor = value
        print(f"Rebinning factor set to: {self.rebin_factor}")
        self.update()

    def on_key(self, event):
        current = int(self.frame_slider.val)

        if event.key == 'right' and current < len(self.frames) - 1:
            self.frame_slider.set_val(current + 1)

        elif event.key == 'left' and current > 0:
            self.frame_slider.set_val(current - 1)

        elif event.key == ' ':
            if self.current_frame_index not in self.selected_frames_indices:
                self.selected_frames_indices.append(self.current_frame_index)
                print(f"Selected frames: {self.selected_frames_indices}")
                self.update_selected_lines()
            self.update_title()
            self.fig.canvas.draw_idle()

        elif event.key == 'w':
            self.fit_selected_frames()

    def fit_selected_frames(self):
        if not self.selected_frames_indices:
            print("No frames selected.")
            return

        self.update_title(status="Calculating fit...")
        self.fig.canvas.draw_idle()
        plt.pause(0.01)  # Let the GUI refresh before the fit blocks it

        averaged_frame = np.mean(self.frames[self.selected_frames_indices], axis=0)
        try:
            # `fit_gaussian` rebins internally and reports back in full-resolution pixels
            success, par = fit_gaussian(averaged_frame, rebinning=self.rebin_factor)
            print("Fit parameters:", par)
            self.fit_data = par if success else None
            status = "Fit finished" if success else "Fit did not converge"
        except Exception as e:
            print("Fit failed:", e)
            self.fit_data = None
            status = "Fit failed"

        self.update()
        self.update_title(status=status)
        self.fig.canvas.draw_idle()

    def on_click(self, event):
        toolbar = self.fig.canvas.manager.toolbar
        if event.inaxes is not self.ax or (toolbar is not None and toolbar.mode != ''):
            return

        if len(self.clicked_points) >= 3:  # A circle is already drawn - start over
            self.clear_click_overlay()

        self.clicked_points.append((event.xdata, event.ydata))
        marker, = self.ax.plot(event.xdata, event.ydata, 'ro')
        self.click_markers.append(marker)

        if len(self.clicked_points) == 3:
            (x1, y1), (x2, y2), (x3, y3) = self.clicked_points
            try:
                cx, cy, radius = calc_circle(x1, y1, x2, y2, x3, y3)
            except ValueError as e:
                print("Error:", e)
                return
            self.circle_patch = Circle((cx, cy), radius, color='cyan', linestyle='--',
                                       fill=False, linewidth=2, alpha=0.5)
            self.ax.add_patch(self.circle_patch)
            # The clicks are in display units; report the radius in camera pixels
            radius_px = radius * self.rebin_factor
            print(f"Radius: {radius_px:.2f} pixels, {radius_px * PIXEL_SIZE_MM:.3f} mm")
            self.update_title(radius=radius_px)

        self.fig.canvas.draw_idle()


def main():
    path = wait_for_path_from_clipboard(filetype='media')
    frames, fps = load_frames(path)
    VideoInspector(frames, fps)
    plt.show()


if __name__ == '__main__':
    main()
