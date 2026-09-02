"""Hover a peak in the spectrum, see the transverse mode that produced it.

    python pico_scope/mode_video_sync_show.py --session <capture folder>
    python pico_scope/mode_video_sync_show.py --session <folder> --scope <file>.psdata
    python pico_scope/mode_video_sync_show.py --self-test

This is what the whole synchronized capture exists for. When the coupling is
good the spectrum reads on its own - one 0th order, one 1st, fit them and stop.
When it is bad the peaks are ambiguous, and the only way to tell a 0th order
from a 1st from something higher is to look at the transverse pattern. Because
the video and the spectrum were recorded together, that look is no longer
guesswork: every instant of the trace maps to a definite frame.

## Using it

- **Move the mouse** across the spectrum: the image follows, showing the frame
  whose exposure covers that instant.
- **Click** to pin a frame, click again to unpin and resume following.
- **Left / right arrows** step one frame; **shift** steps ten.
- **b** toggles snap-to-brightest (on by default, see below).
- **fit Gaussian** (the checkbox, or **f**) fits a 2D Gaussian to the frame on
  screen and draws its 1/e^2 contour, reporting the beam radii in millimetres.

## Fitting the mode

The checkbox uses the same `gaussian_fit` routine as kalishlot's camera boxes,
so a mode measured here and one measured live in the browser cannot disagree.
It runs in that module's `FitLoop`: a fit costs about 140 ms and hovering
changes the frame far faster, so the loop keeps only the newest frame and skips
the ones the cursor swept past rather than falling behind it.

The millimetre readout is the one thing that cannot be assumed. It uses the
capture's own `effective_pixel_size_mm` - the sensor pitch times the binning.
The Basler acA2040 and the XIMEA MQ042 happen to share a 5.5 um pitch, so it is
the *binning* that matters: a 2x2 binned frame is 11 um per pixel, and quoting
the bare pitch would halve every width reported. A capture that records no
pixel size at all gets the widths in pixels and says so, rather than inventing
a scale.

## Snap to brightest

The offset between the two records is good to roughly a frame when it comes
from the calibrated host clock alone, and to a hundredth of one after
`mode_video_sync.py --refine`. A whole frame of error is enough to show the
dark neighbour of a resonance rather than the resonance itself, so by default
the viewer snaps to the brightest frame within +-1 of the one the offset names.
That absorbs the residual error without inventing anything: a resonance
genuinely brighter than both its neighbours is the frame you meant. Press **b**
to see the unsnapped mapping.

Frame boundaries are drawn as light shading so the 10 ms granularity is visible
- a peak narrower than one band was integrated whole into a single image.
"""

import argparse
import sys
from pathlib import Path

# --- what happens when this file is run (edit these, then press Run) -------
# Nothing here needs the command line; the arguments exist for scripting.
ACTION = 'show'      # 'show' | 'self-test'
SESSION = ''         # capture folder; '' means the most recent one
SCOPE_FILE = ''      # the .psdata of a Phase 1 capture; '' for Phase 2
SNAP_TO_BRIGHTEST = True

import matplotlib
matplotlib.use('Agg' if (ACTION == 'self-test' or '--self-test' in sys.argv)
               else 'Qt5Agg')

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Ellipse  # noqa: E402
from matplotlib.widgets import CheckButtons  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pico_scope.mode_video_sync import (fit_session, frame_at_time,  # noqa: E402
                                        frame_brightness, frame_start_times,
                                        frame_windows, latest_session,
                                        load_session, load_session_trace,
                                        nearest_frame, release_frames)

# The same fitter kalishlot's camera boxes use, straight from the device
# layer - one routine, so a mode measured here and one measured in the
# browser cannot disagree. It lives under basler_cam/ for historical reasons
# but is camera-agnostic; the XIMEA adapter uses it too.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'basler_cam'))
from gaussian_fit import FitLoop  # noqa: E402

SNAP_RADIUS = 1          # frames either side, when snapping to the brightest
SHADE_ALPHA = 0.06      # faint: at 120 frames these are stripes until you zoom
HELP_TEXT = ('move: follow the cursor   click: pin/unpin   '
             'left/right: step (shift = 10)   b: snap-to-brightest   '
             'f: fit a Gaussian')

# Matches kalishlot's CameraFitMixin.FIT_REBINNING, so the browser and this
# viewer fit the same frame the same way. 4x costs ~140 ms on a 448x1024
# frame and agrees with 2x to better than 1% on the widths.
FIT_REBINNING = 4
FIT_POLL_MS = 80        # how often the GUI thread looks for a finished fit
# fit_gaussian bounds amplitude and offset at 4095, the 12-bit full scale it
# was written for. A binned frame can exceed that - 2x2 summing 10-bit XIMEA
# pixels reaches 4092, but 4x4 would reach 16368 - so a frame that scales
# past it is divided down before fitting and the amplitude scaled back.
FIT_MAX_LEVEL = 4095


class ModeSpectrumViewer:
    """Spectrum on top, mode image below, tied together by the fitted offset.

    The layout follows utilities/media_tools/plot_video.py (image axes plus a
    full-width trace axes with an axvline marking what is displayed), and the
    hover is the blit template from
    utilities/media_tools/postprocessing_camera_video.py - restoring a cached
    background and drawing one artist per motion event, rather than redrawing a
    megapixel image while the mouse moves.
    """

    def __init__(self, trace, frames, windows, brightness, title='',
                 snap=True, pixel_size_mm=None, camera_label=''):
        self.trace = trace
        self.frames = frames
        self.windows = np.asarray(windows)
        self.brightness = np.asarray(brightness)
        self.snap = snap
        self.pinned = None
        self.index = 0
        self.background = None

        # Millimetres per pixel *of a stored frame*, which is the sensor pitch
        # times the binning - both cameras here are 5.5 um, but a binned frame
        # is 11 um, so reading the pitch alone would report half the real size.
        # It comes from the capture, never from a constant in this file.
        self.pixel_size_mm = pixel_size_mm
        self.camera_label = camera_label
        self.fitting = False
        self._fit_loop = None
        self._fit_result = None     # written by the fit thread, read by the timer
        self._fit_seen = None
        self._fit_scale = 1.0
        self._fit_timer = None

        self._build_figure(title)
        self._connect()
        self.show_frame(0)

    # ------------------------------------------------------------- layout
    def _build_figure(self, title):
        self.fig = plt.figure(figsize=(15, 9))
        self.ax_trace = self.fig.add_axes([0.07, 0.70, 0.88, 0.23])
        self.ax_bright = self.fig.add_axes([0.07, 0.55, 0.88, 0.12],
                                           sharex=self.ax_trace)
        self.ax_image = self.fig.add_axes([0.30, 0.06, 0.42, 0.44])

        lo, hi = self.windows[0, 0], self.windows[-1, 1]
        inside = (self.trace.t >= lo) & (self.trace.t <= hi)
        self.ax_trace.plot(self.trace.t[inside] * 1e3,
                           self.trace.signal[inside] * 1e3, lw=0.8)
        self.ax_trace.set_ylabel('Channel D [mV]')
        self.ax_trace.set_title(title, fontsize=10)
        self.ax_trace.tick_params(labelbottom=False)

        # Frame boundaries as light shading: the eye can then see that a peak
        # narrower than one band went into a single image whole.
        for start, end in self.windows[::2]:
            self.ax_trace.axvspan(start * 1e3, end * 1e3, color='k',
                                  alpha=SHADE_ALPHA, lw=0)

        centres = self.windows.mean(axis=1) * 1e3
        width = np.median(np.diff(centres)) * 0.85 if centres.size > 1 else 1.0
        self.ax_bright.bar(centres, self.brightness, width=width,
                           color='tab:orange')
        self.ax_bright.set_ylabel('frame\nbrightness', fontsize=9)
        self.ax_bright.set_xlabel('time in the scope record [ms]')

        # The window of the frame on screen, highlighted. This is what makes the
        # 10 ms granularity concrete: everything inside the band was integrated
        # into the single image below.
        self.window_patch = self.ax_trace.axvspan(
            self.windows[0, 0] * 1e3, self.windows[0, 1] * 1e3,
            color='crimson', alpha=0.20, lw=0)
        self.cursor_trace = self.ax_trace.axvline(centres[0], color='crimson',
                                                  lw=1.2)
        self.cursor_bright = self.ax_bright.axvline(centres[0], color='crimson',
                                                    lw=1.2)

        first = np.asarray(self.frames[0])
        self.image = self.ax_image.imshow(first, cmap='inferno',
                                          vmin=0, vmax=max(int(first.max()), 1))
        self.ax_image.set_xticks([])
        self.ax_image.set_yticks([])
        self.image_title = self.ax_image.set_title('', fontsize=10)
        self.fig.text(0.5, 0.005, HELP_TEXT, ha='center', va='bottom',
                      fontsize=9)

        # One shared scale, so brightness differences between frames are real
        # rather than an artefact of autoscaling each image to itself.
        self.image.set_clim(0, max(int(np.asarray(self.frames).max()), 1))

        self._build_fit_controls()

    def _build_fit_controls(self):
        """The fit checkbox, and the overlay it draws on the image."""
        self.ax_fit = self.fig.add_axes([0.755, 0.42, 0.115, 0.07])
        self.ax_fit.set_frame_on(False)
        self.check_fit = CheckButtons(self.ax_fit, ['fit Gaussian'], [False])
        self.check_fit.on_clicked(lambda _label: self.toggle_fit())

        # 1/e^2 contour of the fitted beam, plus its centre.
        self.fit_ellipse = Ellipse((0, 0), 0, 0, angle=0.0, fill=False,
                                   color='#39d0ff', lw=1.4, zorder=5)
        self.fit_ellipse.set_visible(False)
        self.ax_image.add_patch(self.fit_ellipse)
        self.fit_centre, = self.ax_image.plot([], [], '+', color='#39d0ff',
                                              ms=11, mew=1.4, zorder=6)
        self.fit_text = self.fig.text(0.755, 0.10, '', fontsize=8.5,
                                      va='bottom', ha='left', family='monospace')

    def _connect(self):
        self.cids = [
            self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion),
            self.fig.canvas.mpl_connect('button_press_event', self.on_click),
            self.fig.canvas.mpl_connect('key_press_event', self.on_key),
            self.fig.canvas.mpl_connect('draw_event', self.on_draw),
        ]

    # ---------------------------------------------------------------- fit
    def toggle_fit(self, enabled=None):
        """Turn the Gaussian fit on or off; returns the new state.

        The fit runs in kalishlot's FitLoop rather than inline: one fit costs
        about 140 ms, and hovering changes the frame far faster than that. The
        loop keeps only the newest submitted frame, so a fast sweep of the
        mouse skips the frames it passed over instead of queueing 100 fits and
        falling seconds behind the cursor.
        """
        self.fitting = (not self.fitting) if enabled is None else bool(enabled)
        if self.check_fit.get_status()[0] != self.fitting:
            # keeps the box in step when toggled by the 'f' key
            self.check_fit.eventson = False
            self.check_fit.set_active(0)
            self.check_fit.eventson = True
        if self.fitting:
            if self._fit_loop is None:
                self._fit_loop = FitLoop(on_result=self._on_fit_result,
                                         rebinning=FIT_REBINNING)
            self._fit_loop.start()
            self._start_fit_timer()
            self.submit_fit()
        else:
            if self._fit_loop is not None:
                self._fit_loop.stop()
            self._fit_result = self._fit_seen = None
            self.fit_ellipse.set_visible(False)
            self.fit_centre.set_data([], [])
            self.fit_text.set_text('')
            self.fig.canvas.draw_idle()
        return self.fitting

    def submit_fit(self):
        """Hand the frame on screen to the fit thread."""
        if not self.fitting or self._fit_loop is None:
            return
        frame = np.asarray(self.frames[self.index], dtype=float)
        # fit_gaussian bounds amplitude and offset at 4095; scale a deeper
        # frame down rather than letting the bound silently clip the fit.
        peak = float(frame.max())
        self._fit_scale = FIT_MAX_LEVEL / peak if peak > FIT_MAX_LEVEL else 1.0
        self._fit_loop.submit(frame * self._fit_scale if self._fit_scale != 1.0
                              else frame)

    def _on_fit_result(self, success, parameters):
        """Called on the fit thread: only store, never touch an artist."""
        self._fit_result = (success, parameters, self.index, self._fit_scale)

    def _start_fit_timer(self):
        """Poll for finished fits on the GUI thread.

        Matplotlib artists must not be touched from the fit thread, so the
        result is picked up here instead of drawn where it is produced.
        """
        if self._fit_timer is not None:
            return
        self._fit_timer = self.fig.canvas.new_timer(interval=FIT_POLL_MS)
        self._fit_timer.add_callback(self._poll_fit)
        self._fit_timer.start()

    def _poll_fit(self):
        result = self._fit_result
        if result is None or result is self._fit_seen:
            return
        self._fit_seen = result
        self.draw_fit(*result)

    def draw_fit(self, success, parameters, index, scale=1.0):
        """Put a finished fit on the image. Safe to call directly in tests."""
        if not success:
            self.fit_ellipse.set_visible(False)
            self.fit_centre.set_data([], [])
            reason = parameters.get('reason', 'fit did not converge')
            self.fit_text.set_text(f'frame {index}: {reason}')
            self.fig.canvas.draw_idle()
            return

        w_x, w_y = parameters['w_x'], parameters['w_y']
        self.fit_ellipse.set_center((parameters['x_0'], parameters['y_0']))
        # w is the 1/e^2 radius, so the ellipse's full width is twice it.
        self.fit_ellipse.set_width(2 * w_x)
        self.fit_ellipse.set_height(2 * w_y)
        self.fit_ellipse.set_angle(np.degrees(parameters['angle']))
        self.fit_ellipse.set_visible(True)
        self.fit_centre.set_data([parameters['x_0']], [parameters['y_0']])

        lines = [f'frame {index}',
                 f'x0 {parameters["x_0"]:7.1f} px',
                 f'y0 {parameters["y_0"]:7.1f} px']
        if self.pixel_size_mm:
            lines += [f'wx {w_x * self.pixel_size_mm:7.3f} mm',
                      f'wy {w_y * self.pixel_size_mm:7.3f} mm']
        else:
            lines += [f'wx {w_x:7.1f} px', f'wy {w_y:7.1f} px',
                      'pixel size unknown']
        lines.append(f'amp {parameters["amplitude"] / scale:7.0f}')
        if self.camera_label:
            lines.append(self.camera_label)
        self.fit_text.set_text('\n'.join(lines))
        self.fig.canvas.draw_idle()

    def close_fit(self):
        """Stop the fit thread and its timer; called when the viewer closes."""
        if self._fit_timer is not None:
            self._fit_timer.stop()
            self._fit_timer = None
        if self._fit_loop is not None:
            self._fit_loop.stop()
            self._fit_loop = None
        self.fitting = False

    # -------------------------------------------------------------- logic
    def frame_for_time(self, t_seconds):
        """Which frame to show for an instant in the scope's timebase."""
        index = frame_at_time(self.windows, t_seconds)
        if index is None:                       # dead time, or past the burst
            index = nearest_frame(self.windows, t_seconds)
        if self.snap:
            low = max(0, index - SNAP_RADIUS)
            high = min(len(self.brightness), index + SNAP_RADIUS + 1)
            index = low + int(np.argmax(self.brightness[low:high]))
        return index

    def show_frame(self, index, redraw=True):
        index = int(np.clip(index, 0, len(self.frames) - 1))
        self.index = index
        self.image.set_data(np.asarray(self.frames[index]))
        centre = self.windows[index].mean() * 1e3
        low, high = self.windows[index] * 1e3
        # axvspan gives a Rectangle here, so move it with the rectangle API
        self.window_patch.set_x(low)
        self.window_patch.set_width(high - low)
        self.cursor_trace.set_xdata([centre, centre])
        self.cursor_bright.set_xdata([centre, centre])
        state = 'pinned' if self.pinned is not None else 'following'
        self.image_title.set_text(
            f'frame {index} of {len(self.frames) - 1}   '
            f'{self.windows[index, 0] * 1e3:.2f}-{self.windows[index, 1] * 1e3:.2f} ms   '
            f'brightness {self.brightness[index]:.2f}   [{state}'
            f'{", snap" if self.snap else ""}]')
        self.submit_fit()
        if redraw:
            self._blit()

    def _blit(self):
        """Repaint the cursors over a cached background where possible."""
        if self.background is None:
            self.fig.canvas.draw_idle()
            return
        try:
            self.fig.canvas.restore_region(self.background)
            self.ax_trace.draw_artist(self.window_patch)
            self.ax_trace.draw_artist(self.cursor_trace)
            self.ax_bright.draw_artist(self.cursor_bright)
            self.fig.canvas.blit(self.ax_trace.bbox)
            self.fig.canvas.blit(self.ax_bright.bbox)
        except Exception:
            self.background = None
        # the image itself is not blitted: it changes rarely enough that a
        # normal idle redraw keeps up, and it lives outside the cursor axes
        self.fig.canvas.draw_idle()

    # ------------------------------------------------------------- events
    def on_draw(self, _event):
        try:
            self.background = self.fig.canvas.copy_from_bbox(
                self.fig.bbox)
        except Exception:
            self.background = None

    def on_motion(self, event):
        if self.pinned is not None or event.inaxes not in (self.ax_trace,
                                                           self.ax_bright):
            return
        if event.xdata is None:
            return
        self.show_frame(self.frame_for_time(event.xdata / 1e3))

    def on_click(self, event):
        if event.inaxes not in (self.ax_trace, self.ax_bright):
            return
        if self.pinned is None and event.xdata is not None:
            self.pinned = self.frame_for_time(event.xdata / 1e3)
            self.show_frame(self.pinned)
        else:
            self.pinned = None
            self.show_frame(self.index)

    def on_key(self, event):
        step = 10 if (event.key or '').startswith('shift+') else 1
        key = (event.key or '').replace('shift+', '')
        if key == 'left':
            self.pinned = self.index - step
        elif key == 'right':
            self.pinned = self.index + step
        elif key == 'b':
            self.snap = not self.snap
        elif key == 'f':
            self.toggle_fit()
            return
        else:
            return
        if self.pinned is not None:
            self.pinned = int(np.clip(self.pinned, 0, len(self.frames) - 1))
        self.show_frame(self.pinned if self.pinned is not None else self.index)


# ------------------------------------------------------------------ loading
def viewer_from_session(session_path, scope_path=None, snap=True):
    """Build a viewer from a capture, aligning it however it can.

    A Phase 2 capture carries its own scope trace and an offset, so nothing
    else is needed; a Phase 1 capture needs the .psdata that was recorded
    alongside it, and is aligned by fitting.
    """
    session_path = Path(session_path)
    if scope_path is not None:
        result = fit_session(session_path, scope_path, verbose=True)
        trace, frames = result['trace'], result['frames']
        windows = result['windows']
        session = result['session']
        source = (f'fitted against {Path(scope_path).name}, '
                  f'depth {result["best"].depth:.1f}x')
    else:
        session, frames = load_session(session_path)
        trace = load_session_trace(session_path)
        sync = session['sync']
        key = 't0_fitted_s' if 't0_fitted_s' in sync else 't0_host_s'
        starts = frame_start_times(session['meta'])
        windows = frame_windows(starts, float(session['exposure_s']),
                                float(sync[key]))
        source = ('refined offset' if key == 't0_fitted_s'
                  else 'calibrated host clock (run --refine to sharpen it)')

    brightness = np.array(session.get('brightness_masked')
                          or frame_brightness(frames), dtype=float)
    title = f'{Path(session_path).name} - {source}'
    return ModeSpectrumViewer(trace, frames, windows, brightness, title, snap,
                              pixel_size_mm=session_pixel_size_mm(session),
                              camera_label=camera_label(session))


def session_pixel_size_mm(session):
    """Millimetres per pixel of a stored frame, or None if unrecorded.

    A capture writes `effective_pixel_size_mm`, which is already the sensor
    pitch times the binning. Older captures predate that key, so it is rebuilt
    from the camera block - which carries the pitch per make, 5.5 um for both
    the Basler acA2040 and the XIMEA MQ042 - times the binning that was used.
    Falling back to a constant is what this must never do: a wrong pitch turns
    every millimetre in the readout into a plausible, silent lie.
    """
    recorded = session.get('effective_pixel_size_mm')
    if recorded:
        return float(recorded)
    camera = session.get('camera') or {}
    pitch = camera.get('pixel_size_mm')
    if not pitch:
        return None
    binning = camera.get('binning_x') or session.get('binning') or 1
    return float(pitch) * float(binning)


def camera_label(session):
    """One line naming the camera and the pixel size the fit is quoting."""
    camera = session.get('camera') or {}
    make = camera.get('make') or 'camera'
    size = session_pixel_size_mm(session)
    if size is None:
        return f'{make}, pixel size unknown'
    return f'{make} {size * 1e3:.1f} um/px'


# --------------------------------------------------------------- self-test
def _self_test():
    print('mode_video_sync_show self-test')
    from pico_scope.mode_video_sync import _cumulative, _synthetic_trace, \
        predicted_brightness

    rng = np.random.default_rng(5)
    trace = _synthetic_trace(duration=1.4)
    period, exposure, n = 0.0100406, 0.0099, 100
    starts = np.arange(n) * period
    t0 = 0.15
    brightness = predicted_brightness(np.array([t0]), starts, exposure,
                                      trace.t, _cumulative(trace.t, trace.signal))[0]
    height, width = 24, 32
    yy, xx = np.mgrid[0:height, 0:width]
    blob = np.exp(-(((yy - 12) / 3.0) ** 2 + ((xx - 16) / 3.0) ** 2))
    frames = np.clip(brightness[:, None, None] * (200 / brightness.max()) * blob
                     + rng.normal(0, 1.0, (n, height, width)), 0, 255
                     ).astype(np.uint8)
    windows = frame_windows(starts, exposure, t0)

    viewer = ModeSpectrumViewer(trace, frames, windows, brightness,
                                'self-test', snap=False)
    assert viewer.index == 0

    # the frame for an instant is the frame whose window contains it
    for probe in (3, 27, 61, 99):
        centre = windows[probe].mean()
        assert viewer.frame_for_time(centre) == probe, probe
    # window edges: start inclusive, end exclusive
    assert viewer.frame_for_time(windows[10, 0]) == 10
    assert viewer.frame_for_time(windows[10, 1] - 1e-9) == 10
    # outside the burst clamps rather than failing
    assert viewer.frame_for_time(windows[-1, 1] + 5.0) == n - 1
    assert viewer.frame_for_time(windows[0, 0] - 5.0) == 0
    print('  frame_for_time is exact inside windows and clamps outside')

    # snapping moves to a brighter neighbour, and only by one frame
    viewer.snap = True
    peak = int(np.argmax(brightness))
    for offset in (-1, 0, 1):
        probe = np.clip(peak + offset, 0, n - 1)
        snapped = viewer.frame_for_time(windows[probe].mean())
        assert abs(snapped - probe) <= SNAP_RADIUS
        assert brightness[snapped] >= brightness[probe] - 1e-9
    assert viewer.frame_for_time(windows[peak].mean()) == peak
    viewer.snap = False
    print('  snap-to-brightest moves at most one frame, and never to a dimmer one')

    # the hover handler drives the display through synthetic events
    class _Event:
        def __init__(self, inaxes, xdata):
            self.inaxes, self.xdata, self.key, self.button = inaxes, xdata, None, 1

    target = 42
    viewer.on_motion(_Event(viewer.ax_trace, windows[target].mean() * 1e3))
    assert viewer.index == target, viewer.index
    # a motion event outside the trace axes changes nothing
    viewer.on_motion(_Event(viewer.ax_image, 0.0))
    assert viewer.index == target
    print('  motion over the spectrum moves the image, and elsewhere does not')

    # click pins, a second click releases
    viewer.on_click(_Event(viewer.ax_trace, windows[7].mean() * 1e3))
    assert viewer.pinned == 7 and viewer.index == 7
    viewer.on_motion(_Event(viewer.ax_trace, windows[80].mean() * 1e3))
    assert viewer.index == 7, 'a pinned viewer ignores the cursor'
    viewer.on_click(_Event(viewer.ax_trace, windows[7].mean() * 1e3))
    assert viewer.pinned is None
    viewer.on_motion(_Event(viewer.ax_trace, windows[80].mean() * 1e3))
    assert viewer.index == 80, 'unpinning resumes following'
    print('  click pins the frame and a second click resumes following')

    # arrow keys step, shift steps ten, and the ends clamp
    class _Key:
        def __init__(self, key):
            self.key, self.inaxes, self.xdata = key, None, None

    viewer.on_key(_Key('left'))
    assert viewer.index == 79, viewer.index
    viewer.on_key(_Key('shift+right'))
    assert viewer.index == 89, viewer.index
    for _ in range(30):
        viewer.on_key(_Key('shift+right'))
    assert viewer.index == n - 1, 'stepping past the end clamps'
    for _ in range(30):
        viewer.on_key(_Key('shift+left'))
    assert viewer.index == 0, 'stepping past the start clamps'
    snap_before = viewer.snap
    viewer.on_key(_Key('b'))
    assert viewer.snap is not snap_before, 'b toggles snapping'
    print('  arrow keys step and clamp; b toggles snapping')

    # the highlighted band must track the frame being shown
    viewer.show_frame(33, redraw=False)
    low = viewer.window_patch.get_x()
    width = viewer.window_patch.get_width()
    assert np.isclose(low, windows[33, 0] * 1e3), (low, windows[33, 0] * 1e3)
    assert np.isclose(low + width, windows[33, 1] * 1e3)
    assert np.isclose(width, exposure * 1e3), 'the band is one exposure wide'
    print('  the highlighted band tracks the frame and is one exposure wide')

    # --- the Gaussian fit ---------------------------------------------
    # The pixel size is the part that fails silently, so it is pinned for
    # both makes. Both sensors are 5.5 um, so it is the binning that decides
    # the answer, and a capture that records nothing must say so rather than
    # inventing a scale.
    assert session_pixel_size_mm({'effective_pixel_size_mm': 0.011}) == 0.011
    for make in ('basler', 'ximea'):
        camera = {'make': make, 'pixel_size_mm': 0.0055, 'binning_x': 2}
        rebuilt = session_pixel_size_mm({'camera': camera})
        assert np.isclose(rebuilt, 0.011), (make, rebuilt)
        assert np.isclose(session_pixel_size_mm(
            {'camera': dict(camera, binning_x=1)}), 0.0055), make
        assert f'{make} 11.0 um/px' == camera_label({'camera': camera}), make
    assert session_pixel_size_mm({}) is None
    assert 'unknown' in camera_label({})
    print('  the fit reads its pixel size from the capture - sensor pitch '
          'times binning - for either make, and says so when it has none')

    # a fit of a known blob: the 1/e^2 radii come back in millimetres
    sigma_px = 3.0
    fit_viewer = ModeSpectrumViewer(trace, frames, windows, brightness,
                                    'fit', snap=False, pixel_size_mm=0.011,
                                    camera_label='ximea 11.0 um/px')
    from gaussian_fit import fit_gaussian
    ok, pars = fit_gaussian(np.asarray(frames[int(np.argmax(brightness))],
                                       dtype=float), rebinning=1)
    assert ok, 'the synthetic blob must fit'
    # the blob is exp(-(r/3)^2), i.e. sigma = 3/sqrt(2), and w = 2 sigma
    expected_w = 2 * sigma_px / np.sqrt(2)
    assert np.isclose(pars['w_x'], expected_w, rtol=0.1), (pars['w_x'], expected_w)
    assert np.isclose(pars['x_0'], 16, atol=0.5), pars['x_0']
    assert np.isclose(pars['y_0'], 12, atol=0.5), pars['y_0']

    fit_viewer.draw_fit(True, pars, 7)
    assert fit_viewer.fit_ellipse.get_visible()
    centre = fit_viewer.fit_ellipse.get_center()
    assert np.isclose(centre[0], pars['x_0']) and np.isclose(centre[1], pars['y_0'])
    # w is a radius, so the drawn contour is twice it across
    assert np.isclose(fit_viewer.fit_ellipse.get_width(), 2 * pars['w_x'])
    assert np.isclose(fit_viewer.fit_ellipse.get_height(), 2 * pars['w_y'])
    readout = fit_viewer.fit_text.get_text()
    assert f'{pars["w_x"] * 0.011:7.3f} mm' in readout, readout
    assert 'ximea 11.0 um/px' in readout
    print('  the fitted contour is drawn at 1/e^2 and the widths are '
          'reported in millimetres, not pixels')

    # a failed fit says why instead of leaving a stale ellipse on screen
    fit_viewer.draw_fit(False, {'reason': 'low signal'}, 8)
    assert not fit_viewer.fit_ellipse.get_visible()
    assert 'low signal' in fit_viewer.fit_text.get_text()
    # without a pixel size the widths stay in pixels rather than being scaled
    fit_viewer.pixel_size_mm = None
    fit_viewer.draw_fit(True, pars, 9)
    assert 'mm' not in fit_viewer.fit_text.get_text()
    assert 'pixel size unknown' in fit_viewer.fit_text.get_text()
    print('  a fit that does not converge clears the overlay and says why')

    # toggling off stops the thread and clears the overlay
    fit_viewer.pixel_size_mm = 0.011
    assert fit_viewer.toggle_fit(True) is True
    assert fit_viewer.check_fit.get_status()[0] is True
    assert fit_viewer.toggle_fit(False) is False
    assert not fit_viewer.fit_ellipse.get_visible()
    assert fit_viewer.fit_text.get_text() == ''
    assert fit_viewer.check_fit.get_status()[0] is False
    fit_viewer.close_fit()
    plt.close(fit_viewer.fig)
    print('  the checkbox turns the fit on and off and stops its thread')

    plt.close(viewer.fig)
    # the run-button configuration has to name something this file can do
    assert ACTION in ('show', 'self-test'), ACTION
    print('self-test passed')


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--self-test', action='store_true',
                        help='run the offline checks and exit')
    parser.add_argument('--session', default=SESSION or None,
                        help='capture folder or *_session.json; defaults to '
                             'SESSION in this file, or the newest capture')
    parser.add_argument('--scope', default=SCOPE_FILE or None,
                        help='the .psdata recorded alongside a Phase 1 capture; '
                             'omit for a Phase 2 capture, which carries its own')
    parser.add_argument('--no-snap', action='store_true',
                        help='show the frame the offset names, without snapping '
                             'to the brightest neighbour')
    args = parser.parse_args()

    if args.self_test or ACTION == 'self-test':
        _self_test()
        return
    if ACTION != 'show':
        raise SystemExit(f'ACTION must be show or self-test, not {ACTION!r}')

    session = args.session or latest_session()
    print(f'session: {session}')
    snap = SNAP_TO_BRIGHTEST and not args.no_snap
    viewer = viewer_from_session(session, args.scope, snap=snap)
    plt.show()
    viewer.close_fit()
    release_frames(viewer.frames)


if __name__ == '__main__':
    main()
