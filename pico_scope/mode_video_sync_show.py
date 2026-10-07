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
- **fit Gaussian** (the checkbox, or **f**) fits a 2D Gaussian to the frame on
  screen and draws its 1/e^2 contour, reporting the beam radii in millimetres.
- **renormalize** (the checkbox, or **n**; on when the viewer opens) scales
  each frame to itself, with its RENORMALIZE_PERCENTILE-th percentile as full
  brightness, so a dim mode is as visible as a bright one. Off, every frame
  shares one scale and the brightness differences between them are real.

## Which capture it opens

`--session` wins whenever it is given, which is how `run_mode_video_pipeline.py`
hands over the burst just recorded: the viewer opens that folder even when it is
not the newest one on the local bank, because a capture saved to a folder the
user pasted is not where `latest_session()` would look.

Run on its own the script takes the config's SESSION - `'clipboard'` asks for
the folder when the viewer starts, `''` takes the newest capture, and anything
else is the path itself. That is `mode_video_sync_mark.py`'s convention, so the
two viewers are entered the same way.

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

The frame shown is the one the offset names, with nothing interposed: good to
roughly a frame when the offset comes from the calibrated host clock alone, and
to a hundredth of one after `mode_video_sync.py --refine`.

## The second channel

When the record carries one, the temperature-modulation ramp is drawn over the
spectrum in green, on its own y axis to the right. That axis is not decoration:
the transmission is tens of millivolts and the ramp is volts, so one shared
scale would flatten the transmission into a line. What the ramp is for is the
*direction* - which way the laser is being scanned at the instant under the
cursor, which the transmission alone cannot say.

It is carried, never fitted against: the alignment uses the transmission only,
so the ramp cannot move a frame. A record without the channel - every capture
made before it was recorded, and any .psdata exported without the column - is
drawn exactly as it was, with no second axis and no legend.

Frame boundaries are drawn as light shading so the 10 ms granularity is visible
- a peak narrower than one band was integrated whole into a single image.
"""

import argparse
import sys
from pathlib import Path

# --- the run parameters (defaults; the config file overrides them) ---------
# The values a run uses come from pico_scope/run_config_local.py, which is
# git-ignored - see run_config.py. Nothing here needs the command line; the
# arguments exist for scripting and override both when given.
ACTION = 'show'      # 'show' | 'self-test'
# 'clipboard' asks for the folder when the script runs, which is how this is
# normally used on its own; run_mode_video_pipeline.py passes --session instead
# and never reaches the prompt. A sentinel rather than a call at module scope,
# so importing this file or running its self-test does not stop and wait for a
# path nobody asked for - the prompt happens in main(), where it belongs.
SESSION = 'clipboard'  # 'clipboard' | '' = the newest capture | a path
SCOPE_FILE = ''      # the .psdata of a Phase 1 capture; '' for Phase 2
SHADE_ALPHA = 0.06   # faint: at 120 frames these are stripes until you zoom
# Matches kalishlot's CameraFitMixin.FIT_REBINNING, so the browser and this
# viewer fit the same frame the same way. 4x costs ~140 ms on a 448x1024
# frame and agrees with 2x to better than 1% on the widths.
FIT_REBINNING = 4
# With renormalize on, this percentile of the frame on screen is drawn at full
# brightness. Not the maximum: one hot pixel would then set the scale and leave
# the mode as dim as before. Brighter pixels than it saturate the colormap.
RENORMALIZE_PERCENTILE = 99.0

# Applied before matplotlib, because the backend below is chosen from ACTION
# and a config that asks for 'self-test' has to reach that line first.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pico_scope import run_config  # noqa: E402
CONFIG_CHANGES = run_config.apply('show', globals())

import matplotlib
matplotlib.use('Agg' if (ACTION == 'self-test' or '--self-test' in sys.argv)
               else 'Qt5Agg')

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Ellipse  # noqa: E402
from matplotlib.widgets import CheckButtons  # noqa: E402

from pico_scope.mode_video_sync import (ScopeTrace, find_scope_record,  # noqa: E402
                                        fit_session, frame_at_time,
                                        frame_brightness, frame_start_times,
                                        frame_windows, latest_session,
                                        load_scope_record, load_session,
                                        load_session_tail, load_session_trace,
                                        nearest_frame, release_frames)

# Only the threading comes from the device layer; the fitting itself is
# utilities.utils.fit_gaussian_beam, which reports the beam along its own
# principal axes (minor and major) instead of along the image's. A tilted mode
# has no width "along x", and the older routine's w_x / w_y named the image's
# axes for widths that were measured along the beam's.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'basler_cam'))
from gaussian_fit import FitLoop  # noqa: E402
from utilities.utils import (fit_gaussian_beam,  # noqa: E402
                             wait_for_path_from_clipboard)

HELP_TEXT = ('move: follow the cursor   click: pin/unpin   '
             'left/right: step (shift = 10)   f: fit a Gaussian   '
             'n: renormalize')
FIT_POLL_MS = 80        # how often the GUI thread looks for a finished fit


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
                 pixel_size_mm=None, camera_label='', tail=None):
        self.trace = trace
        # (ScopeTrace, meta) of the trailing scope capture, drawn in a panel of
        # its own beside the spectrum - never on the video's timebase, and never
        # part of the hover. None: the figure is exactly what it was.
        self.tail = tail
        self.frames = frames
        self.windows = np.asarray(windows)
        self.brightness = np.asarray(brightness)
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
        self._fit_timer = None
        self.renormalize = True     # on from the start; the box follows

        self._build_figure(title)
        self._connect()
        self.show_frame(0)

    # ------------------------------------------------------------- layout
    def _build_figure(self, title):
        self.fig = plt.figure(figsize=(15, 9))
        # with a tail the spectrum gives up its right-hand quarter to it
        width = 0.58 if self.tail is not None else 0.88
        self.ax_trace = self.fig.add_axes([0.07, 0.70, width, 0.23])
        self.ax_bright = self.fig.add_axes([0.07, 0.55, width, 0.12],
                                           sharex=self.ax_trace)
        self.ax_image = self.fig.add_axes([0.30, 0.06, 0.42, 0.44])

        lo, hi = self.windows[0, 0], self.windows[-1, 1]
        inside = (self.trace.t >= lo) & (self.trace.t <= hi)
        transmission, = self.ax_trace.plot(
            self.trace.t[inside] * 1e3, self.trace.signal[inside] * 1e3,
            lw=0.8, color='tab:blue', label='Channel D (transmission)')
        self.ax_trace.set_ylabel('Channel D [mV]')
        self.ax_trace.set_title(title, fontsize=10)
        self.ax_trace.tick_params(labelbottom=False)
        self._build_aux(inside, transmission)

        # Frame boundaries as light shading: the eye can then see that a peak
        # narrower than one band went into a single image whole.
        for start, end in self.windows[::2]:
            self.ax_trace.axvspan(start * 1e3, end * 1e3, color='k',
                                  alpha=SHADE_ALPHA, lw=0)

        centres = self.windows.mean(axis=1) * 1e3
        bar_width = np.median(np.diff(centres)) * 0.85 if centres.size > 1 else 1.0
        self.ax_bright.bar(centres, self.brightness, width=bar_width,
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
        self.shared_clim = (0, max(int(np.asarray(self.frames).max()), 1))
        self.image.set_clim(*self.shared_clim)

        self._build_tail()
        self._build_fit_controls()

    def _build_tail(self):
        """The trailing scope capture: the transmission (on the spectrum's own
        y scale, so the two compare) and the second channel, with a mark where
        the function generator's channel came on when the capture knows it."""
        self.ax_tail = None
        if self.tail is None:
            return
        trace, meta = self.tail
        self.ax_tail = self.fig.add_axes([0.78, 0.70, 0.15, 0.23],
                                         sharey=self.ax_trace)
        self.ax_tail.plot(trace.t * 1e3, trace.signal * 1e3, lw=0.8,
                          color='tab:blue')
        self.ax_tail.tick_params(labelleft=False)
        self.ax_tail.set_xlabel('trailing capture [ms from its start]',
                                fontsize=9)
        gap = meta.get('start_offset_s')
        note = '' if gap is None else f' (starts {gap * 1e3:.0f} ms after t = 0)'
        self.ax_tail.set_title('trailing scope capture' + note, fontsize=8)
        if trace.has_aux:
            twin = self.ax_tail.twinx()
            twin.plot(trace.t * 1e3, trace.aux, lw=0.8, color='tab:green',
                      alpha=0.75)
            twin.set_ylabel('[V]', fontsize=9, color='tab:green')
            twin.tick_params(axis='y', labelsize=8, colors='tab:green')
        fg = meta.get('function_generator') or {}
        # The output changes between the command being sent and it returning
        # (the instrument is slow to answer): a band when they differ.
        sent, done = fg.get('on_sent_after_start_s'), fg.get('on_done_after_start_s')
        if sent is None:
            sent = done = fg.get('on_after_start_s')   # an older capture: one time
        end = float(trace.t[-1]) if trace.t.size else 0.0
        if sent is not None and not fg.get('error') and sent > end:
            # an older capture, whose switch-on command was slow enough to
            # land after the recording ended: say so rather than stretch the axis
            self.ax_tail.text(0.5, 0.55, f'FG CH{fg.get("channel", "?")} came on\n'
                              f'{(sent - end) * 1e3:.0f} ms after this ended',
                              transform=self.ax_tail.transAxes, ha='center',
                              fontsize=7, color='crimson',
                              bbox=dict(fc='white', ec='none', alpha=0.8))
        elif sent is not None and not fg.get('error'):
            label = f'FG CH{fg.get("channel", "?")} on'
            if done is not None and done > sent:
                self.ax_tail.axvspan(sent * 1e3, done * 1e3, color='crimson',
                                     alpha=0.2, lw=0, label=label)
            else:
                self.ax_tail.axvline(sent * 1e3, color='crimson', lw=1.2,
                                     label=label)
            self.ax_tail.legend(loc='upper left', fontsize=7, framealpha=0.7)
        self.ax_tail.set_xlim(trace.t[0] * 1e3, end * 1e3)

    def _build_aux(self, inside, transmission):
        """The second channel, on its own y axis, when the record has one.

        Its own axis because the two are not comparable: the transmission is
        tens of millivolts and the temperature ramp is volts, so sharing a
        scale would flatten the transmission - the signal everything else here
        is about - into a line.

        A record without it (every capture before it was recorded, and any
        .psdata exported without the column) leaves the figure exactly as it
        was: no twin axis, no legend, nothing moved.
        """
        self.ax_aux = None
        if not self.trace.has_aux:
            return

        self.ax_aux = self.ax_trace.twinx()
        # The twin is created on top, which would draw the ramp over the
        # crimson cursor and the highlighted exposure band - the two things the
        # viewer exists to point at. Putting it behind and making the host
        # axes' own background transparent keeps the ramp visible underneath
        # while the markers stay on top of it.
        self.ax_aux.set_zorder(self.ax_trace.get_zorder() - 1)
        self.ax_trace.patch.set_visible(False)

        ramp, = self.ax_aux.plot(self.trace.t[inside] * 1e3,
                                 self.trace.aux[inside], lw=0.8,
                                 color='tab:green', alpha=0.75,
                                 label=self.trace.aux_label or 'aux channel')
        label = self.trace.aux_label or 'aux channel'
        self.ax_aux.set_ylabel(f'{label} [V]', fontsize=9, color='tab:green')
        self.ax_aux.tick_params(axis='y', labelsize=8, colors='tab:green')
        self.ax_trace.legend(handles=[transmission, ramp], loc='upper right',
                             fontsize=8, framealpha=0.7)

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

        self.ax_norm = self.fig.add_axes([0.755, 0.36, 0.115, 0.07])
        self.ax_norm.set_frame_on(False)
        self.check_norm = CheckButtons(self.ax_norm, ['renormalize'],
                                       [self.renormalize])
        self.check_norm.on_clicked(lambda _label: self.toggle_renormalize())

    def _connect(self):
        self.cids = [
            self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion),
            self.fig.canvas.mpl_connect('button_press_event', self.on_click),
            self.fig.canvas.mpl_connect('key_press_event', self.on_key),
            self.fig.canvas.mpl_connect('draw_event', self.on_draw),
        ]

    # -------------------------------------------------------- renormalize
    def toggle_renormalize(self, enabled=None):
        """Scale each frame to itself, or all frames to one; returns the state."""
        self.renormalize = ((not self.renormalize) if enabled is None
                            else bool(enabled))
        if self.check_norm.get_status()[0] != self.renormalize:
            # keeps the box in step when toggled by the 'n' key
            self.check_norm.eventson = False
            self.check_norm.set_active(0)
            self.check_norm.eventson = True
        self.show_frame(self.index)
        return self.renormalize

    def frame_clim(self, frame):
        """The colour limits for a frame: its own percentile, or the shared."""
        if not self.renormalize:
            return self.shared_clim
        top = float(np.percentile(frame, RENORMALIZE_PERCENTILE))
        if top <= 0:            # a frame dark below the percentile
            top = max(float(frame.max()), 1.0)
        return 0, top

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
                                         rebinning=FIT_REBINNING,
                                         fitter=fit_gaussian_beam)
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
        # Straight to the fitter, at the frame's own level:
        # fit_gaussian_beam takes its amplitude bound from the data, so a deep
        # binned frame needs no scaling down to fit inside a 12-bit one.
        self._fit_loop.submit(np.asarray(self.frames[self.index], dtype=float))

    def _on_fit_result(self, success, parameters):
        """Called on the fit thread: only store, never touch an artist."""
        self._fit_result = (success, parameters, self.index)

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

    def draw_fit(self, success, parameters, index):
        """Put a finished fit on the image. Safe to call directly in tests."""
        if not success:
            self.fit_ellipse.set_visible(False)
            self.fit_centre.set_data([], [])
            reason = parameters.get('reason', 'fit did not converge')
            self.fit_text.set_text(f'frame {index}: {reason}')
            self.fig.canvas.draw_idle()
            return

        w_minor, w_major = parameters['w_minor'], parameters['w_major']
        self.fit_ellipse.set_center((parameters['x_0'], parameters['y_0']))
        # w is the 1/e^2 radius, so the ellipse's full width is twice it. The
        # ellipse's width axis is its major one, and fit_gaussian_beam reports
        # that axis' direction in the convention Ellipse takes - so the angle
        # goes in as it comes out, with no sign to flip.
        self.fit_ellipse.set_width(2 * w_major)
        self.fit_ellipse.set_height(2 * w_minor)
        self.fit_ellipse.set_angle(parameters['major_axis_deg'])
        self.fit_ellipse.set_visible(True)
        self.fit_centre.set_data([parameters['x_0']], [parameters['y_0']])

        lines = [f'frame {index}',
                 f'x0 {parameters["x_0"]:7.1f} px',
                 f'y0 {parameters["y_0"]:7.1f} px']
        if self.pixel_size_mm:
            lines += [f'w_minor {w_minor * self.pixel_size_mm:7.3f} mm',
                      f'w_major {w_major * self.pixel_size_mm:7.3f} mm']
        else:
            lines += [f'w_minor {w_minor:7.1f} px', f'w_major {w_major:7.1f} px',
                      'pixel size unknown']
        lines.append(f'major axis {parameters["major_axis_deg"]:+6.1f} deg')
        lines.append(f'amp {parameters["amplitude"]:7.0f}')
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
        return index

    def show_frame(self, index, redraw=True):
        index = int(np.clip(index, 0, len(self.frames) - 1))
        self.index = index
        frame = np.asarray(self.frames[index])
        self.image.set_data(frame)
        self.image.set_clim(*self.frame_clim(frame))
        centre = self.windows[index].mean() * 1e3
        low, high = self.windows[index] * 1e3
        # axvspan gives a Rectangle here, so move it with the rectangle API
        self.window_patch.set_x(low)
        self.window_patch.set_width(high - low)
        self.cursor_trace.set_xdata([centre, centre])
        self.cursor_bright.set_xdata([centre, centre])
        state = 'pinned' if self.pinned is not None else 'following'
        if self.renormalize:
            state += f', p{RENORMALIZE_PERCENTILE:g} = max'
        self.image_title.set_text(
            f'frame {index} of {len(self.frames) - 1}   '
            f'{self.windows[index, 0] * 1e3:.2f}-{self.windows[index, 1] * 1e3:.2f} ms   '
            f'brightness {self.brightness[index]:.2f}   [{state}]')
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
        elif key == 'f':
            self.toggle_fit()
            return
        elif key == 'n':
            self.toggle_renormalize()
            return
        else:
            return
        if self.pinned is not None:
            self.pinned = int(np.clip(self.pinned, 0, len(self.frames) - 1))
        self.show_frame(self.pinned if self.pinned is not None else self.index)


# ------------------------------------------------------------------ loading
def load_synced_trace(session_path, scope_path=None):
    """Load a capture's trace/frames/offset, aligning it however it can.

    A Phase 2 capture carries its own scope trace and an offset, so nothing
    else is needed; a Phase 1 capture needs the .psdata that was recorded
    alongside it, and is aligned by fitting.

    Returns (trace, frames, windows, brightness, session, source,
    mark_target_path). `mark_target_path` is the file a modemarks.json
    sidecar for this trace belongs next to - the external .psdata for a
    Phase 1 capture (the same file extract_df_and_fsr_from_scope_csv.py would
    use for it), or the session's own recorded scope trace file for a
    Phase 2 capture - so a marking of this trace is cached and found the same
    way as any other marked measurement (see pico_scope/mode_marks_cache.py).
    """
    session_path = Path(session_path)
    if scope_path is not None:
        scope_path = Path(scope_path)
        result = fit_session(session_path, scope_path, verbose=True)
        trace, frames = result['trace'], result['frames']
        windows = result['windows']
        session = result['session']
        source = (f'fitted against {scope_path.name}, '
                  f'depth {result["best"].depth:.1f}x')
        mark_target_path = scope_path
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
        session_dir = session_path if session_path.is_dir() else session_path.parent
        mark_target_path = session_dir / session['scope']['file']

    brightness = np.array(session.get('brightness_masked')
                          or frame_brightness(frames), dtype=float)
    return (trace, frames, windows, brightness, session, source,
           mark_target_path)


class ScopeOnlyViewer:
    """A scope-only capture ("record scope only"): the transmission and, when
    recorded, the second channel on its own axis - no video, so no frames, no
    hover and nothing to fit. The window's toolbar zooms and pans."""

    def __init__(self, trace, title=''):
        self.trace = trace
        self.fig, self.ax_trace = plt.subplots(figsize=(13, 5))
        transmission, = self.ax_trace.plot(
            trace.t, trace.signal * 1e3, lw=0.8, color='tab:blue',
            label='Channel D (transmission)')
        self.ax_trace.set_ylabel('Channel D [mV]')
        self.ax_trace.set_xlabel('time in the scope record [s]')
        self.ax_trace.set_title(title, fontsize=10)
        self.ax_aux = None
        if trace.has_aux:
            self.ax_aux = self.ax_trace.twinx()
            label = trace.aux_label or 'aux channel'
            ramp, = self.ax_aux.plot(trace.t, trace.aux, lw=0.8,
                                     color='tab:green', alpha=0.75, label=label)
            self.ax_aux.set_ylabel(f'{label} [V]', fontsize=9, color='tab:green')
            self.ax_aux.tick_params(axis='y', labelsize=8, colors='tab:green')
            self.ax_trace.legend(handles=[transmission, ramp], loc='upper right',
                                 fontsize=8, framealpha=0.7)
        self.fig.tight_layout()

    def close_fit(self):
        pass


def viewer_from_session(session_path, scope_path=None):
    """Build a viewer from a capture - see load_synced_trace() for how it is
    aligned. A folder of a scope-only capture gets a ScopeOnlyViewer."""
    session_path = Path(session_path)
    if scope_path is None and find_scope_record(session_path) is not None:
        record, trace = load_scope_record(session_path)
        folder = session_path if session_path.is_dir() else session_path.parent
        return ScopeOnlyViewer(trace, f'{folder.name} - scope only, '
                                      f'{trace.duration:.3g} s')
    trace, frames, windows, brightness, session, source, _ = load_synced_trace(
        session_path, scope_path)
    title = f'{session_path.name} - {source}'
    return ModeSpectrumViewer(trace, frames, windows, brightness, title,
                              pixel_size_mm=session_pixel_size_mm(session),
                              camera_label=camera_label(session),
                              tail=None if scope_path is not None
                              else load_session_tail(session_path))


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
                                'self-test')
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

    # A record with no second channel - which is every capture made before it
    # was recorded - must look exactly as it did: no twin axis, no legend.
    assert not trace.has_aux
    assert viewer.ax_aux is None, 'no aux channel, no second axis'
    assert viewer.ax_trace.get_legend() is None
    assert viewer.ax_trace.patch.get_visible(), \
        'the host axes keeps its own background when there is nothing behind it'
    print('  a capture without the second channel is drawn exactly as before')

    # With one, it gets its own y axis rather than sharing the transmission's:
    # volts against tens of millivolts would flatten the transmission to a line
    ramp = 2.5 * np.sin(2 * np.pi * trace.t / max(trace.t[-1] - trace.t[0], 1e-9))
    with_aux = ScopeTrace(trace.t, trace.signal, 's', 'V', None,
                          aux=ramp, aux_unit='V',
                          aux_label='Temperature modulation Voltage')
    aux_viewer = ModeSpectrumViewer(with_aux, frames, windows, brightness,
                                    'aux')
    assert aux_viewer.ax_aux is not None, 'the second channel needs its own axis'
    assert aux_viewer.ax_aux is not aux_viewer.ax_trace
    assert 'Temperature modulation Voltage' in aux_viewer.ax_aux.get_ylabel()
    assert aux_viewer.ax_trace.get_legend() is not None, 'two traces, one legend'
    # behind the cursor and the exposure band, which are what the viewer points at
    assert aux_viewer.ax_aux.get_zorder() < aux_viewer.ax_trace.get_zorder()
    assert not aux_viewer.ax_trace.patch.get_visible(), \
        'the host background would hide the channel drawn behind it'
    # the two axes are on one timebase and the ramp keeps its own scale
    low, high = aux_viewer.ax_aux.get_ylim()
    assert high > 1.0, (low, high)     # volts, not the transmission's millivolts
    assert aux_viewer.ax_aux.get_xlim() == aux_viewer.ax_trace.get_xlim()
    # and the frame mapping is untouched by it: the alignment uses the
    # transmission alone and the second channel is carried, never fitted
    for probe in (3, 27, 61, 99):
        assert (aux_viewer.frame_for_time(windows[probe].mean())
                == viewer.frame_for_time(windows[probe].mean()) == probe)
    aux_viewer.close_fit()
    print('  the second channel gets its own y axis, behind the markers, and '
          'changes no frame the viewer maps to')

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
    print('  arrow keys step and clamp')

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
                                    'fit', pixel_size_mm=0.011,
                                    camera_label='ximea 11.0 um/px')
    ok, pars = fit_gaussian_beam(np.asarray(frames[int(np.argmax(brightness))],
                                            dtype=float), rebinning=1)
    assert ok, 'the synthetic blob must fit'
    # the blob is exp(-(r/3)^2), i.e. sigma = 3/sqrt(2), and w = 2 sigma
    expected_w = 2 * sigma_px / np.sqrt(2)
    for key in ('w_minor', 'w_major'):
        assert np.isclose(pars[key], expected_w, rtol=0.1), (key, pars[key], expected_w)
    assert np.isclose(pars['x_0'], 16, atol=0.5), pars['x_0']
    assert np.isclose(pars['y_0'], 12, atol=0.5), pars['y_0']

    fit_viewer.draw_fit(True, pars, 7)
    assert fit_viewer.fit_ellipse.get_visible()
    centre = fit_viewer.fit_ellipse.get_center()
    assert np.isclose(centre[0], pars['x_0']) and np.isclose(centre[1], pars['y_0'])
    # w is a radius, so the drawn contour is twice it across - and the
    # ellipse's width axis is the beam's MAJOR one, turned by the angle the fit
    # reports, with no sign flipped on the way
    assert np.isclose(fit_viewer.fit_ellipse.get_width(), 2 * pars['w_major'])
    assert np.isclose(fit_viewer.fit_ellipse.get_height(), 2 * pars['w_minor'])
    assert np.isclose(fit_viewer.fit_ellipse.get_angle(), pars['major_axis_deg'])
    readout = fit_viewer.fit_text.get_text()
    assert f'w_major {pars["w_major"] * 0.011:7.3f} mm' in readout, readout
    assert 'major axis' in readout, readout
    assert 'ximea 11.0 um/px' in readout
    print("  the fitted contour is drawn at 1/e^2, along the beam's own axes, "
          'and the widths are reported in millimetres, not pixels')

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

    # renormalize: each frame to its own percentile, and back to the shared scale
    # it opens with renormalize on
    assert viewer.renormalize is True
    assert viewer.check_norm.get_status()[0] is True
    assert viewer.toggle_renormalize() is False
    assert viewer.check_norm.get_status()[0] is False
    dim = int(np.argmin(brightness))
    viewer.show_frame(dim, redraw=False)
    assert viewer.image.get_clim() == viewer.shared_clim
    assert viewer.toggle_renormalize() is True
    assert viewer.check_norm.get_status()[0] is True
    frame = np.asarray(frames[dim])
    assert np.isclose(viewer.image.get_clim()[1],
                      np.percentile(frame, RENORMALIZE_PERCENTILE)
                      if np.percentile(frame, RENORMALIZE_PERCENTILE) > 0
                      else max(frame.max(), 1))
    viewer.show_frame(int(np.argmax(brightness)), redraw=False)
    assert viewer.image.get_clim()[1] <= viewer.shared_clim[1]
    viewer.on_key(_Key('n'))
    assert viewer.renormalize is False
    assert viewer.check_norm.get_status()[0] is False
    assert viewer.image.get_clim() == viewer.shared_clim
    print('  renormalize scales each frame to its own percentile, and off '
          'restores the one shared scale')

    plt.close(viewer.fig)

    # --- the trailing capture and scope-only captures --------------------
    import json
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        t = np.arange(2000) * 1e-4
        np.savez(folder / 'x_scope.npz', t=t, signal=np.sin(t * 50) * 0.01,
                 aux=t * 5)
        scope_block = {'file': 'x_scope.npz', 'aux': {'label': 'ramp'}}
        # a scope-only capture: *_scope.json and no session file
        (folder / 'x_scope.json').write_text(json.dumps({'scope': scope_block}))
        assert find_scope_record(folder) == folder / 'x_scope.json'
        only = viewer_from_session(folder)
        assert isinstance(only, ScopeOnlyViewer) and only.ax_aux is not None
        assert only.ax_trace.get_lines()[0].get_xdata()[-1] == t[-1]
        plt.close(only.fig)
        assert load_session_tail(folder) is None
        # a video capture is not one, and has no tail until it records one
        (folder / 'x_session.json').write_text(json.dumps({'scope': scope_block}))
        assert find_scope_record(folder) is None
        assert load_session_tail(folder) is None
        np.savez(folder / 'x_tail.npz', t=t[:500], signal=t[:500], aux=t[:500])
        (folder / 'x_session.json').write_text(json.dumps({
            'scope': scope_block,
            'scope_tail': {'file': 'x_tail.npz', 'start_offset_s': 2.0,
                           'function_generator': {
                               'channel': 2, 'on_sent_after_start_s': 0.01,
                               'on_done_after_start_s': 0.03}}}))
        tail_trace, tail_meta = load_session_tail(folder)
        assert tail_trace.t.size == 500 and tail_trace.has_aux
        assert tail_meta['start_offset_s'] == 2.0
    tail_viewer = ModeSpectrumViewer(trace, frames, windows, brightness, 'tail',
                                     tail=(tail_trace, tail_meta))
    assert tail_viewer.ax_tail is not None
    assert viewer.ax_tail is None, 'no tail, no panel'
    assert tail_viewer.ax_tail.get_ylim() == tail_viewer.ax_trace.get_ylim()
    for probe in (3, 61):         # the tail changes no frame the viewer maps to
        assert tail_viewer.frame_for_time(windows[probe].mean()) == probe
    tail_viewer.close_fit()
    plt.close(tail_viewer.fig)
    print('  a trailing capture gets a panel of its own, and a scope-only '
          'capture opens as a trace with no video')

    # the run-button configuration has to name something this file can do
    assert ACTION in ('show', 'self-test'), ACTION
    print('self-test passed')


def resolve_session():
    """The capture to open, from SESSION.

    'clipboard' asks for it, which is the usual way in when the viewer is run
    on its own; '' takes the newest capture; anything else is the path itself.
    Only reached when --session was not given, so the pipeline's handover of
    the burst it just recorded never stops for a clipboard.

    The prompt happens here rather than at module scope so that importing this
    file - mode_video_sync_mark.py does, for ModeSpectrumViewer - or running
    --self-test does not block waiting for a path nobody asked for.
    """
    if SESSION == 'clipboard':
        return wait_for_path_from_clipboard(filetype='folder')
    return SESSION or latest_session()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--config', default=None,
                        help='config file to run from, instead of '
                             'run_config_local.py')
    parser.add_argument('--self-test', action='store_true',
                        help='run the offline checks and exit')
    parser.add_argument('--session', default=None,
                        help="capture folder or *_session.json; defaults to "
                             "SESSION in the config - 'clipboard' to be asked "
                             "for it, '' for the newest capture")
    parser.add_argument('--scope', default=SCOPE_FILE or None,
                        help='the .psdata recorded alongside a Phase 1 capture; '
                             'omit for a Phase 2 capture, which carries its own')
    args = parser.parse_args()
    print(run_config.describe('show', CONFIG_CHANGES))

    if args.self_test or ACTION == 'self-test':
        _self_test()
        return
    if ACTION != 'show':
        raise SystemExit(f'ACTION must be show or self-test, not {ACTION!r}')

    session = args.session or resolve_session()
    print(f'session: {session}')
    viewer = viewer_from_session(session, args.scope)
    plt.show()
    viewer.close_fit()
    if hasattr(viewer, 'frames'):
        release_frames(viewer.frames)


if __name__ == '__main__':
    main()
