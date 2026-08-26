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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pico_scope.mode_video_sync import (fit_session, frame_at_time,  # noqa: E402
                                        frame_brightness, frame_start_times,
                                        frame_windows, latest_session,
                                        load_session, load_session_trace,
                                        nearest_frame, release_frames)

SNAP_RADIUS = 1          # frames either side, when snapping to the brightest
SHADE_ALPHA = 0.06      # faint: at 120 frames these are stripes until you zoom
HELP_TEXT = ('move: follow the cursor   click: pin/unpin   '
             'left/right: step (shift = 10)   b: snap-to-brightest')


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
                 snap=True):
        self.trace = trace
        self.frames = frames
        self.windows = np.asarray(windows)
        self.brightness = np.asarray(brightness)
        self.snap = snap
        self.pinned = None
        self.index = 0
        self.background = None
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

    def _connect(self):
        self.cids = [
            self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion),
            self.fig.canvas.mpl_connect('button_press_event', self.on_click),
            self.fig.canvas.mpl_connect('key_press_event', self.on_key),
            self.fig.canvas.mpl_connect('draw_event', self.on_draw),
        ]

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
    return ModeSpectrumViewer(trace, frames, windows, brightness, title, snap)


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
    release_frames(viewer.frames)


if __name__ == '__main__':
    main()
