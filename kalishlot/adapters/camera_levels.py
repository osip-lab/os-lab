"""Frame-intensity history shared by camera adapters.

Mixin adding the levels vocabulary on top of DeviceAdapter: commands
levels_on / levels_off, the 'levels_status' event, and the periodic 'levels'
event carrying the last LEVELS_WINDOW_S of three numbers per sampled frame.
It answers "did the light change when I turned that knob?" — a live strip
chart beside the image, deliberately nothing more: the ring buffer forgets
anything older than the window, and nothing is ever written to disk.

Why those three numbers, on a beam image:
  median  the background, and with it the camera's own offset and any stray
          room light — the baseline everything else should be read against
  p99     the beam itself, without being at the mercy of one hot pixel the
          way the maximum is
  max     saturation: the one value that says the frame has stopped being a
          measurement (compare it with describe()'s levels_max)

Sampling is throttled to LEVELS_SAMPLE_INTERVAL_S rather than run per frame:
this is a monitor of slow changes, the statistics cost milliseconds in the
camera's own thread, and a 100 Hz camera must not pay for a 10 Hz question.
"""

import threading
import time

import numpy as np

LEVELS_WINDOW_S = 30.0            # how far back the chart remembers
LEVELS_SAMPLE_INTERVAL_S = 0.1    # 10 Hz: one point per 100 ms of frames
LEVELS_EMIT_INTERVAL_S = 0.2      # 5 Hz: how often the window is broadcast
# Pixels to measure the quantiles on. A quarter of a million pins them far
# tighter than the shot noise between two frames does; see frame_levels().
LEVELS_QUANTILE_SAMPLES = 250_000


def frame_levels(frame):
    """(median, 99th percentile, maximum) of a frame's pixel values.

    The quantiles are measured on a regular subsample, the maximum on every
    pixel. Measured on a 2048x2048 frame: the exact quantiles cost ~46 ms,
    the subsampled ones ~4 ms and differ by ~0.07% — while the maximum costs
    1.4 ms on the whole frame and is the one statistic a subsample would get
    badly wrong, because it exists to catch the single hot spot.
    """
    step = max(1, int(np.sqrt(frame.size / LEVELS_QUANTILE_SAMPLES)))
    median, p99 = np.percentile(frame[::step, ::step], (50, 99))
    return round(float(median), 1), round(float(p99), 1), int(frame.max())


class CameraLevelsMixin:
    LEVELS_WINDOW_S = LEVELS_WINDOW_S

    def _init_levels(self):
        self._levels_on = False
        # (monotonic seconds, median, p99, max), newest last
        self._levels = []
        self._levels_lock = threading.Lock()
        self._levels_last_sample = 0.0
        self._levels_last_emit = 0.0

    def _store_levels_frame(self, frame):
        """Call from the frame-producing thread with each full-res frame."""
        if not self._levels_on:
            return
        now = time.monotonic()
        # the 10% tolerance keeps the throttle from beating against the frame
        # rate: at exactly 10 Hz, a strict comparison would reject every other
        # frame by a microsecond and sample at 5 Hz instead of 10
        if now - self._levels_last_sample < 0.9 * LEVELS_SAMPLE_INTERVAL_S:
            return
        self._levels_last_sample = now
        point = (now,) + frame_levels(frame)
        with self._levels_lock:
            self._levels.append(point)
            oldest = now - self.LEVELS_WINDOW_S
            first = 0
            while first < len(self._levels) and self._levels[first][0] < oldest:
                first += 1
            if first:
                del self._levels[:first]
        if now - self._levels_last_emit >= LEVELS_EMIT_INTERVAL_S:
            self._levels_last_emit = now
            self.emit(self.levels_event())

    def levels_event(self):
        """The whole visible window, as one event.

        The window rather than the newest point, for the same reason the
        scope box redraws from a whole chunk: 'levels' is coalesced for a
        stalled viewer (server.py COALESCE_EVENT_TYPES), so a dropped event
        must cost that viewer nothing but a late redraw — never a hole in
        the trace. Times are seconds BEFORE now, so nothing depends on the
        browser's clock agreeing with this machine's.
        """
        now = time.monotonic()
        with self._levels_lock:
            points = list(self._levels)
        # Age out here as well as on append: the ring is only trimmed when a
        # frame arrives, so between frames — and forever, on a paused camera
        # — its oldest points drift past the window. The chart's x axis is
        # "seconds ago", and a point outside the window has no place on it.
        window = self.LEVELS_WINDOW_S
        fresh = []
        for timestamp, median, p99, maximum in points:
            age = round(timestamp - now, 2)
            if age >= -window:
                fresh.append([age, median, p99, maximum])
        return {'type': 'levels', 'window_s': window, 'points': fresh}

    def _clear_levels(self):
        """Forget the history — the frames stopped being comparable.

        Cropping to an ROI changes which pixels are being measured, so the
        trace would step for a reason that has nothing to do with the light.
        """
        with self._levels_lock:
            self._levels.clear()

    def levels_describe(self):
        """Merge into describe() so a re-attaching viewer redraws at once."""
        described = {'levels': self._levels_on,
                     'levels_window_s': self.LEVELS_WINDOW_S}
        if self._levels_on:
            described['levels_points'] = self.levels_event()['points']
        return described

    def levels_command(self, name, args):
        """Handle a levels command; return None if `name` is not one of them."""
        if name in ('levels_on', 'levels_off'):
            self._levels_on = name == 'levels_on'
            self._clear_levels()
            # let the next frame be sampled rather than waiting out the
            # throttle from whenever the monitor was last switched off
            self._levels_last_sample = 0.0
            self._levels_last_emit = 0.0
            self.emit({'type': 'levels_status', 'enabled': self._levels_on})
            return {'ok': True}
        return None
