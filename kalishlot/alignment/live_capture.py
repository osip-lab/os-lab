"""Measurements cut out of the continuously running scope and camera streams.

The alternative to a synchronized capture per evaluation (grab both devices,
configure, record, write, sync; kalishlot/MECHANICAL_CONTROL.md). Here both
devices keep streaming as they do in kalishlot, and a measurement only:
  1. arms the camera adapter's recorder for `duration_s` (every frame, with the
     camera's own timestamp), then disarms it;
  2. cuts the matching span out of the scope's ring buffer (the scope always
     holds the last ~70 s, so it needs no arming);
  3. places the frames on the scope's timebase.

Timing. Both streams carry host times: every camera frame (host_time_s next to
camera_timestamp_ns) and every scope driver callback (PicoScope4000A
.sample_at_host_time). That puts the frames on the scope's timebase to
~1.5 ms (measured; nearly constant). The first capture then fits the exact offset optically, with
mode_video_sync.fit_time_offset over +-0.1 s (well under the ~0.4 s between
modes, so it cannot lock onto an alias), and keeps the correction. Later
captures search only +-15 ms around it, to follow slow clock drift. The
correction is dropped when the scope restarts its stream or the camera's
geometry or exposure changes.

Drift (measured 2026-10-19, 12 captures over 3 min): the scope clock runs
-13 ppm against the host's, but both device clocks are re-tied to the host
clock in every capture, so the correction did not walk: +1.50 +- 0.22 ms,
no trend, the fit locked every time. Every capture is re-fitted anyway, so
the frame windows come from the data, not from an extrapolated clock.

The scope timebase is absolute: t = sample index / rate since the stream
started. Pure Python, no kalishlot imports: it takes a PicoScope4000A (device
layer) and any object with start_recording()/stop_recording() (the camera
adapters in kalishlot/adapters/camera_base.py).
"""

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO / 'pico_scope') not in sys.path:
    sys.path.insert(0, str(_REPO / 'pico_scope'))
from mode_video_sync import (ScopeTrace, fit_time_offset, frame_brightness,  # noqa: E402
                             frame_windows, varying_pixel_mask)

FIRST_SEARCH_S = 0.1        # first optical fit: +- around the host-clock estimate
TRACK_SEARCH_S = 0.015      # later fits: +- around the kept correction
SCOPE_PAD_S = 0.05          # scope kept before the first and after the last frame
SCOPE_WAIT_S = 2.0          # how long to wait for the scope ring to catch up


@dataclass
class LiveCaptureData:
    t: np.ndarray                   # scope time, s (absolute in the stream)
    signal: np.ndarray              # V
    aux: np.ndarray | None          # V (scan ramp)
    frames: np.ndarray | None = None        # (n, h, w)
    windows: np.ndarray | None = None       # (n, 2) exposure [start, end] in scope time
    meta: list = field(default_factory=list)
    roi: dict | None = None
    sync: dict = field(default_factory=dict)  # how the frames were placed
    saturated: bool = False         # the signal reached the scope's range

    def evaluate_args(self):
        """Positional args for alignment.metrics.evaluate_capture."""
        return self.t, self.signal, self.aux, self.frames, self.windows


class LiveCapture:
    def __init__(self, scope, camera=None, signal_channel='D', aux_channel='B'):
        self.scope = scope
        self.camera = camera
        self.signal_channel = signal_channel
        self.aux_channel = aux_channel
        self._correction = None     # fitted - host estimate of frame 0's start, s
        self._sync_key = None       # (scope stream generation, roi, exposure)

    def reset_sync(self):
        self._correction = None
        self._sync_key = None

    # ---------------------------------------------------------------- scope
    def _scope_time(self, host_time_s):
        idx = self.scope.sample_at_host_time(host_time_s)
        if idx is None:
            raise RuntimeError('the scope is not streaming')
        return idx / self.scope.sample_rate_hz

    def _scope_span(self, t_start, t_end):
        rate = self.scope.sample_rate_hz
        i0, i1 = int(np.floor(t_start * rate)), int(np.ceil(t_end * rate))
        names = [self.signal_channel] + ([self.aux_channel] if self.aux_channel else [])
        deadline = time.time() + SCOPE_WAIT_S
        while min(self.scope.samples_written(n) or 0 for n in names) < i1:
            if time.time() > deadline:
                raise RuntimeError('the scope stream fell behind the camera')
            time.sleep(0.02)
        raw = {n: self.scope.read_span(n, i0, i1) for n in names}
        missing = [n for n, v in raw.items() if v is None]
        if missing:
            raise RuntimeError(f'scope span no longer in the ring buffer: {missing}')
        signal = self.scope.to_volts(self.signal_channel, raw[self.signal_channel])
        aux = self.scope.to_volts(self.aux_channel, raw[self.aux_channel]) \
            if self.aux_channel else None
        t = np.arange(i0, i1) / rate
        rng = self.scope.channels[self.signal_channel]['range_v']
        saturated = bool(np.abs(signal).max() >= 0.995 * rng) if signal.size else False
        return t, signal, aux, saturated

    # --------------------------------------------------------------- capture
    def capture(self, duration_s, with_frames=True, max_bytes=1.0e9):
        """Record `duration_s` from now; return LiveCaptureData."""
        if not with_frames or self.camera is None:
            t_start = self._scope_time(time.time())
            time.sleep(duration_s)
            t, signal, aux, sat = self._scope_span(t_start, t_start + duration_s)
            return LiveCaptureData(t, signal, aux, saturated=sat, sync={'mode': 'scope only'})

        self.camera.start_recording(max_s=duration_s + 1.0, max_bytes=max_bytes)
        try:
            time.sleep(duration_s)
        finally:
            rec = self.camera.stop_recording()
        if rec is None or len(rec.frames) < 3:
            raise RuntimeError(f'the camera recorded {0 if rec is None else len(rec.frames)} '
                               f'frames (is it streaming?)')
        meta = rec.meta
        if 'camera_timestamp_ns' not in meta[0]:
            raise RuntimeError('the camera does not stamp streamed frames')
        exposure_s = float(meta[0]['exposure_us']) / 1e6
        cam_s = np.array([m['camera_timestamp_ns'] for m in meta], dtype=float) / 1e9
        host = np.array([m['host_time_s'] for m in meta])
        starts = cam_s - cam_s[0]                   # exposure starts relative to frame 0
        # host time of frame 0's exposure start: the frame reaches the host some
        # latency after its exposure ends, never before
        host0 = float(np.min(host - starts)) - exposure_s
        t0_host = self._scope_time(host0)

        t, signal, aux, sat = self._scope_span(t0_host - SCOPE_PAD_S - FIRST_SEARCH_S,
                                               t0_host + starts[-1] + exposure_s
                                               + SCOPE_PAD_S + FIRST_SEARCH_S)
        frames = rec.stack()
        mask = varying_pixel_mask(frames)
        brightness = frame_brightness(frames, mask if mask.any() else None)

        key = (self.scope.stream_generation, rec.roi, meta[0]['exposure_us'])
        if key != self._sync_key:
            self.reset_sync()
            self._sync_key = key
        tracked = self._correction is not None
        centre = t0_host + (self._correction or 0.0)
        half = TRACK_SEARCH_S if tracked else FIRST_SEARCH_S
        trace = ScopeTrace(t, signal, 's', 'V')
        fit = fit_time_offset(starts, exposure_s, brightness, trace,
                              search=(centre - half, centre + half))
        at_edge = min(fit.t0 - (centre - half), (centre + half) - fit.t0) < 0.1 * half
        if fit.locked and not at_edge:
            t0, mode = fit.t0, 'fit'
            self._correction = fit.t0 - t0_host
        elif tracked:
            t0, mode = centre, 'kept correction (fit not locked)'
        else:
            t0, mode = t0_host, 'host clock only (fit not locked)'
        sync = {'mode': mode, 't0_s': t0, 't0_host_s': t0_host,
                'correction_s': self._correction, 'depth': fit.depth,
                'locked': bool(fit.locked), 'searched_s': half}
        return LiveCaptureData(t, signal, aux, frames, frame_windows(starts, exposure_s, t0),
                               meta, rec.roi, sync, sat)
