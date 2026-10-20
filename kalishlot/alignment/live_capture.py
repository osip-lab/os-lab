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
~1.5 ms (measured; nearly constant). What is left, the correction, is fitted
optically with mode_video_sync.fit_time_offset over +-0.1 s (well under the
~0.4 s between modes, so it cannot lock onto an alias) and kept in a ClockSync.

The ClockSync of a (scope, camera) pair is shared (shared_sync): the alignment
measurements and the quick synced capture use and refresh the same one. It is
cleared when either box is reopened, the scope restarts its stream (a rate or
range change) or the camera's ROI or exposure changes, and it says why.

A capture places its frames in one of three ways (`placement`):
  'track'      the alignment loop: fit +-15 ms around the kept correction (+-0.1 s
               when there is none) and keep the result
  'establish'  the "sync camera and scope" button: fit +-0.1 s, keep the result,
               raise SyncError if it does not lock (no blinks in the capture)
  'kept'       the quick capture: place the frames with the kept correction, as is;
               a +-15 ms fit only checks it (sync['check']), so a capture with no
               blinks at all is placed as well as one with many

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

import ctypes
import os
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
CHECK_WARN_S = 0.001        # a 'kept' capture whose check fit is further off: re-sync
RECORD_RAM_FRACTION = 0.5   # of the free memory, at most, for one recording's frames


class SyncError(RuntimeError):
    """No usable camera-to-scope correction (none kept, or the sync fit failed)."""


def available_memory_bytes():
    """Free physical memory now (Windows, else /proc/meminfo); None if unknown."""
    if os.name == 'nt':
        class Status(ctypes.Structure):
            _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong),
                        ('total_phys', ctypes.c_ulonglong), ('avail_phys', ctypes.c_ulonglong),
                        ('total_page', ctypes.c_ulonglong), ('avail_page', ctypes.c_ulonglong),
                        ('total_virtual', ctypes.c_ulonglong),
                        ('avail_virtual', ctypes.c_ulonglong),
                        ('avail_extended', ctypes.c_ulonglong)]
        status = Status()
        status.length = ctypes.sizeof(Status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.avail_phys)
        return None
    try:
        for line in Path('/proc/meminfo').read_text().splitlines():
            if line.startswith('MemAvailable:'):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


def recording_budget_bytes(fallback=1.0e9):
    """How many bytes of frames one recording may hold: a share of the free memory."""
    free = available_memory_bytes()
    return fallback if free is None else RECORD_RAM_FRACTION * free


def _what_changed(old, new):
    labels = {'scope_stream': 'the scope restarted its stream',
              'roi': 'the camera ROI changed', 'exposure_us': 'the camera exposure changed'}
    changed = [labels.get(k, k) for k in new if old.get(k) != new.get(k)]
    return ', '.join(changed) or 'the devices changed'


class ClockSync:
    """The kept camera-to-scope correction of one (scope, camera) pair.

    Valid while the stream key it was fitted under holds (the scope's stream
    generation, the camera's ROI and exposure); a capture under another key
    clears it, and `cleared` says why.
    """

    def __init__(self, cleared=None):
        self.correction_s = None    # fitted - host estimate of frame 0's start, s
        self.key = None             # {'scope_stream', 'roi', 'exposure_us'} it holds for
        self.depth = None           # the fit's depth when it was set
        self.method = None          # 'sync' (the button) | 'tracked' (alignment captures)
        self.set_at = None          # host time.time() it was last set
        self.established_at = None  # host time of the last 'establish'
        self.cleared = cleared      # why there is none, if there was one
        self.last_check = None      # the last 'kept' capture's check fit

    @property
    def valid(self):
        return self.correction_s is not None

    def set(self, correction_s, key, depth, method):
        self.correction_s, self.key, self.depth = float(correction_s), dict(key), float(depth)
        self.method, self.set_at, self.cleared = method, time.time(), None
        if method == 'sync':
            self.established_at = self.set_at

    def clear(self, reason):
        if self.valid:
            self.cleared = reason
        self.correction_s = self.key = self.depth = self.method = self.set_at = None
        self.established_at = self.last_check = None

    def check_key(self, key):
        """Clear the correction if it was fitted under another stream key."""
        if self.key is not None and dict(key) != self.key:
            self.clear(_what_changed(self.key, key))

    def to_dict(self):
        return {'valid': self.valid, 'correction_s': self.correction_s, 'depth': self.depth,
                'method': self.method, 'set_at': self.set_at,
                'established_at': self.established_at, 'cleared': self.cleared,
                'key': self.key, 'last_check': self.last_check}


# (scope id, camera id) -> (scope, camera, ClockSync): one per pair of open devices
_SHARED = {}


def shared_sync(scope_id, scope, camera_id, camera):
    """The ClockSync every capture from this scope and camera shares.

    `scope` / `camera` are the open objects. Another object under the same id
    means the box was closed and opened again, which starts a fresh, empty one:
    the device clocks restart with the device."""
    key = (scope_id, camera_id)
    entry = _SHARED.get(key)
    if entry is not None and entry[0] is scope and entry[1] is camera:
        return entry[2]
    cleared = None
    if entry is not None and entry[2].valid:
        cleared = ('the scope box was reopened' if entry[0] is not scope
                   else 'the camera box was reopened')
    elif entry is not None:
        cleared = entry[2].cleared
    sync = ClockSync(cleared)
    _SHARED[key] = (scope, camera, sync)
    return sync


def clear_shared(reason):
    for _, _, sync in _SHARED.values():
        sync.clear(reason)


def stream_key(scope, camera):
    """What a kept correction depends on (ClockSync.key), as the streams stand now."""
    geometry = camera.recording_geometry() if hasattr(camera, 'recording_geometry') else {}
    roi = geometry.get('roi')
    return {'scope_stream': getattr(scope, 'stream_generation', None),
            'roi': dict(roi) if roi else None, 'exposure_us': geometry.get('exposure_us')}


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
    timing: dict = field(default_factory=dict)  # seconds per stage of capture()

    def evaluate_args(self):
        """Positional args for alignment.metrics.evaluate_capture."""
        return self.t, self.signal, self.aux, self.frames, self.windows


class LiveCapture:
    def __init__(self, scope, camera=None, signal_channel='D', aux_channel='B', sync=None):
        self.scope = scope
        self.camera = camera
        self.signal_channel = signal_channel
        self.aux_channel = aux_channel
        self.sync = sync if sync is not None else ClockSync()   # shared_sync() to share one

    def reset_sync(self, reason='reset by hand'):
        self.sync.clear(reason)

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
    def capture(self, duration_s, with_frames=True, max_bytes=1.0e9, placement='track',
                pad_s=SCOPE_PAD_S):
        """Record `duration_s` from now; return LiveCaptureData.

        placement: 'track' | 'establish' | 'kept' (module docstring). pad_s: scope
        kept beyond the first and last frame (besides the +-0.1 s search margin)."""
        if placement not in ('track', 'establish', 'kept'):
            raise ValueError(f'unknown placement {placement!r}')
        if placement == 'kept' and with_frames and self.camera is not None:
            self.sync.check_key(stream_key(self.scope, self.camera))
            if not self.sync.valid:
                raise SyncError('camera and scope are not synced'
                                + (f' ({self.sync.cleared})' if self.sync.cleared else ''))
        if not with_frames or self.camera is None:
            t_start = self._scope_time(time.time())
            time.sleep(duration_s)
            t, signal, aux, sat = self._scope_span(t_start, t_start + duration_s)
            return LiveCaptureData(t, signal, aux, saturated=sat, sync={'mode': 'scope only'})

        clock = [time.perf_counter()]
        timing = {}

        def lap(name):
            now = time.perf_counter()
            timing[name] = now - clock[0]
            clock[0] = now

        self.camera.start_recording(max_s=duration_s + 1.0, max_bytes=max_bytes)
        try:
            time.sleep(duration_s)
        finally:
            rec = self.camera.stop_recording()
        lap('record')
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

        t, signal, aux, sat = self._scope_span(t0_host - pad_s - FIRST_SEARCH_S,
                                               t0_host + starts[-1] + exposure_s
                                               + pad_s + FIRST_SEARCH_S)
        lap('scope_span')       # mostly waiting for the scope stream to catch up
        frames = rec.stack()
        lap('stack_frames')
        mask = varying_pixel_mask(frames)
        brightness = frame_brightness(frames, mask if mask.any() else None)
        lap('brightness')

        key = {'scope_stream': self.scope.stream_generation,
               'roi': dict(rec.roi) if rec.roi else None,
               'exposure_us': meta[0]['exposure_us']}
        self.sync.check_key(key)
        if placement == 'kept' and not self.sync.valid:
            raise SyncError(f'the sync was cleared during the capture ({self.sync.cleared})')
        tracked = self.sync.valid and placement != 'establish'
        centre = t0_host + (self.sync.correction_s if tracked else 0.0)
        half = TRACK_SEARCH_S if tracked else FIRST_SEARCH_S
        trace = ScopeTrace(t, signal, 's', 'V')
        fit = fit_time_offset(starts, exposure_s, brightness, trace,
                              search=(centre - half, centre + half))
        lap('sync_fit')
        at_edge = min(fit.t0 - (centre - half), (centre + half) - fit.t0) < 0.1 * half
        good = bool(fit.locked and not at_edge)
        check = None
        if placement == 'kept':
            t0, mode = centre, 'kept correction'
            check = {'locked': good, 'depth': float(fit.depth),
                     'residual_s': float(fit.t0 - centre) if good else None}
            check['resync_advised'] = bool(good and abs(check['residual_s']) > CHECK_WARN_S)
            self.sync.last_check = check
        elif good:
            t0, mode = fit.t0, 'fit'
            self.sync.set(fit.t0 - t0_host, key, fit.depth,
                          'sync' if placement == 'establish' else 'tracked')
        elif placement == 'establish':
            found = (f'depth {fit.depth:.1f}' if np.isfinite(fit.depth)
                     else 'nothing varies in the frames')
            raise SyncError(f'the sync fit did not lock ({found}): it needs mode blinks - '
                            f'laser on, scan running, the cavity transmitting')
        elif tracked:
            t0, mode = centre, 'kept correction (fit not locked)'
        else:
            t0, mode = t0_host, 'host clock only (fit not locked)'
        sync = {'mode': mode, 't0_s': t0, 't0_host_s': t0_host,
                'correction_s': self.sync.correction_s, 'depth': fit.depth,
                'locked': bool(fit.locked), 'searched_s': half, 'check': check,
                'truncated': rec.truncated}
        return LiveCaptureData(t, signal, aux, frames, frame_windows(starts, exposure_s, t0),
                               meta, rec.roi, sync, sat, timing)
