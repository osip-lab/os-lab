"""Live measurement: alignment metrics from the scope and camera boxes' streams.

A virtual device (no hardware), like synced_pipeline. It owns no instrument:
on each `measure` it finds the open PicoScope box and an open camera box that
can record (the XIMEA), arms the camera's recorder, cuts the scope span, and
runs alignment/metrics on it (see alignment/live_capture.py and
LIVE_CAPTURE.md). The boxes keep streaming throughout, so the live view stays
on while a script steps the motors and measures.

First piece of the aligner box (MECHANICAL_CONTROL.md): the loop will call
the same measurement in-process.

Commands (args):
    measure {duration_s=3.0, with_frames=True, camera=None, scope=None}
        camera / scope: a device_id; None = the only open one of that kind.
        save: a .npz path to also write the raw capture to (offline debugging).
        profile: True = also return where the metrics' time went ('profile_s').
        fast: True (default) = alignment-loop metrics (MetricsConfig.fast: fitted
              on a rebinned ROI around the spot, ~1 s); False = the full metrics
              incl. first-order camera templates (~10 s).
        Returns {'metrics': {name: Metric dict}, 'by_direction', 'direction',
                 'sync', 'n_frames', 'n_samples', 'saturated', 'timing_s'}.
    reset_sync {}   forget the kept camera-to-scope correction
Events: {'type': 'measurement', ...the measure result}
"""

import math
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # kalishlot/: alignment package
from alignment.live_capture import LiveCapture  # noqa: E402
from alignment.metrics import MetricsConfig, evaluate_capture  # noqa: E402

from .base import DeviceAdapter  # noqa: E402

SCOPE_TYPE = 'picoscope'
MIN_FUNDAMENTAL_OVERLAP = 0.85  # shape guard: live 0th orders 0.96-0.985, first orders 0.48-0.66
FIRST_ORDER_S_MEMORY = 10       # recent captures whose first-order offset is remembered
SIGNAL_CHANNEL = 'D'        # cavity transmission
AUX_CHANNEL = 'B'           # temperature-modulation ramp


class LiveMeasureAdapter(DeviceAdapter):
    type_name = 'live_measure'
    display_name = 'Live measurement (scope + camera streams)'

    # Set by server.py: () -> (devices dict, loans dict)
    registry = None

    @staticmethod
    def list_available():
        return [{'address': 'main', 'label': 'Live measurement'}]

    def __init__(self, address):
        super().__init__(address)
        self._lock = threading.Lock()
        self._captures = {}         # (scope id, camera id) -> LiveCapture (keeps its sync)
        self._last = None
        self._first_order_s = []     # |s| of the first order in recent valid captures

    def open(self):
        pass

    def close(self):
        self._captures.clear()

    def describe(self):
        return {'type': self.type_name, 'label': self.display_name,
                'commands': ['measure', 'reset_sync'], 'last': self._last}

    # ------------------------------------------------------------ devices
    def _pick(self, wanted, test, what):
        devices = dict(self.registry()[0]) if self.registry is not None else {}
        found = {i: a for i, a in devices.items() if test(a)}
        if wanted is not None:
            if wanted not in found:
                raise ValueError(f'{what} {wanted!r} is not open')
            return wanted, found[wanted]
        if len(found) != 1:
            raise ValueError(f'open exactly one {what} box (found {len(found)}), '
                             f'or name one')
        return next(iter(found.items()))

    def _devices(self, scope_id, camera_id):
        scope_id, scope_adapter = self._pick(
            scope_id, lambda a: getattr(a, 'type_name', None) == SCOPE_TYPE, 'scope')
        camera_id, camera_adapter = self._pick(
            camera_id, lambda a: hasattr(a, 'start_recording')
            and getattr(a, 'type_name', None) == 'ximea_camera', 'XIMEA camera')
        return scope_id, scope_adapter, camera_id, camera_adapter

    # ------------------------------------------------------------ commands
    def command(self, name, args):
        args = args or {}
        if name == 'reset_sync':
            for live in self._captures.values():
                live.reset_sync()
            return {'ok': True}
        if name == 'measure':
            if not self._lock.acquire(blocking=False):
                raise ValueError('a measurement is already running')
            try:
                return self._measure(float(args.get('duration_s', 3.0)),
                                     bool(args.get('with_frames', True)),
                                     args.get('scope'), args.get('camera'), args.get('save'),
                                     bool(args.get('fast', True)), bool(args.get('profile', False)))
            finally:
                self._lock.release()
        raise ValueError(f'unknown command {name!r}')

    def _measure(self, duration_s, with_frames, scope_id, camera_id, save=None, fast=True,
                 profile=False):
        scope_id, scope_adapter, camera_id, camera = self._devices(scope_id, camera_id)
        scope = scope_adapter.scope
        if not scope.is_streaming:
            raise ValueError('the scope box is not streaming')
        for ch in (SIGNAL_CHANNEL, AUX_CHANNEL):
            if not scope.channels[ch]['enabled']:
                raise ValueError(f'enable scope channel {ch} '
                                 f'({"transmission" if ch == SIGNAL_CHANNEL else "scan ramp"})')
        key = (scope_id, camera_id)
        live = self._captures.get(key)
        if live is None:
            live = self._captures[key] = LiveCapture(scope, camera, SIGNAL_CHANNEL, AUX_CHANNEL)
        tic = time.time()
        data = live.capture(duration_s, with_frames=with_frames)
        captured = time.time() - tic
        roi = data.roi or {'x': 0, 'y': 0}
        cfg = MetricsConfig(
            saturation_v=0.995 * scope.channels[SIGNAL_CHANNEL]['range_v'], fast=fast,
            min_fundamental_overlap=MIN_FUNDAMENTAL_OVERLAP,
            recover_first_order_s=(float(np.median(self._first_order_s))
                                   if self._first_order_s else None),
            sensor_geometry={'offset_x': roi['x'], 'offset_y': roi['y'],
                             'binning_x': 1, 'binning_y': 1} if with_frames else None)
        if save:
            # raw capture for offline work (np.load(...)); arrays as evaluate_capture takes them
            Path(save).parent.mkdir(parents=True, exist_ok=True)
            np.savez(save, t=data.t, signal=data.signal, aux=data.aux, frames=data.frames,
                     windows=data.windows, roi=np.array([roi['x'], roi['y']]),
                     saturation_v=cfg.saturation_v)
        tic_m = time.time()
        if profile:
            import cProfile
            prof = cProfile.Profile()
            res = prof.runcall(evaluate_capture, *data.evaluate_args(), cfg=cfg)
        else:
            res = evaluate_capture(*data.evaluate_args(), cfg=cfg)
        metrics_s = time.time() - tic_m
        fo = res.metrics.get('first_order_s')
        if fo is not None and fo.valid and fo.value > 0:
            self._first_order_s = (self._first_order_s + [fo.value])[-FIRST_ORDER_S_MEMORY:]
        rows = [r for sw in res.sweeps for r in sw['fundamentals']]
        blinks = {'zeroth_order_picks': len(rows),
                  'used': sum(bool(r['usable']) for r in rows),
                  'wrong_mode_rejected': sum(r['why'] in ('frame not Gaussian (higher order?)',
                                                          'not the same mode as the 0th order')
                                             for r in rows),
                  'recovered': sum(sw.get('recovered', 0) for sw in res.sweeps),
                  'first_order_s_used': cfg.recover_first_order_s}
        result = {
            'metrics': {k: m.to_dict() for k, m in res.metrics.items()},
            'by_direction': {str(d): {k: m.to_dict() for k, m in ms.items()}
                             for d, ms in res.by_direction.items()},
            'direction': res.direction,
            'sync': {k: v for k, v in data.sync.items()},
            'n_frames': 0 if data.frames is None else int(len(data.frames)),
            'n_samples': int(data.t.size),
            'saturated': data.saturated,
            'blinks': blinks,
            'timing_s': {'capture': captured, 'capture_stages': data.timing,
                         'metrics': metrics_s},
            'time': time.time(),
        }
        if profile:
            result['profile_s'] = _profile_summary(prof)
        result = _json_safe(result)
        self._last = result
        self.emit({'type': 'measurement', **result})
        return result


# where the metrics spend their time: cumulative seconds of these functions
# (nested: camera_fits is inside frame_scores, peak_fits inside find_0th_orders)
_PROFILED = {'split_sweeps': ('sweeps.py', 'split_sweeps'),
             'find_0th_orders': ('spectrum.py', 'find_fundamentals'),
             'fsr_estimate': ('spectrum.py', 'estimate_fsr_time'),
             'peak_fits': ('spectrum.py', 'fit_peak'),
             'side_peaks_and_families': ('spectrum.py', 'side_offsets'),
             'first_orders': ('spectrum.py', 'first_orders'),
             'frame_scores': ('metrics.py', '_frame_score'),
             'camera_fits': ('camera.py', 'fast_gaussian_score'),
             'camera_fits_full': ('camera.py', 'gaussian_score'),
             'same_mode_check': ('metrics.py', '_same_mode_impostors'),
             'wrong_mode_recovery': ('metrics.py', '_recover_wrong_mode')}


def _profile_summary(prof):
    import pstats
    stats = pstats.Stats(prof).stats
    out = {}
    for label, (file, func) in _PROFILED.items():
        total = sum(v[3] for (f, _, fn), v in stats.items()
                    if fn == func and f.replace(chr(92), '/').endswith(file))
        if total:
            out[label] = total
    return out


def _json_safe(value):
    """NaN/inf -> None and numpy scalars -> Python (JSON has no NaN)."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if hasattr(value, 'item') and not isinstance(value, (str, bytes)):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value
