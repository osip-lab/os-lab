"""PicoScope 4000A streaming-scope adapter: rolling chart-recorder view.

Wraps the pure device layer in picoscope/ps4000a_scope.py. The smooth-plot
recipe (never a per-sample redraw anywhere):
  device thread -> numpy ring buffers (device layer)
  emitter thread here, ~20 Hz -> min/max envelope decimation of the visible
  window to <= MAX_POINTS per channel -> one 'scope_data' JSON event
  browser -> one uPlot setData() per event.
Mutually exclusive with PicoScope 7 (single owner) — open failure surfaces
as the standard 409 popup.
"""

import sys
import threading
import time
from pathlib import Path

import numpy as np

# the device layer lives at the repo root, outside kalishlot/
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / 'picoscope'))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))  # for the pico_scope analysis package
from ps4000a_scope import CHANNEL_NAMES, RANGES, PicoScope4000A  # noqa: E402
from pico_scope.mode_analysis import decimate  # noqa: E402

from .analyses.scope_pairs import PairsAnalysis  # noqa: E402
from .analyses.scope_sidebands import SidebandsAnalysis  # noqa: E402
from .base import DeviceAdapter  # noqa: E402

EMIT_INTERVAL_S = 0.05  # ~20 chunks/s to the browsers
MAX_POINTS = 1000  # per channel per chunk (500 min/max pairs)
WINDOW_CHOICES_S = (0.1, 1.0, 10.0, 60.0)
RATE_CHOICES_HZ = (100.0, 1000.0, 10_000.0, 100_000.0)


def envelope(samples, max_points):
    """Decimate to <= max_points, keeping each bucket's min AND max so
    narrow spikes stay visible (standard oscilloscope display trick)."""
    n = len(samples)
    if n <= max_points:
        return samples, 1
    buckets = max_points // 2
    per_bucket = n // buckets
    trimmed = samples[n - buckets * per_bucket:]  # newest-aligned
    blocks = trimmed.reshape(buckets, per_bucket)
    out = np.empty(2 * buckets, dtype=samples.dtype)
    out[0::2] = blocks.min(axis=1)
    out[1::2] = blocks.max(axis=1)
    return out, per_bucket / 2  # each output point spans half a bucket


class PicoScopeAdapter(DeviceAdapter):
    type_name = 'picoscope'
    display_name = 'PicoScope'

    @staticmethod
    def list_available():
        return [{'address': d['serial'], 'label': f"PicoScope s/n {d['serial']}"}
                for d in PicoScope4000A.list_devices()]

    def __init__(self, address):
        super().__init__(address)
        self.scope = PicoScope4000A(serial=address)
        self.scope.on_error = lambda error: self.emit(
            {'type': 'error', 'message': str(error)})
        self.window_s = 10.0
        self._playing = threading.Event()
        self._stopping = threading.Event()
        self._emitter = None
        self._snapshot = None  # full-res data frozen at pause: {'dt', 'channels'}
        # analysis extensions: each owns its commands and reattach state and
        # uses this adapter as its host (snapshot_region + emit). See
        # kalishlot/ADDING_ANALYSES.md for the recipe.
        self.analyses = [SidebandsAnalysis(self), PairsAnalysis(self)]

    # ------------------------------------------------------------ lifecycle
    def open(self):
        self.scope.open()
        self.scope.start_streaming()
        self._playing.set()
        self._stopping.clear()
        self._emitter = threading.Thread(target=self._emit_loop, daemon=True,
                                         name=f'pico-emit-{self.address}')
        self._emitter.start()

    def close(self):
        self._stopping.set()
        if self._emitter is not None:
            self._emitter.join(timeout=5)
            self._emitter = None
        self.scope.close()

    def describe(self):
        description = {
            'type': self.type_name,
            'label': f'PicoScope {self.scope.variant or ""} — '
                     f's/n {self.address}',
            'commands': ['play', 'pause', 'set_setting', 'set_channel']
                        + [name for analysis in self.analyses
                           for name in analysis.COMMANDS],
            'playing': self._playing.is_set(),
            'channels': {name: dict(config) for name, config
                         in self.scope.channels.items()},
            'ranges_v': sorted(RANGES.values()),
            'window_choices_s': list(WINDOW_CHOICES_S),
            'rate_choices_hz': list(RATE_CHOICES_HZ),
            'settings': [
                {'name': 'sample_rate_hz', 'label': 'sample rate',
                 'unit': 'S/s', 'value': self.scope.sample_rate_hz},
                {'name': 'window_s', 'label': 'window', 'unit': 's',
                 'value': self.window_s},
            ]}
        for analysis in self.analyses:  # e.g. 'analysis', 'analysis_pairs'
            description.update(analysis.describe_state())
        return description

    def settings_snapshot(self):
        return {'sample_rate_hz': self.scope.sample_rate_hz,
                'window_s': self.window_s,
                'channels': {name: dict(config) for name, config
                             in self.scope.channels.items()}}

    def restore_settings(self, snapshot):
        # only touch what actually differs: every channel/rate change is a
        # stop-reconfigure-restart of the streaming (audible relay clicks)
        for name, saved in (snapshot.get('channels') or {}).items():
            current = self.scope.channels.get(name)
            if current is None or all(saved.get(key) == current.get(key)
                                      for key in current):
                continue
            self.scope.configure_channel(name,
                                         enabled=saved.get('enabled'),
                                         coupling=saved.get('coupling'),
                                         range_v=saved.get('range_v'))
        rate = snapshot.get('sample_rate_hz')
        if rate and rate != self.scope.sample_rate_hz:
            self.scope.set_sample_rate(float(rate))
        window = snapshot.get('window_s')
        if window:
            self.window_s = float(np.clip(window, 0.01, 60.0))

    # ------------------------------------------------------------- commands
    def command(self, name, args):
        for analysis in self.analyses:
            result = analysis.command(name, args)
            if result is not None:
                return result
        if name == 'play':
            self._playing.set()
            self._snapshot = None
            for analysis in self.analyses:
                analysis.reset()  # overlays belong to the frozen data
            self.emit({'type': 'status', 'playing': True})
            return {'ok': True}
        if name == 'pause':
            # pause FREEZES THE DATA, not just the display: the window is
            # captured at full resolution so analysis (and every viewer's
            # chart) works on exactly what is on screen. Acquisition keeps
            # running underneath so resume is instant.
            dt, window = self.scope.read_window(self.window_s)
            self._snapshot = {'dt': dt, 'channels': window}
            self._playing.clear()
            try:
                self._emit_chunk(self._snapshot)
            except Exception as error:
                self.emit({'type': 'error', 'message': str(error)})
            self.emit({'type': 'status', 'playing': False})
            return {'ok': True}
        if name == 'set_setting':
            setting = args['name']
            value = float(args['value'])
            if setting == 'sample_rate_hz':
                accepted = self.scope.set_sample_rate(value)
            elif setting == 'window_s':
                self.window_s = float(np.clip(value, 0.01, 60.0))
                accepted = self.window_s
            else:
                raise ValueError(f'unknown setting {setting!r}')
            self.emit({'type': 'setting_applied', 'name': setting,
                       'value': accepted})
            return {'ok': True, 'value': accepted}
        if name == 'set_channel':
            channel = args['channel']
            accepted = self.scope.configure_channel(
                channel,
                enabled=args.get('enabled'),
                coupling=args.get('coupling'),
                range_v=args.get('range_v'))
            self.emit({'type': 'channel', 'channel': channel,
                       'state': accepted})
            return {'ok': True, 'state': accepted}
        raise ValueError(f'unknown command {name!r}')

    # -------------------------------------------------------- emitter thread
    def _emit_loop(self):
        while not self._stopping.is_set():
            tic = time.time()
            if self._playing.is_set():
                try:
                    self._emit_chunk()
                except Exception as error:
                    self.emit({'type': 'error', 'message': str(error)})
            elapsed = time.time() - tic
            self._stopping.wait(max(EMIT_INTERVAL_S - elapsed, 0.005))

    def _emit_chunk(self, snapshot=None):
        if snapshot is not None:
            dt, window = snapshot['dt'], snapshot['channels']
        else:
            dt, window = self.scope.read_window(self.window_s)
        channels = {}
        n_max = 0
        for name, adc in window.items():
            if not len(adc):
                continue
            decimated, stride = envelope(adc, MAX_POINTS)
            volts = self.scope.to_volts(name, decimated)
            channels[name] = [round(float(v), 5) for v in volts]
            n_max = max(n_max, len(adc))
        if not channels:
            return
        self.emit({'type': 'scope_data',
                   'window_s': self.window_s,
                   'span_s': n_max * dt,  # actual data span (fills up after start)
                   'channels': channels})

    # -------------------------------------------- analysis on the snapshot
    def snapshot_region(self, args):
        """Host service for the analysis extensions: guard-check the paused
        snapshot and return the fit input — the selected region decimated to
        fit density as (t, volts, t_min, t_max). Times are on the chart's
        axis: newest sample at 0, past negative. ValueErrors surface as
        HTTP 400 with the message in the box."""
        if self._playing.is_set() or self._snapshot is None:
            raise ValueError('pause the stream first — '
                             'analysis runs on the frozen snapshot')
        channel = args['channel']
        adc = self._snapshot['channels'].get(channel)
        if adc is None or not len(adc):
            raise ValueError(f'no snapshot data for channel {channel!r}')
        dt = self._snapshot['dt']
        t = (np.arange(len(adc)) - (len(adc) - 1)) * dt
        t_min, t_max = float(args['t_min']), float(args['t_max'])
        mask = (t >= t_min) & (t <= t_max)
        if mask.sum() < 20:
            raise ValueError('selected region holds too few samples')
        x_fit, y_fit = decimate(t[mask], self.scope.to_volts(channel, adc[mask]))
        return x_fit, y_fit, t_min, t_max

