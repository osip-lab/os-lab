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
from pico_scope.mode_analysis import (DEFAULT_SIDEBAND_FREQ_MHZ,  # noqa: E402
                                      SIX_LORENTZIAN_PARAMS, decimate,
                                      fit_six_lorentzians,
                                      get_na_interpolators, sideband_results,
                                      six_lorentzian_model)

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
        self._last_analysis = None  # last 'analysis_result' event, for reattach

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
        return {'type': self.type_name,
                'label': f'PicoScope {self.scope.variant or ""} — '
                         f's/n {self.address}',
                'commands': ['play', 'pause', 'set_setting', 'set_channel',
                             'analyze_sidebands'],
                'playing': self._playing.is_set(),
                'analysis': self._last_analysis,
                'sideband_freq_default_mhz': DEFAULT_SIDEBAND_FREQ_MHZ,
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

    # ------------------------------------------------------------- commands
    def command(self, name, args):
        if name == 'play':
            self._playing.set()
            self._snapshot = None
            self._last_analysis = None  # overlay belongs to the frozen data
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
        if name == 'analyze_sidebands':
            return self._analyze_sidebands(args)
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

    # ---------------------------------------------------- sideband analysis
    def _analyze_sidebands(self, args):
        """6-Lorentzian sideband fit on the paused snapshot -> mode spacing,
        linewidths and NA. Same math as the offline script
        (pico_scope/mode_spacing_extraction_sidebands.py) via mode_analysis.
        Times are on the chart's axis: newest sample at 0, past negative.
        ValueErrors surface as HTTP 400 with the message in the box."""
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
        x0_guess, x1_guess = float(args['x0']), float(args['x1'])
        d_guess = abs(float(args['x_sb']) - x0_guess)
        if d_guess <= 0:
            raise ValueError('the sideband mark coincides with the 0th-order mark')
        f_sb_mhz = float(args.get('f_sb_mhz') or DEFAULT_SIDEBAND_FREQ_MHZ)

        try:
            params, errors = fit_six_lorentzians(
                x_fit, y_fit, x0_guess=x0_guess, x1_guess=x1_guess,
                d_guess=d_guess, region=(t_min, t_max))
        except RuntimeError as error:
            raise ValueError(f'fit did not converge: {error}')
        # NA interpolators: built by the (slow) cavity-design simulation on
        # the first call, cached after; unavailable -> MHz results only
        na_interp, _, na_error = get_na_interpolators()
        results = sideband_results(params['x0'], params['x1'], params['d'],
                                   params['s0'], params['s1'], f_sb_mhz,
                                   na_interp=na_interp)
        t_dense = np.linspace(t_min, t_max, 500)
        curve = six_lorentzian_model(
            t_dense, *(params[name] for name in SIX_LORENTZIAN_PARAMS))
        event = {'type': 'analysis_result',
                 'channel': channel,
                 'f_sb_mhz': f_sb_mhz,
                 'params': params,
                 'errors': errors,
                 'results': {**{key: (None if value is None else float(value))
                                for key, value in results.items()},
                             'na_error': na_error},
                 'curve': {'t': [round(float(v), 9) for v in t_dense],
                           'v': [round(float(v), 5) for v in curve]}}
        self._last_analysis = event
        self.emit(event)
        return {'ok': True, 'results': event['results']}
