"""PicoScope 4000A streaming-scope adapter: rolling chart-recorder view.

Wraps the pure device layer in pico_scope/ps4000a_scope.py. The smooth-plot
recipe (never a per-sample redraw anywhere):
  device thread -> numpy ring buffers (device layer)
  emitter thread here, ~20 Hz -> min/max envelope decimation of the visible
  window to <= MAX_POINTS per channel -> one 'scope_data' JSON event
  browser -> one uPlot setData() per event.
Mutually exclusive with PicoScope 7 (single owner) — open failure surfaces
as the standard 409 popup.

The trigger (see Trigger) also lives here rather than in the browser: it has
to see every sample to catch a short crossing, and the browser only ever gets
the decimated envelope.
"""

import sys
import threading
import time
from pathlib import Path

import numpy as np

# the device layer lives in pico_scope (alongside analysis)
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))
from pico_scope.ps4000a_scope import CHANNEL_NAMES, RANGES, PicoScope4000A  # noqa: E402
from pico_scope.mode_analysis import decimate  # noqa: E402

from .analyses.scope_pairs import PairsAnalysis  # noqa: E402
from .analyses.scope_sidebands import SidebandsAnalysis  # noqa: E402
from .base import DeviceAdapter  # noqa: E402

EMIT_INTERVAL_S = 0.05  # ~20 chunks/s to the browsers
MAX_POINTS = 1000  # per channel per chunk (500 min/max pairs)
WINDOW_CHOICES_S = (0.1, 1.0, 10.0, 60.0)
RATE_CHOICES_HZ = (100.0, 1000.0, 10_000.0, 100_000.0)
# a channel counts as clipped once a sample reaches this fraction of its
# range: the ADC saturates at full scale, a hair short of it after rounding
CLIP_FRACTION = 0.999


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


class Trigger:
    """Rising-edge trigger that freezes the view on each crossing.

    A crossing of `level_v` upwards on `channel` (None = off) is shown as a
    static window with the crossing at its middle - so it can only be shown
    once half a window more has been acquired. After a trigger the next
    crossing is ignored for a whole window: half while waiting for the frame
    to fill, half more after it is shown. Consecutive frames therefore never
    overlap, and a second crossing soon after the first cannot shift the view
    by part of a window. When a whole window passes with no crossing at all
    after a frame appeared, the view rolls live again (like a scope's auto
    mode), and the next crossing triggers.

    All indices are absolute sample numbers of the streamed rings
    (PicoScope4000A.samples_written / read_span); a stream restart or a new
    window length makes them meaningless, and starts the trigger over.
    """

    def __init__(self):
        self.channel = None
        self.level_v = 0.0
        self.reset()

    def reset(self):
        self.generation = None      # scope.stream_generation scanned under
        self.n_window = None
        self.scanned_to = None      # absolute index scanned up to
        self.armed_from = 0         # crossings before this are held off
        self.last_crossing = None   # any crossing, held off or not
        self.pending = None         # a trigger whose frame is still filling
        self.frame = None           # the frozen frame: {'dt', 'channels'}
        self.frame_event = None     # ... and its scope_data event, cached
        self.shown_at = -1          # written count when that frame appeared

    def settings(self):
        return {'channel': self.channel, 'level_v': self.level_v}

    def update(self, scope, window_s, make_event):
        """Scan what was acquired since the last call; returns the frozen
        frame's event while one is shown, None while the view should roll."""
        written = scope.samples_written(self.channel)
        if written is None:     # the trigger channel is switched off
            self.reset()
            return None
        dt = 1.0 / scope.sample_rate_hz
        n_window = max(int(round(window_s / dt)), 2)
        n_half = n_window // 2
        if (self.generation, self.n_window) != (scope.stream_generation,
                                                n_window):
            self.reset()
            self.generation, self.n_window = scope.stream_generation, n_window
            self.scanned_to = written

        # one sample of overlap with the last scan, so a crossing between
        # two scans is not missed; never more than a window back
        start = max(self.scanned_to - 1, written - n_window - 1, 0)
        samples = scope.read_span(self.channel, start, written)
        self.scanned_to = written
        if samples is not None and len(samples) > 1:
            above = samples >= scope.to_adc(self.channel, self.level_v)
            crossings = np.flatnonzero(~above[:-1] & above[1:]) + 1 + start
            if len(crossings):
                self.last_crossing = int(crossings[-1])
                eligible = crossings[crossings >= self.armed_from]
                if self.pending is None and len(eligible):
                    self.pending = int(eligible[0])
                    self.armed_from = self.pending + n_window

        if self.pending is not None and written >= self.pending + n_half:
            first = self.pending - n_half
            channels = {}
            for name in scope.channels:
                if scope.samples_written(name) is None:
                    continue
                span = scope.read_span(name, first, first + n_window)
                if span is None:    # this channel's ring lags a block behind
                    break
                channels[name] = span
            else:
                self.frame = {'dt': dt, 'channels': channels}
                self.frame_event = make_event(self.frame)
                self.shown_at = written
                self.pending = None

        # quiet for a whole window - counted from the newest frame appearing,
        # so each frozen frame stays up at least that long - then roll
        quiet_since = max(self.last_crossing if self.last_crossing is not None
                          else -n_window - 1, self.shown_at)
        if self.pending is None and written - quiet_since > n_window:
            self.frame = self.frame_event = None
        return self.frame_event


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
        self.trigger = Trigger()
        self._trigger_lock = threading.Lock()  # the emitter vs set_trigger
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
            'commands': ['play', 'pause', 'set_setting', 'set_channel',
                         'view_region', 'set_trigger']
                        + [name for analysis in self.analyses
                           for name in analysis.COMMANDS],
            'playing': self._playing.is_set(),
            'trigger': self.trigger.settings(),
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
                'trigger': self.trigger.settings(),
                'channels': {name: dict(config) for name, config
                             in self.scope.channels.items()}}

    def restore_settings(self, snapshot):
        # only touch what actually differs: every channel/rate change is a
        # stop-reconfigure-restart of the streaming (audible relay clicks).
        # Channels being enabled go first: the scope opens with only A on and
        # refuses to have none, so disabling A before enabling (say) D would
        # fail - and that used to abandon the whole restore, bringing the
        # scope back on A at every start.
        saved_channels = sorted((snapshot.get('channels') or {}).items(),
                                key=lambda item: not item[1].get('enabled'))
        for name, saved in saved_channels:
            current = self.scope.channels.get(name)
            if current is None or all(saved.get(key) == current.get(key)
                                      for key in current):
                continue
            try:
                self.scope.configure_channel(name,
                                             enabled=saved.get('enabled'),
                                             coupling=saved.get('coupling'),
                                             range_v=saved.get('range_v'))
            except Exception:
                pass  # one stale channel must not cost the rest of the restore
        rate = snapshot.get('sample_rate_hz')
        if rate and rate != self.scope.sample_rate_hz:
            self.scope.set_sample_rate(float(rate))
        window = snapshot.get('window_s')
        if window:
            self.window_s = float(np.clip(window, 0.01, 60.0))
        trigger = snapshot.get('trigger')
        if trigger:
            self.set_trigger(trigger.get('channel'), trigger.get('level_v', 0.0))

    # ------------------------------------------------------------- commands
    def command(self, name, args):
        for analysis in self.analyses:
            result = analysis.command(name, args)
            if result is not None:
                return result
        if name == 'play':
            with self._trigger_lock:
                self.trigger.reset()    # nothing from before the pause
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
            with self._trigger_lock:
                frozen = self.trigger.frame
            if frozen is not None:      # freeze the triggered frame on screen
                self._snapshot = frozen
            else:
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
        if name == 'set_trigger':
            return {'ok': True, 'trigger': self.set_trigger(
                args.get('channel'), args.get('level_v', self.trigger.level_v))}
        if name == 'view_region':
            return self.view_region(float(args['t_min']), float(args['t_max']))
        raise ValueError(f'unknown command {name!r}')

    def set_trigger(self, channel, level_v):
        """Turn the trigger on (channel 'A'..'D') or off (None), at level_v
        volts; broadcast so every viewer moves its trigger dot."""
        if channel not in (None, '', *CHANNEL_NAMES):
            raise ValueError(f'no channel {channel!r}')
        with self._trigger_lock:
            self.trigger.channel = channel or None
            self.trigger.level_v = float(level_v)
            self.trigger.reset()
            settings = self.trigger.settings()
        self.emit({'type': 'trigger', **settings})
        return settings

    def view_region(self, t_min, t_max):
        """Part of the paused snapshot at the resolution the box can show:
        the samples between t_min and t_max (chart time, newest sample at 0)
        envelope-decimated to <= MAX_POINTS per channel, in volts.

        For the box's x zoom. The chunk broadcast at pause spreads the whole
        window over MAX_POINTS, so zooming into it alone would only magnify
        the decimation; this goes back to the full-resolution samples. It is
        the asking viewer's view only - returned, never broadcast.
        """
        if self._playing.is_set() or self._snapshot is None:
            raise ValueError('pause the stream first — zoom detail comes from '
                             'the frozen snapshot')
        dt = self._snapshot['dt']
        channels = {}
        t_first = t_last = None
        for name, adc in self._snapshot['channels'].items():
            n = len(adc)
            if not n:
                continue
            first = max(0, int(np.floor((n - 1) + t_min / dt)))
            last = min(n - 1, int(np.ceil((n - 1) + t_max / dt)))
            if last <= first:
                continue
            if last + 1 - first > MAX_POINTS:
                # whole buckets only, as envelope() would trim them anyway -
                # trimmed here so t_first names the first sample kept
                buckets = MAX_POINTS // 2
                first = last + 1 - (last + 1 - first) // buckets * buckets
            decimated, _ = envelope(adc[first:last + 1], MAX_POINTS)
            channels[name] = [round(float(v), 5)
                              for v in self.scope.to_volts(name, decimated)]
            t_first = (first - (n - 1)) * dt
            t_last = (last - (n - 1)) * dt
        return {'t_first': t_first, 't_last': t_last, 'channels': channels}

    # -------------------------------------------------------- emitter thread
    def _emit_loop(self):
        while not self._stopping.is_set():
            tic = time.time()
            if self._playing.is_set():
                try:
                    self._emit_tick()
                except Exception as error:
                    self.emit({'type': 'error', 'message': str(error)})
            elapsed = time.time() - tic
            self._stopping.wait(max(EMIT_INTERVAL_S - elapsed, 0.005))

    def _emit_tick(self):
        with self._trigger_lock:
            if self.trigger.channel is None:
                frozen, state = None, 'off'
            else:
                frozen = self.trigger.update(
                    self.scope, self.window_s,
                    lambda frame: self._chunk_event(frame, 'triggered'))
                state = 'triggered' if frozen else 'auto'
        if frozen is not None:
            self.emit(frozen)   # the same frame again: a new viewer gets it too
        else:
            self._emit_chunk(trigger_state=state)

    def _emit_chunk(self, snapshot=None, trigger_state='off'):
        if snapshot is None:
            dt, window = self.scope.read_window(self.window_s)
            snapshot = {'dt': dt, 'channels': window}
        event = self._chunk_event(snapshot, trigger_state)
        if event is not None:
            self.emit(event)

    def _chunk_event(self, snapshot, trigger_state):
        """The scope_data event for a window of raw samples, or None when
        there is nothing in it. trigger_state is what the box reports: 'off',
        'auto' (rolling, waiting for a crossing) or 'triggered' (frozen)."""
        dt, window = snapshot['dt'], snapshot['channels']
        channels = {}
        clipped = []
        n_max = 0
        for name, adc in window.items():
            if not len(adc):
                continue
            decimated, stride = envelope(adc, MAX_POINTS)
            volts = self.scope.to_volts(name, decimated)
            channels[name] = [round(float(v), 5) for v in volts]
            n_max = max(n_max, len(adc))
            # the envelope keeps every bucket's min and max, so a sample
            # pinned at the ADC rail anywhere in the window shows up here
            if np.abs(volts).max() >= CLIP_FRACTION * \
                    self.scope.channels[name]['range_v']:
                clipped.append(name)
        if not channels:
            return None
        return {'type': 'scope_data',
                'window_s': self.window_s,
                'span_s': n_max * dt,  # actual data span (fills up after start)
                'trigger_state': trigger_state,
                'clipped': clipped,  # channels that hit their range
                'channels': channels}

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

