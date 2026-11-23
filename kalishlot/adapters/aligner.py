"""Aligner box: automatic alignment from kalishlot's own devices.

A virtual device (no hardware), like synced_pipeline and live_measure. The
user describes an alignment as a table of signals with goals plus the
controls it may move (alignment/recipe.py); the box runs it with
alignment/loop.py's AlignmentLoop, in this process:
- signals come from the open boxes through alignment/signals.py (so far the
  cameras' centre of mass), measured in-process, newest frames only;
- controls are the SafeAxis objects of open motion-axes boxes, which the run
  claims (MotionAxesAdapter.claim): every safety rule binds the loop, the jog
  box refuses other moves meanwhile, and its E-STOP still stops everything.

The setup (levels, rows, controls, fields) persists with the box's settings.
One level for now; several levels = the nested loop, later.

Commands (args):
    set_setup {setup}        {'levels': [{'rows': [...], 'controls': [...]}], 'fields': {...}}
    refresh  {}              re-read what can be measured and moved ('catalog' event)
    dry_run  {}              measure once; per row value, target, miss, ok. Nothing moves.
    start    {}              run level 1 (refused while running)
    pause / resume {}        hold before the next measurement (axes stand still)
    stop     {on_abort}      end after the current measurement; then 'best' (default),
                             'start' or 'stay'
Fields: delay_s (wait after a move before frames count), samples (CoM results per
camera per measurement), max_evaluations, on_abort.
Events: catalog, setup, status {state}, evaluation {index, positions, rows, kind?},
        log {line}, result {summary, rows}, error {message}
"""

import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # kalishlot/: alignment, motion
from alignment import recipe  # noqa: E402
from alignment.loop import AlignmentLoop  # noqa: E402
from alignment.signals import ComSignals, SignalSet, resolve_value  # noqa: E402

from .base import DeviceAdapter  # noqa: E402

RUNS_DIR = Path(__file__).resolve().parents[1] / 'alignment' / 'runs'
DEFAULT_FIELDS = {'delay_s': 0.3, 'samples': 10, 'max_evaluations': 80, 'on_abort': 'best'}
ON_ABORT = ('best', 'start', 'stay')
HISTORY_KEPT = 300          # evaluations kept for a box that re-attaches mid-run
WHO = 'aligner'


def _num(v):
    return None if v is None or v != v else round(float(v), 6)


class AlignerAdapter(DeviceAdapter):
    type_name = 'aligner'
    display_name = 'Aligner'
    VIRTUAL = True

    registry = None     # set by server.py: () -> (devices, loans)
    activity = None     # set by server.py: note_activity (a run keeps the idle watchdog quiet)

    @staticmethod
    def list_available():
        return [{'address': 'main', 'label': 'Aligner (automatic alignment)'}]

    def __init__(self, address):
        super().__init__(address)
        self.setup = {'levels': [{'rows': [], 'controls': []}], 'fields': dict(DEFAULT_FIELDS)}
        self.state = 'idle'                 # idle | running | paused | stopping
        self.history = []
        self.result = None
        self._lock = threading.Lock()
        self._loop = None
        self._worker = None
        self._resume = threading.Event()
        self._resume.set()

    # ------------------------------------------------------------ lifecycle
    def open(self):
        pass

    def close(self):
        loop = self._loop
        if loop is not None:
            self._resume.set()
            loop.on_abort = 'best'
            loop.stop()
        if self._worker is not None:
            self._worker.join(timeout=30)

    def settings_snapshot(self):
        return {'setup': self.setup}

    def restore_settings(self, snapshot):
        setup = (snapshot or {}).get('setup')
        if isinstance(setup, dict) and isinstance(setup.get('levels'), list) and setup['levels']:
            fields = dict(DEFAULT_FIELDS, **(setup.get('fields') or {}))
            self.setup = {'levels': setup['levels'], 'fields': fields}

    # ------------------------------------------------------------- devices
    def _devices(self):
        return dict(self.registry()[0]) if self.registry is not None else {}

    def _motion_boxes(self, devices=None):
        devices = self._devices() if devices is None else devices
        return {i: a for i, a in devices.items() if getattr(a, 'type_name', None) == 'motion_axes'}

    def _axis_index(self, devices=None):
        """axis name -> (box id, box adapter)."""
        out = {}
        for box_id, box in self._motion_boxes(devices).items():
            for name in box.axes:
                out[name] = (box_id, box)
        return out

    def catalog(self):
        devices = self._devices()
        controls = []
        for box_id, box in self._motion_boxes(devices).items():
            for name, axis in box.axes.items():
                d = axis.describe()
                controls.append({'axis': name, 'box': box_id, 'units': d.get('units'),
                                 'position': d.get('position'), 'frozen': d.get('frozen'),
                                 'max_step': axis.limits.max_step, 'min': axis.limits.min,
                                 'max': axis.limits.max, 'owner': box.owner})
        markers = {}
        for device_id, adapter in devices.items():
            if hasattr(adapter, 'com_samples'):
                ms = adapter.describe().get('markers') or []
                markers[device_id] = [{'id': m.get('id'), 'label': m.get('label') or m.get('id'),
                                       'x': m.get('x'), 'y': m.get('y'),
                                       'visible': m.get('visible', True)} for m in ms]
        signals = SignalSet(lambda: devices).available()
        return {'signals': signals, 'controls': controls, 'markers': markers,
                'goals': recipe.GOALS}

    def describe(self):
        return {'type': self.type_name, 'label': self.display_name,
                'commands': ['set_setup', 'refresh', 'dry_run', 'start', 'pause', 'resume', 'stop'],
                'setup': self.setup, 'state': self.state, 'catalog': self.catalog(),
                'history': self.history[-HISTORY_KEPT:], 'result': self.result}

    # ------------------------------------------------------------ helpers
    def _set_state(self, state):
        self.state = state
        self.emit({'type': 'status', 'state': state})

    def _fields(self):
        f = dict(DEFAULT_FIELDS, **(self.setup.get('fields') or {}))
        f['delay_s'] = max(float(f['delay_s']), 0.0)
        f['samples'] = max(int(f['samples']), 1)
        f['max_evaluations'] = max(int(f['max_evaluations']), 1)
        if f['on_abort'] not in ON_ABORT:
            f['on_abort'] = 'best'
        return f

    def _signal_set(self, fields):
        return SignalSet(self._devices, [ComSignals(samples=fields['samples'])])

    def _level(self):
        levels = self.setup.get('levels') or []
        if not levels:
            raise ValueError('no alignment level set up')
        if len(levels) > 1:
            raise ValueError('nested alignments (several levels) are not available yet')
        level = levels[0]
        recipe.validate(level)
        return level

    def _resolve(self, level, signals, fields):
        """(row values, the start measurement) -- values {row index: float}."""
        ids = recipe.signals_of(level)
        signals.prepare(ids)
        start = signals.measure(ids, fields['delay_s'])
        devices = self._devices()
        values = {}
        for i, r in enumerate(level['rows']):
            spec = r.get('value') or {}
            if r['goal'] in ('max', 'min') and spec.get('kind') not in ('number',):
                continue                     # no "good enough" threshold
            values[i] = resolve_value(spec, r['signal'], devices, start.metrics)
        return values, start

    @staticmethod
    def _rows_report(level, values, cm):
        out = []
        for i, r in enumerate(level['rows']):
            m = cm.metrics.get(r['signal'])
            value = m.value if m is not None and m.valid else None
            target = values.get(i)
            row = {'signal': r['signal'], 'goal': r['goal'], 'target': _num(target),
                   'value': _num(value), 'spread': _num(m.spread) if m is not None and m.valid else None,
                   'reason': '' if value is not None else (m.reason if m is not None else 'missing'),
                   'miss': None, 'ok': None}
            if value is not None and target is not None:
                if r['goal'] == 'goto':
                    row['miss'] = _num(value - target)
                    row['ok'] = abs(value - target) <= float(r['tolerance'])
                elif r['goal'] == 'ge':
                    row['ok'] = value >= target
                elif r['goal'] == 'le':
                    row['ok'] = value <= target
            out.append(row)
        return out

    # ------------------------------------------------------------- commands
    def command(self, name, args):
        args = args or {}
        if name == 'refresh':
            cat = self.catalog()
            self.emit({'type': 'catalog', 'catalog': cat})
            return {'ok': True, 'catalog': cat}
        if name == 'set_setup':
            if self.state != 'idle':
                raise ValueError('the setup cannot change during a run')
            setup = args.get('setup')
            if not isinstance(setup, dict) or not isinstance(setup.get('levels'), list):
                raise ValueError('setup must be {"levels": [...], "fields": {...}}')
            self.setup = {'levels': setup['levels'] or [{'rows': [], 'controls': []}],
                          'fields': dict(DEFAULT_FIELDS, **(setup.get('fields') or {}))}
            self.emit({'type': 'setup', 'setup': self.setup})
            return {'ok': True}
        if name == 'dry_run':
            if self.state != 'idle':
                raise ValueError('a run is in progress')
            fields = self._fields()
            level = self.setup['levels'][0]
            if not level.get('rows'):
                raise ValueError('add at least one signal row')
            for i, r in enumerate(level['rows'], 1):
                if not r.get('signal'):
                    raise ValueError(f'row {i}: choose a signal')
            values, start = self._resolve(level, self._signal_set(fields), fields)
            rows = self._rows_report(level, values, start)
            self.emit({'type': 'dry_run', 'rows': rows})
            return {'ok': True, 'rows': rows}
        if name == 'start':
            return self._start()
        if name == 'pause':
            if self.state == 'running':
                self._resume.clear()
                self._set_state('paused')
            return {'ok': True}
        if name == 'resume':
            if self.state == 'paused':
                self._resume.set()
                self._set_state('running')
            return {'ok': True}
        if name == 'stop':
            loop = self._loop
            if loop is None:
                return {'ok': True}
            on_abort = args.get('on_abort') or self._fields()['on_abort']
            if on_abort not in ON_ABORT:
                raise ValueError(f'on_abort must be one of {ON_ABORT}')
            loop.on_abort = on_abort
            loop.stop()
            self._set_state('stopping')
            self._resume.set()
            return {'ok': True}
        raise ValueError(f'unknown command {name!r}')

    # ------------------------------------------------------------------ run
    def _start(self):
        with self._lock:
            if self.state != 'idle':
                raise ValueError('a run is already in progress')
            level = self._level()
            fields = self._fields()
            index = self._axis_index()
            controls = [c for c in level['controls'] if c.get('axis')]
            missing = [c['axis'] for c in controls if c['axis'] not in index]
            if missing:
                raise ValueError(f'controls not open in any motion-axes box: {missing}')
            boxes = {index[c['axis']][0]: index[c['axis']][1] for c in controls}
            stopped = [c['axis'] for c in controls if index[c['axis']][1].axes[c['axis']].estopped]
            if stopped:
                raise ValueError(f'E-stopped: {", ".join(stopped)} (reset in the motion box first)')
            claimed = []
            try:
                for box in boxes.values():
                    box.claim(WHO)
                    claimed.append(box)
                signals = self._signal_set(fields)
                values, start = self._resolve(level, signals, fields)
            except Exception:
                for box in claimed:
                    box.release(WHO)
                raise
            axes = {c['axis']: index[c['axis']][1].axes[c['axis']] for c in controls}
            steps = {c['axis']: float(c['step']) for c in controls if c.get('step')}
            self.history, self.result = [], None
            self._resume.set()
            self._set_state('running')
            self._worker = threading.Thread(
                target=self._run, args=(level, fields, signals, values, axes, steps, list(boxes.values())),
                daemon=True, name='aligner run')
            self._worker.start()
        return {'ok': True, 'started': True}

    def _run(self, level, fields, signals, values, axes, steps, boxes):
        ids = recipe.signals_of(level)
        names = list(axes)
        count = [0]

        def measure():
            self._resume.wait()
            # an E-stopped axis refuses every move, which the loop would take for the edge
            # of its range and carry on: end the run instead
            stopped = [n for n, a in axes.items() if a.estopped]
            if stopped:
                raise RuntimeError(f'E-stop on {", ".join(stopped)}')
            if self.activity is not None:
                self.activity()
            for box in boxes:
                box.notify([n for n in names if n in box.axes])
            cm = signals.measure(ids, fields['delay_s'])
            count[0] += 1
            event = {'type': 'evaluation', 'index': count[0],
                     'positions': {n: _num(getattr(a, 'commanded', None)) for n, a in axes.items()},
                     'rows': self._rows_report(level, values, cm), 'time': time.time()}
            self.history.append(event)
            del self.history[:-HISTORY_KEPT]
            self.emit(event)
            return cm

        def log(*parts, **_):
            self.emit({'type': 'log', 'line': ' '.join(str(p) for p in parts)})

        summary, error = None, None
        try:
            objective = recipe.build_objective(level, values, label='aligner')
            loop = AlignmentLoop(axes, measure, objective, tiers=[names], steps=steps or None,
                                 max_evaluations=fields['max_evaluations'],
                                 on_abort=fields['on_abort'],
                                 log_dir=RUNS_DIR / time.strftime('%Y%m%d-%H%M%S'),
                                 who=WHO, print_fn=log)
            self._loop = loop
            summary = loop.run()
        except Exception as e:                  # E-stop, driver error, signal lost
            error = str(e)
            self.emit({'type': 'error', 'message': f'run ended: {error}'})
        finally:
            self._loop = None
            for box in boxes:
                box.release(WHO)
            last = self.history[-1]['rows'] if self.history else []
            self.result = {'summary': {k: v for k, v in (summary or {}).items()
                                       if k in ('best', 'evaluations', 'target_reached',
                                                'aborted', 'seconds', 'start')},
                           'error': error, 'rows': last}
            self.emit({'type': 'result', **self.result})
            self._resume.set()
            self._set_state('idle')
