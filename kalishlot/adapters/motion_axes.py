"""Motorized axes adapter: one box per driver group of motion/axes_config.py.

Control-only device (no frames), modelled on rigol_dg.py. The box never talks
to a motor directly: every axis is a motion.SafeAxis, so range, step, slew,
freeze, E-stop and approach rules apply to the UI, to scripts sending commands
through loan_client.device_command, and to the alignment loop alike. A refused
move raises SafetyError (a ValueError), which the server returns as HTTP 400.

Address = a driver key of AXES ('simulated', later e.g. 'thorlabs_mdt693b');
the box holds every configured axis with that driver.

Commands (args):
    move_to      {axis, target, wait=False}   ramped absolute move
    move_by      {axis, delta, wait=False}    ramped relative move
    estop        {axis=None}                  stop + hold; all axes if none given
    reset_estop  {axis=None}
    freeze       {axis, frozen}               run-time freeze (config freeze is final)
    set_known_position {axis, value}          declare position without moving
    refresh      {}
With wait=False the move runs on a worker thread and the call returns at once;
with wait=True it returns the MoveResult when the axis has settled
mechanically (for scripts: the HTTP timeout of loan_client is 15 s).

Events: {'type': 'axis', 'state': SafeAxis.describe(), 'moving': bool}
        {'type': 'error', 'message': str}
Opening reads positions and never moves; closing E-stops any running move
(stop and hold) and never moves either.
"""

import sys
import threading
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # kalishlot/: motion package
from motion import build_axis  # noqa: E402
from motion.axes_config import AXES, TIERS  # noqa: E402
from motion.safety import SafetyError  # noqa: E402

from .base import DeviceAdapter  # noqa: E402

DRIVER_LABELS = {'simulated': 'Simulated axes'}


class MotionAxesAdapter(DeviceAdapter):
    type_name = 'motion_axes'
    display_name = 'Motorized axes'

    @staticmethod
    def list_available():
        groups = {}
        for name, entry in AXES.items():
            groups.setdefault(entry['driver'], []).append(name)
        return [{'address': driver,
                 'label': f"{DRIVER_LABELS.get(driver, driver)} ({', '.join(names)})"}
                for driver, names in groups.items()]

    def __init__(self, address, config=None):
        super().__init__(address)
        self.config = AXES if config is None else config
        names = [n for n, e in self.config.items() if e['driver'] == self.address]
        if not names:
            raise ValueError(f'no configured axes with driver {self.address!r}')
        self.axes = {n: build_axis(n, self.config) for n in names}
        self._workers = {}               # axis name -> running move thread
        self._lock = threading.Lock()

    # ------------------------------------------------------------ lifecycle
    def open(self):
        for axis in self.axes.values():
            axis.open()

    def close(self):
        for name, axis in self.axes.items():
            if self._moving(name):
                axis.estop('box closed')
        for worker in list(self._workers.values()):
            worker.join(timeout=5)
        for axis in self.axes.values():
            axis.close()

    def describe(self):
        return {'type': self.type_name,
                'label': DRIVER_LABELS.get(self.address, self.address),
                'commands': ['move_to', 'move_by', 'estop', 'reset_estop', 'freeze',
                             'set_known_position', 'refresh'],
                'axes': [self._state(n) for n in self.axes],
                'tiers': [[n for n in tier if n in self.axes] for tier in TIERS]}

    # ------------------------------------------------------------- helpers
    def _axis(self, args):
        name = args.get('axis')
        if name not in self.axes:
            raise ValueError(f'no axis {name!r} in this box')
        return name, self.axes[name]

    def _moving(self, name):
        worker = self._workers.get(name)
        return worker is not None and worker.is_alive()

    def _state(self, name):
        return {**self.axes[name].describe(), 'moving': self._moving(name)}

    def _emit_axis(self, name):
        self.emit({'type': 'axis', 'state': self._state(name)})

    def _run_move(self, name, fn, value, who):
        try:
            fn(value, who=who)
        except Exception as error:          # SafetyError mid-ramp (E-stop), driver errors
            self.emit({'type': 'error', 'message': f'{name}: {error}'})
        finally:
            with self._lock:
                self._workers.pop(name, None)
            self._emit_axis(name)

    def _move(self, args, absolute):
        name, axis = self._axis(args)
        value = float(args['target'] if absolute else args['delta'])
        fn = axis.move_to if absolute else axis.move_by
        who = str(args.get('who', 'kalishlot'))
        with self._lock:
            if self._moving(name):
                raise ValueError(f'{name} is already moving')
            # refuse up front (HTTP 400) rather than failing quietly on the worker;
            # an unknown position is refused by check_target before the target matters
            target = value if absolute or not axis.position_known else axis.commanded + value
            axis.check_target(target)
            if not args.get('wait'):
                worker = threading.Thread(target=self._run_move, args=(name, fn, value, who),
                                          daemon=True, name=f'move {name}')
                self._workers[name] = worker
                worker.start()
                self._emit_axis(name)
                return {'ok': True, 'started': True}
        try:
            result = fn(value, who=who)
        finally:
            self._emit_axis(name)
        return {'ok': True, 'result': asdict(result)}

    # ------------------------------------------------------------- commands
    def command(self, name, args):
        args = args or {}
        if name == 'move_to':
            return self._move(args, absolute=True)
        if name == 'move_by':
            return self._move(args, absolute=False)
        if name in ('estop', 'reset_estop'):
            targets = [self._axis(args)[0]] if args.get('axis') else list(self.axes)
            for n in targets:
                if name == 'estop':
                    self.axes[n].estop(who=str(args.get('who', 'kalishlot')))
                else:
                    self.axes[n].reset_estop(who=str(args.get('who', 'kalishlot')))
                self._emit_axis(n)
            return {'ok': True}
        if name == 'freeze':
            n, axis = self._axis(args)
            axis.set_frozen(bool(args['frozen']), who=str(args.get('who', 'kalishlot')))
            self._emit_axis(n)
            return {'ok': True}
        if name == 'set_known_position':
            n, axis = self._axis(args)
            axis.set_known_position(float(args['value']), who=str(args.get('who', 'kalishlot')))
            self._emit_axis(n)
            return {'ok': True}
        if name == 'refresh':
            for n in self.axes:
                self._emit_axis(n)
            return {'ok': True}
        raise ValueError(f'unknown command {name!r}')


__all__ = ['MotionAxesAdapter', 'SafetyError']
