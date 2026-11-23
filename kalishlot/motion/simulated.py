"""A simulated motor axis with realistic imperfections, for the laptop.

The *true* position differs from the commanded one by:
- backlash / hysteresis: a play (dead-band) of width `backlash`; the true
  position lags the command by backlash/2 against the last direction of motion.
  (A piezo's hysteresis loop is approximated by the same play operator.)
- random landing error: every finished move adds N(0, repeatability_sigma).
  This is the inherent, non-removable part of the repeatability error: the
  same target from the same side still lands in a slightly different place.
- drift: `drift_per_s` units per second of wall (or injected) time.
- step asymmetry: `backward_scale` != 1 makes every backward move cover that
  fraction of a forward move of the same size (an open-loop Picomotor: the
  8742 manual warns forward and backward steps differ; the lab saw ~20-30%).
  The true position then wanders off the command with every round trip, so
  returning to a counter value does not return to the same place.
- step jitter: `step_jitter` = relative std of a move's length (open loop):
  every move covers delta * (1 + N(0, step_jitter)), so the landing error
  grows with the move, unlike repeatability_sigma (a fixed error per move).

What read_position() reports depends on the device type being imitated:
- closed_loop=True: a real sensor, true position + N(0, sensor_noise);
  the servo also removes backlash, so only the random error remains.
- has_readback=True, closed_loop=False: the last command echoed back (like an
  open-loop piezo driver reporting its output voltage).
- has_readback=False: None (stick-slip / inertia drives, steppers w/o encoder).
"""

import time

import numpy as np

from motion.axis import Axis, AxisInfo


class SimulatedAxis(Axis):
    def __init__(self, name, units='V', kind='voltage', start=0.0, has_readback=True,
                 closed_loop=False, backlash=0.0, repeatability_sigma=0.0,
                 drift_per_s=0.0, sensor_noise=0.0, can_home=False, seed=None,
                 now=time.monotonic, known_on_open=True, backward_scale=1.0, step_jitter=0.0):
        self.info = AxisInfo(name, units, kind, has_readback or closed_loop, closed_loop, can_home)
        self.backlash = float(backlash)
        self.repeatability_sigma = float(repeatability_sigma)
        self.drift_per_s = float(drift_per_s)
        self.sensor_noise = float(sensor_noise)
        self.known_on_open = known_on_open
        self.backward_scale = float(backward_scale)
        self.step_jitter = float(step_jitter)
        self._rng = np.random.default_rng(seed)
        self._now = now
        self._commanded = float(start)
        self._true = float(start)
        self._last_dir = 0
        self._t_last = now()
        self.moves = 0
        self.last_move_t = -1e9         # for the simulator's vibration model
        self.is_open = False

    def open(self):
        self.is_open = True

    def close(self):
        self.is_open = False

    def _apply_drift(self):
        t = self._now()
        self._true += self.drift_per_s * (t - self._t_last)
        self._t_last = t

    @property
    def true_position(self):
        """Where the axis really is (only a simulator can know this)."""
        self._apply_drift()
        return self._true

    def read_position(self):
        if self.info.closed_loop:
            return self.true_position + float(self._rng.normal(0, self.sensor_noise)) \
                if self.sensor_noise else self.true_position
        if self.info.has_readback:
            return self._commanded
        return self._commanded if (self.known_on_open and self.moves == 0) else None

    def move_to(self, target):
        if not self.is_open:
            raise RuntimeError(f'{self.info.name} is not open')
        self._apply_drift()
        target = float(target)
        direction = int(np.sign(target - self._commanded))
        if direction:
            self._last_dir = direction
        delta = target - self._commanded
        self._commanded = target
        if self.info.closed_loop:
            self._true = target
        elif self.backward_scale != 1.0 or self.step_jitter:   # open-loop steps: the truth wanders off
            covered = delta if delta >= 0 else delta * self.backward_scale
            if self.step_jitter:
                covered *= 1.0 + float(self._rng.normal(0, self.step_jitter))
            self._true += covered
        else:
            self._true = target - self._last_dir * self.backlash / 2
        if self.repeatability_sigma:
            self._true += float(self._rng.normal(0, self.repeatability_sigma))
        self.moves += 1
        self.last_move_t = self._now()

    def stop(self):
        pass

    def home(self):
        if not self.info.can_home:
            raise NotImplementedError(f'{self.info.name} cannot home')
        self._commanded = self._true = 0.0
        self._last_dir = 0
