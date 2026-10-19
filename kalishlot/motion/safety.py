"""SafeAxis: the only way anything moves a motor.

Safety by construction (kalishlot/MECHANICAL_CONTROL.md, section 1):
- Out-of-range or non-finite targets are REJECTED (SafetyError, a ValueError),
  never clamped, so a caller always knows where it asked to go.
- Legal but large moves are RAMPED: split into steps of at most `max_step`,
  sent no faster than `max_slew` units/s.
- A frozen axis refuses every move (this is how a DoF is banned; the
  optimizer goes through here too, so it cannot bypass it). Frozen in the
  config = hard; set_frozen() = run-time toggle that cannot undo the config.
- Position must be known before the first move (read on open, homed, or
  declared by a person with set_known_position).
- E-stop stops motion and holds; further moves are refused until reset.
- close() never moves (no zeroing a piezo on exit: that would be a jump).
- Optional anti-backlash approach: always arrive from the same side.
- Optional travel budget per session.
- Every move is logged: commanded, read-back, who asked, when.

Measurement timing: motion shakes the setup, so a capture must wait until the
axis is done AND its vibration settle delay has passed. `ready_at` holds that
time; `wait_until_settled()` sleeps until then.
"""

import json
import math
import threading
import time
from dataclasses import dataclass, asdict
from pathlib import Path


class SafetyError(ValueError):
    """A move was refused by a safety rule. Nothing was sent to the hardware."""


@dataclass(frozen=True)
class AxisLimits:
    min: float
    max: float
    max_step: float                     # largest single step sent to the driver
    max_slew: float                     # units per second (ramp speed)
    settle_s: float = 0.0               # mechanical settle after the last step
    vibration_settle_s: float = 0.5     # extra wait before a measurement may start
    repeatability_sigma: float = 0.0    # measured random landing error (units)
    approach: int = 0                   # +1: always arrive moving up, -1: moving down, 0: off
    approach_overshoot: float = 0.0     # how far past the target the approach starts
    travel_budget: float | None = None  # max total travel per session (units)
    frozen: bool = False

    def __post_init__(self):
        if not (math.isfinite(self.min) and math.isfinite(self.max) and self.min < self.max):
            raise ValueError(f'bad range [{self.min}, {self.max}]')
        if self.max_step <= 0 or self.max_slew <= 0:
            raise ValueError('max_step and max_slew must be positive')
        if self.approach not in (-1, 0, 1):
            raise ValueError('approach must be -1, 0 or +1')


@dataclass
class MoveResult:
    target: float
    commanded: float            # where the axis was last commanded (== target on success)
    readback: float | None      # sensor/driver position after the move, if any
    travel: float               # path length including approach overshoot
    steps: int
    duration_s: float


class MoveLog:
    """Append-only move log, in memory and optionally as JSON lines on disk."""

    def __init__(self, path=None):
        self.path = Path(path) if path else None
        self.entries = []
        self._lock = threading.Lock()

    def append(self, **entry):
        entry.setdefault('time', time.time())
        with self._lock:
            self.entries.append(entry)
            if self.path:
                with self.path.open('a', encoding='utf-8') as fh:
                    fh.write(json.dumps(entry, default=str) + '\n')


class SafeAxis:
    def __init__(self, raw, limits, log=None, sleep=time.sleep, now=time.monotonic):
        self._raw = raw
        self.limits = limits
        self.log = log if log is not None else MoveLog()
        self._sleep = sleep
        self._now = now
        self._lock = threading.RLock()
        self._estop = threading.Event()
        self._commanded = None          # last commanded position (None = unknown)
        self.travel_used = 0.0
        self.ready_at = 0.0             # monotonic time after which a measurement may start
        self.is_open = False
        self._runtime_frozen = False    # toggled from the UI; limits.frozen is the hard one

    # ------------------------------------------------------------- identity
    @property
    def info(self):
        return self._raw.info

    @property
    def name(self):
        return self._raw.info.name

    @property
    def units(self):
        return self._raw.info.units

    @property
    def position_known(self):
        return self._commanded is not None

    @property
    def position(self):
        """Best current position: the sensor reading for closed-loop axes,
        else the last commanded position. None if unknown."""
        if self.info.closed_loop:
            p = self._raw.read_position()
            if p is not None:
                return p
        return self._commanded

    @property
    def commanded(self):
        return self._commanded

    @property
    def frozen(self):
        return self.limits.frozen or self._runtime_frozen

    def set_frozen(self, frozen, who):
        """Freeze/unfreeze at run time. An axis frozen in its config stays frozen."""
        if not frozen and self.limits.frozen:
            raise SafetyError(f'{self.name} is frozen in its config; edit the config to unfreeze')
        self._runtime_frozen = bool(frozen)
        self.log.append(axis=self.name, event='freeze' if frozen else 'unfreeze', who=who)

    @property
    def estopped(self):
        return self._estop.is_set()

    # ------------------------------------------------------------ lifecycle
    def open(self):
        """Connect and read the position. Never writes or moves."""
        with self._lock:
            self._raw.open()
            self.is_open = True
            self._commanded = self._raw.read_position()
            self.log.append(axis=self.name, event='open', position=self._commanded)

    def close(self):
        """Disconnect without moving (holds wherever it is)."""
        with self._lock:
            self._raw.close()
            self.is_open = False
            self.log.append(axis=self.name, event='close', position=self._commanded)

    def set_known_position(self, value, who):
        """Declare the current position (e.g. read off a micrometer by a person,
        or an open-loop axis whose driver cannot report it). Does not move."""
        value = float(value)
        if not (math.isfinite(value) and self.limits.min <= value <= self.limits.max):
            raise SafetyError(f'{self.name}: declared position {value} is outside '
                              f'[{self.limits.min}, {self.limits.max}] {self.units}')
        with self._lock:
            self._commanded = value
            self.log.append(axis=self.name, event='set_known_position', position=value, who=who)

    def home(self, who):
        if self.frozen:
            raise SafetyError(f'{self.name} is frozen')
        self._check_estop()
        with self._lock:
            self._raw.home()
            self._commanded = self._raw.read_position()
            self.log.append(axis=self.name, event='home', position=self._commanded, who=who)

    # ---------------------------------------------------------------- E-stop
    def estop(self, who='estop'):
        """Stop motion now and hold. Moves are refused until reset_estop()."""
        self._estop.set()
        try:
            self._raw.stop()
        finally:
            self.log.append(axis=self.name, event='estop', who=who)

    def reset_estop(self, who):
        self._estop.clear()
        self.log.append(axis=self.name, event='reset_estop', who=who)

    def _check_estop(self):
        if self._estop.is_set():
            raise SafetyError(f'{self.name}: E-stop is active')

    # ---------------------------------------------------------------- checks
    def check_target(self, target):
        """Raise SafetyError if a move to `target` would be refused. Sends nothing."""
        lim = self.limits
        if self.frozen:
            raise SafetyError(f'{self.name} is frozen')
        self._check_estop()
        if not self.is_open:
            raise SafetyError(f'{self.name} is not open')
        if self._commanded is None:
            raise SafetyError(f'{self.name}: position unknown (home it or declare it first)')
        try:
            target = float(target)
        except (TypeError, ValueError):
            raise SafetyError(f'{self.name}: target {target!r} is not a number') from None
        if not math.isfinite(target):
            raise SafetyError(f'{self.name}: target {target} is not finite')
        if not lim.min <= target <= lim.max:
            raise SafetyError(f'{self.name}: target {target:g} {self.units} is outside '
                              f'[{lim.min:g}, {lim.max:g}]')
        path = [self._commanded] + self._waypoints(self._commanded, target)
        travel = sum(abs(b - a) for a, b in zip(path, path[1:]))
        if lim.travel_budget is not None and self.travel_used + travel > lim.travel_budget:
            raise SafetyError(f'{self.name}: travel budget exhausted '
                              f'({self.travel_used:g} of {lim.travel_budget:g} {self.units} used)')
        return target

    def _waypoints(self, start, target):
        """Corner points of the path: [approach start,] target."""
        lim = self.limits
        if lim.approach == 0 or lim.approach_overshoot <= 0:
            return [target]
        arriving_up = target > start
        if (lim.approach == 1 and arriving_up) or (lim.approach == -1 and target < start):
            return [target]
        if target == start:
            return [target]
        pre = target - lim.approach * lim.approach_overshoot
        pre = min(max(pre, lim.min), lim.max)        # the overshoot stays inside the range
        if (pre - target) * lim.approach >= 0:       # clamped to the target: no approach possible
            return [target]
        return [pre, target]

    # ---------------------------------------------------------------- motion
    def move_to(self, target, who):
        """Ramp to `target` (with the anti-backlash approach if configured).
        Blocks until done and settled mechanically. Returns a MoveResult."""
        with self._lock:
            target = self.check_target(target)
            lim = self.limits
            t0 = self._now()
            start = self._commanded
            steps = 0
            travel = 0.0
            pos = start
            try:
                for corner in self._waypoints(start, target):
                    n = max(1, math.ceil(abs(corner - pos) / lim.max_step - 1e-12))
                    leg_start = pos
                    for k in range(1, n + 1):
                        self._check_estop()
                        nxt = corner if k == n else leg_start + (corner - leg_start) * k / n
                        dt = abs(nxt - pos) / lim.max_slew
                        self._raw.move_to(nxt)
                        travel += abs(nxt - pos)
                        self._commanded = pos = nxt
                        steps += 1
                        if dt > 0:
                            self._sleep(dt)
            finally:
                self.travel_used += travel
                if steps:
                    self.ready_at = self._now() + lim.settle_s + lim.vibration_settle_s
            if lim.settle_s > 0:
                self._sleep(lim.settle_s)
            readback = self._raw.read_position() if self.info.has_readback else None
            result = MoveResult(target, self._commanded, readback, travel, steps,
                                self._now() - t0)
            self.log.append(axis=self.name, event='move', who=who, start=start,
                            **asdict(result))
            return result

    def move_by(self, delta, who):
        if self._commanded is None:
            raise SafetyError(f'{self.name}: position unknown (home it or declare it first)')
        return self.move_to(self._commanded + float(delta), who)

    def wait_until_settled(self):
        """Sleep until a measurement may start (motion + vibration settle done)."""
        remaining = self.ready_at - self._now()
        if remaining > 0:
            self._sleep(remaining)

    def describe(self):
        return {'name': self.name, 'units': self.units, 'kind': self.info.kind,
                'has_readback': self.info.has_readback, 'closed_loop': self.info.closed_loop,
                'position': self.position, 'commanded': self._commanded,
                'position_known': self.position_known, 'frozen': self.frozen,
                'frozen_in_config': self.limits.frozen,
                'estop': self.estopped, 'travel_used': self.travel_used,
                'limits': asdict(self.limits)}
