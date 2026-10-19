"""The raw axis interface every motor driver implements.

A raw Axis talks to hardware and nothing else: no limits, no ramps, no logging.
It is never handed to kalishlot, scripts or the optimizer directly; they get a
SafeAxis (safety.py) that owns the only reference to it.

Units are the axis's own (V for an open-loop piezo, mm or um for a stage, deg
for a rotation/goniometer, steps for an open-loop stepper or inertia drive).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class AxisInfo:
    name: str
    units: str
    kind: str = 'position'      # 'voltage' | 'position' | 'angle' | 'steps'
    has_readback: bool = False  # read_position() returns something meaningful
    closed_loop: bool = False   # the readback is a real sensor (encoder, strain gauge),
                                # not just the last command echoed back
    can_home: bool = False      # home() finds an absolute reference


class Axis(ABC):
    info: AxisInfo

    def open(self):
        """Connect. Must not move the axis or change its output."""

    def close(self):
        """Disconnect. Must not move the axis or change its output (no zeroing)."""

    @abstractmethod
    def read_position(self):
        """Current position in axis units, or None when the device cannot tell
        (open-loop without readback, unhomed stepper)."""

    @abstractmethod
    def move_to(self, target):
        """Go to `target` and return when the motion has finished. SafeAxis only
        ever calls this with small, pre-checked steps."""

    def stop(self):
        """Stop any motion now (E-stop). Must not jump anywhere."""

    def home(self):
        raise NotImplementedError(f'{self.info.name} cannot home')
