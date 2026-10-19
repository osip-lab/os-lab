"""Thorlabs MDT693B 3-channel open-loop piezo driver (USB virtual COM port).

One controller drives up to three axes (channels x, y, z). Every axis on the
same port shares one MDT693BController: one VISA session, one lock, opened
by the first axis and closed when the last one closes.

ASCII protocol (firmware 1.17, 115200 baud, commands end in CR):
    id?               -> multi-line identity
    vlimit?           -> [ 150]       output limit switch on the back (75/100/150 V)
    xvoltage?         -> [  5.07]     measured output voltage of channel x
    xvoltage=5.0      -> (nothing)    set channel x
Every reply ends with the prompt '>'; with echo on, the command is echoed first.

Safety in this layer (SafeAxis adds limits, ramps, freeze, E-stop on top):
- open() and close() never write a voltage (no zeroing on exit).
- open() checks the identity and that the axis's soft max fits under the
  hardware limit switch (SafeAxis is told via `max_allowed`).
- Before every write the output is read back. If it differs from the last
  value written (or read on open) by more than `drift_tolerance`, the move is
  refused: someone turned the front-panel knob, and ramping from the stale
  value would be a jump.

The readback is the controller's own measured output voltage, not a sensor on
the mount, so the axis is open-loop (has_readback=True, closed_loop=False).
It reads ~0.07 V above the command at 5 V (measured 2026-10-19).
"""

import re
import threading
import time

from motion.axis import Axis, AxisInfo

BAUD_RATE = 115200
TIMEOUT_MS = 2000
CHANNELS = ('x', 'y', 'z')


def visa_address(address):
    """'COM4' or 'ASRL4::INSTR' -> 'ASRL4::INSTR'."""
    m = re.fullmatch(r'(?i)com(\d+)', str(address).strip())
    return f'ASRL{m.group(1)}::INSTR' if m else str(address).strip()


class MDT693BController:
    """One physical MDT693B. Use MDT693BController.shared(address)."""

    _instances = {}
    _instances_lock = threading.Lock()

    @classmethod
    def shared(cls, address):
        key = visa_address(address)
        with cls._instances_lock:
            if key not in cls._instances:
                cls._instances[key] = cls(key)
            return cls._instances[key]

    def __init__(self, address):
        self.address = visa_address(address)
        self.lock = threading.RLock()
        self._inst = None
        self._users = 0
        self.identity = None
        self.voltage_limit = None

    # ------------------------------------------------------------ lifecycle
    def acquire(self):
        """Open the port on first use. Reads only."""
        with self.lock:
            if self._users == 0:
                inst = self._open_instrument()
                try:
                    self._inst = inst
                    lines = self.query('id?')
                    if not any('MDT693B' in s for s in lines):
                        raise RuntimeError(f'{self.address} is not an MDT693B: {lines}')
                    self.identity = ' | '.join(lines)
                    self.voltage_limit = float(self._bracketed(self.query('vlimit?')))
                except Exception:
                    self._inst = None
                    inst.close()
                    raise
            self._users += 1

    def _open_instrument(self):
        """The VISA session (tests replace this with a fake instrument)."""
        import pyvisa
        inst = pyvisa.ResourceManager().open_resource(self.address, baud_rate=BAUD_RATE,
                                                      timeout=TIMEOUT_MS)
        inst.write_termination = '\r'
        inst.read_termination = None
        inst.flush(pyvisa.constants.VI_READ_BUF_DISCARD)   # (inst.clear() fails on this port)
        return inst

    def release(self):
        """Close the port when the last user leaves. Never writes."""
        with self.lock:
            if self._users == 0:
                return
            self._users -= 1
            if self._users == 0 and self._inst is not None:
                self._inst.close()
                self._inst = None

    # ------------------------------------------------------------- protocol
    def query(self, command):
        """Send `command`, read up to the '>' prompt, return the reply lines
        (echo removed)."""
        with self.lock:
            if self._inst is None:
                raise RuntimeError(f'MDT693B {self.address} is not open')
            self._inst.write(command)
            buf = b''
            while not buf.endswith(b'>'):
                buf += self._inst.read_bytes(1)     # raises VisaIOError on timeout
            lines = [s.strip() for s in buf[:-1].decode('ascii', 'replace').split('\r')]
            lines = [s for s in lines if s]
            if lines and lines[0] == command:
                lines = lines[1:]
            return lines

    @staticmethod
    def _bracketed(lines):
        for s in lines:
            m = re.fullmatch(r'\[\s*([-+0-9.eE]+)\s*\]', s)
            if m:
                return m.group(1)
        raise RuntimeError(f'MDT693B: no value in reply {lines}')

    def read_voltage(self, channel):
        return float(self._bracketed(self.query(f'{channel}voltage?')))

    def set_voltage(self, channel, volts):
        reply = self.query(f'{channel}voltage={volts:.3f}')
        if reply:       # a set command answers with the prompt only
            raise RuntimeError(f'MDT693B refused {channel}voltage={volts:.3f}: {reply}')


class MDT693BAxis(Axis):
    """One channel of an MDT693B as an open-loop voltage axis.

    Parameters
    ----------
    name : axis name (from axes_config)
    address : 'COM4' or a VISA resource ('ASRL4::INSTR')
    channel : 'x', 'y' or 'z'
    max_allowed : the axis's soft max (V); open() refuses if the hardware limit
        switch is set lower, since the output would then silently saturate
    drift_tolerance : V; refuse to move when the output differs from the last
        known value by more than this (front-panel knob turned)
    """

    def __init__(self, name, address, channel, max_allowed=None, drift_tolerance=0.5):
        channel = str(channel).lower()
        if channel not in CHANNELS:
            raise ValueError(f'MDT693B channel must be one of {CHANNELS}, got {channel!r}')
        self.info = AxisInfo(name, 'V', 'voltage', has_readback=True, closed_loop=False)
        self.channel = channel
        self.max_allowed = max_allowed
        self.drift_tolerance = float(drift_tolerance)
        self.controller = MDT693BController.shared(address)
        self._last = None           # last value written, or read on open
        self.is_open = False

    def open(self):
        if self.is_open:
            return
        self.controller.acquire()
        try:
            lim = self.controller.voltage_limit
            if self.max_allowed is not None and lim is not None and self.max_allowed > lim + 1e-9:
                raise RuntimeError(f'{self.info.name}: soft max {self.max_allowed} V is above the '
                                   f'MDT693B limit switch ({lim:g} V)')
            self._last = self.controller.read_voltage(self.channel)
        except Exception:
            self.controller.release()
            raise
        self.is_open = True

    def close(self):
        if self.is_open:
            self.is_open = False
            self.controller.release()

    def read_position(self):
        if not self.is_open:
            return None
        return self.controller.read_voltage(self.channel)

    def move_to(self, target):
        if not self.is_open:
            raise RuntimeError(f'{self.info.name} is not open')
        with self.controller.lock:
            now = self.controller.read_voltage(self.channel)
            if self._last is not None and abs(now - self._last) > self.drift_tolerance:
                raise RuntimeError(
                    f'{self.info.name}: output is {now:.2f} V but the last known value was '
                    f'{self._last:.2f} V (front-panel knob?). Not moving; reopen the axis to '
                    f'accept the new voltage.')
            self.controller.set_voltage(self.channel, float(target))
            self._last = float(target)

    def stop(self):
        """A voltage step is immediate: there is no motion in progress to stop."""


def find_controllers():
    """[(visa address, identity)] for every serial port that answers id? as an MDT693B.

    Writes 'id?' to every ASRL port, so only call it when no other serial
    instrument would mind an unknown command."""
    import pyvisa
    found = []
    for res in pyvisa.ResourceManager().list_resources('ASRL?*::INSTR'):
        c = MDT693BController(res)
        try:
            c.acquire()
            found.append((res, c.identity))
        except Exception:
            pass
        finally:
            c.release()
    return found
