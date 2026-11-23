"""New Focus / Newport 8742 open-loop Picomotor controller (4 channels, USB).

One controller drives up to four Picomotors (axes 1-4), one at a time. Every
axis on the same controller shares one Picomotor8742Controller: one device
handle, one lock, opened by the first axis and closed when the last one closes.

Transport: Newport's USB driver (usbdll.dll, installed with the Picomotor
Application; the device shows up as "Picomotor Controller", USB VID 104D /
PID 4000, WinUSB service). The Picomotor Application holds the USB device
while it runs: close it first, or no controller is found.

ASCII protocol (8742 user manual, section 6; firmware 3.04 here):
    *IDN?   -> New_Focus 8742 v3.04 09/09/16 103147     (serial number last)
    1QM?    -> 3          motor type: 0 none, 1 unknown, 2 Tiny, 3 Standard
    1TP?    -> 100        step counter (since power-up, reset or DH)
    1MD?    -> 1          1 = motion done, 0 = moving
    1PA500                absolute move to counter value 500 (no reply)
    1VA2000               velocity, steps/s (Standard <= 2000, Tiny <= 1750)
    AB                    abort motion on all axes, abruptly
    TB?     -> 0, NO ERROR DETECTED     (pops the oldest error)
Replies end in CR LF.

What "position" means here. The 8742 has no limit switches, encoders or any
feedback. TP? only counts the steps the controller has sent, from wherever
the motor was at power-up. Forward and backward steps differ in size, so the
count drifts from the true position. The axis therefore uses the step counter
as its position (units 'steps') and SafeAxis's range is a travel window around
the counter's zero, not a physical limit. After a controller power cycle the
counter restarts at 0 wherever the motor is: the window moves with it.

Safety in this layer (SafeAxis adds range, ramps, freeze, E-stop on top):
- open() and close() never move. open() checks the identity, that a motor is
  connected on the channel (QM? = 2 or 3), and clears stale errors.
- Before every move the counter is read. If it differs from the last value
  this driver commanded (or read on open), the move is refused: someone moved
  the motor with the Picomotor Application or the hand pad, and SafeAxis's
  idea of the position is stale.
- Every move waits for MD? = 1 with a timeout from the distance and velocity
  (abort on timeout), then checks TB? for errors and that the counter landed
  on the target.
- Velocity is capped at the motor type's maximum (manual, VA command).
"""

import ctypes
import re
import threading
import time

from motion.axis import Axis, AxisInfo

USB_DLL = r'C:\Program Files\Newport\Newport USB Driver\Bin\usbdll.dll'
PRODUCT_ID = 0x4000
MOTOR_TYPES = {0: 'none', 1: 'unknown', 2: 'Tiny', 3: 'Standard'}
MAX_VELOCITY = {2: 1750, 3: 2000}      # steps/s per motor type (manual, VA)
CHANNELS = (1, 2, 3, 4)
POLL_S = 0.02


class UsbDll:
    """Newport's usbdll.dll, opened once per process (its init/uninit is global)."""

    _lock = threading.Lock()
    _users = 0
    _dll = None
    devices = {}            # device id -> *IDN? string

    @classmethod
    def acquire(cls):
        with cls._lock:
            if cls._users == 0:
                dll = ctypes.WinDLL(USB_DLL)
                dll.newp_usb_init_product.argtypes = [ctypes.c_int]
                dll.newp_usb_init_product.restype = ctypes.c_long
                dll.newp_usb_get_device_info.argtypes = [ctypes.c_char_p]
                dll.newp_usb_get_device_info.restype = ctypes.c_long
                dll.newp_usb_send_ascii.argtypes = [ctypes.c_long, ctypes.c_char_p, ctypes.c_ulong]
                dll.newp_usb_send_ascii.restype = ctypes.c_long
                dll.newp_usb_get_ascii.argtypes = [ctypes.c_long, ctypes.c_char_p, ctypes.c_ulong,
                                                   ctypes.POINTER(ctypes.c_ulong)]
                dll.newp_usb_get_ascii.restype = ctypes.c_long
                dll.newp_usb_uninit_system.argtypes = []
                dll.newp_usb_uninit_system.restype = None
                status = dll.newp_usb_init_product(PRODUCT_ID)
                if status != 0:
                    raise RuntimeError(f'usbdll: newp_usb_init_product failed ({status})')
                buf = ctypes.create_string_buffer(4096)
                status = dll.newp_usb_get_device_info(buf)
                devices = {}
                for item in buf.value.decode('ascii', 'replace').split(';'):
                    if ',' in item:
                        dev_id, desc = item.split(',', 1)
                        devices[int(dev_id)] = desc.strip()
                if status != 0 or not devices:
                    dll.newp_usb_uninit_system()
                    raise RuntimeError('no 8742 Picomotor controller found on USB. Is the New '
                                       'Focus Picomotor Application running? It holds the device.')
                cls._dll, cls.devices = dll, devices
            cls._users += 1
            return cls._dll

    @classmethod
    def release(cls):
        with cls._lock:
            if cls._users == 0:
                return
            cls._users -= 1
            if cls._users == 0:
                cls._dll.newp_usb_uninit_system()
                cls._dll, cls.devices = None, {}


class UsbTransport:
    """One 8742 on USB. `serial` None = the only controller attached."""

    def __init__(self, serial=None):
        self.serial = serial
        self.dll = UsbDll.acquire()
        try:
            self.device_id = self._pick(UsbDll.devices)
        except Exception:
            UsbDll.release()
            raise

    def _pick(self, devices):
        if self.serial is None:
            if len(devices) != 1:
                raise RuntimeError(f'{len(devices)} 8742 controllers on USB; give the serial '
                                   f'number in the address ("usb:<serial>"): {devices}')
            return next(iter(devices))
        for dev_id, desc in devices.items():
            if desc.split()[-1] == str(self.serial):
                return dev_id
        raise RuntimeError(f'no 8742 with serial {self.serial} on USB: {devices}')

    def write(self, command):
        data = command.encode('ascii')
        status = self.dll.newp_usb_send_ascii(self.device_id, data, len(data))
        if status != 0:
            raise RuntimeError(f'8742: sending {command!r} failed ({status})')

    def read(self):
        buf = ctypes.create_string_buffer(256)
        n = ctypes.c_ulong()
        status = self.dll.newp_usb_get_ascii(self.device_id, buf, 256, ctypes.byref(n))
        if status != 0:
            raise RuntimeError(f'8742: reading the reply failed ({status})')
        return buf.raw[:n.value].decode('ascii', 'replace').strip()

    def close(self):
        UsbDll.release()


def parse_address(address):
    """'usb' or 'usb:<serial>' -> serial (None = the only controller)."""
    m = re.fullmatch(r'(?i)usb(?::(\S+))?', str(address).strip())
    if not m:
        raise ValueError(f'8742 address must be "usb" or "usb:<serial>", got {address!r}')
    return m.group(1)


class Picomotor8742Controller:
    """One physical 8742. Use Picomotor8742Controller.shared(address)."""

    _instances = {}
    _instances_lock = threading.Lock()

    @classmethod
    def shared(cls, address):
        key = str(address).strip().lower()
        with cls._instances_lock:
            if key not in cls._instances:
                cls._instances[key] = cls(address)
            return cls._instances[key]

    def __init__(self, address):
        self.address = address
        self.serial = parse_address(address)
        self.lock = threading.RLock()
        self._io = None
        self._users = 0
        self.identity = None

    def _open_transport(self):
        """The device handle (tests replace this with a fake controller)."""
        return UsbTransport(self.serial)

    def acquire(self):
        """Open on first use. Reads only (and pops stale errors)."""
        with self.lock:
            if self._users == 0:
                io = self._open_transport()
                try:
                    self._io = io
                    self.identity = self.query('*IDN?')
                    if '8742' not in self.identity:
                        raise RuntimeError(f'{self.address} is not an 8742: {self.identity!r}')
                    for _ in range(10):                  # the error buffer holds up to 10
                        if self.query('TB?').startswith('0'):
                            break
                except Exception:
                    self._io = None
                    io.close()
                    raise
            self._users += 1

    def release(self):
        """Close when the last user leaves. Never moves."""
        with self.lock:
            if self._users == 0:
                return
            self._users -= 1
            if self._users == 0 and self._io is not None:
                self._io.close()
                self._io = None

    def send(self, command):
        with self.lock:
            if self._io is None:
                raise RuntimeError(f'8742 {self.address} is not open')
            self._io.write(command)

    def query(self, command):
        with self.lock:
            self.send(command)
            return self._io.read()

    def query_int(self, command):
        reply = self.query(command)
        try:
            return int(reply)
        except ValueError:
            raise RuntimeError(f'8742: {command} answered {reply!r}') from None

    def check_error(self):
        """Raise if the controller reports an error (TB? pops the oldest one)."""
        reply = self.query('TB?')
        if not reply.startswith('0'):
            raise RuntimeError(f'8742 error: {reply}')


class Picomotor8742Axis(Axis):
    """One channel of an 8742 as an open-loop axis in steps.

    Parameters
    ----------
    name : axis name (from axes_config)
    address : 'usb' (the only controller) or 'usb:<serial>'
    channel : 1-4
    velocity : steps/s to set on open (capped at the motor type's maximum);
        None leaves the controller's setting
    acceleration : steps/s^2 to set on open; None leaves the controller's setting
    move_timeout_s : extra time allowed beyond distance / velocity
    """

    def __init__(self, name, address, channel, velocity=None, acceleration=None,
                 move_timeout_s=2.0, sleep=time.sleep, now=time.monotonic):
        channel = int(channel)
        if channel not in CHANNELS:
            raise ValueError(f'8742 channel must be one of {CHANNELS}, got {channel!r}')
        self.info = AxisInfo(name, 'steps', 'steps', has_readback=False, closed_loop=False)
        self.channel = channel
        self.velocity = velocity
        self.acceleration = acceleration
        self.move_timeout_s = float(move_timeout_s)
        self.controller = Picomotor8742Controller.shared(address)
        self.motor_type = None
        self._velocity_now = None
        self._last = None           # counter value last commanded, or read on open
        self._sleep = sleep
        self._now = now
        self.is_open = False

    def _q(self, cmd):
        return self.controller.query_int(f'{self.channel}{cmd}')

    def open(self):
        if self.is_open:
            return
        self.controller.acquire()
        try:
            self.motor_type = self._q('QM?')
            if self.motor_type not in MAX_VELOCITY:
                raise RuntimeError(
                    f'{self.info.name}: no motor detected on 8742 channel {self.channel} '
                    f'(motor type {self.motor_type} = {MOTOR_TYPES.get(self.motor_type, "?")})')
            vmax = MAX_VELOCITY[self.motor_type]
            if self.velocity is not None:
                v = int(min(self.velocity, vmax))
                self.controller.send(f'{self.channel}VA{v}')
            v = self._q('VA?')
            if v > vmax:            # e.g. a Tiny motor left at the Standard default
                self.controller.send(f'{self.channel}VA{vmax}')
                v = self._q('VA?')
            self._velocity_now = v
            if self.acceleration is not None:
                self.controller.send(f'{self.channel}AC{int(self.acceleration)}')
            self.controller.check_error()
            self._last = self._q('TP?')
        except Exception:
            self.controller.release()
            raise
        self.is_open = True

    def close(self):
        if self.is_open:
            self.is_open = False
            self.controller.release()

    def read_position(self):
        """The controller's step counter (not a sensor)."""
        if not self.is_open:
            return None
        return self._q('TP?')

    def move_to(self, target):
        if not self.is_open:
            raise RuntimeError(f'{self.info.name} is not open')
        target = int(round(target))
        with self.controller.lock:
            now = self._q('TP?')
            if now != self._last:
                raise RuntimeError(
                    f'{self.info.name}: step counter is {now} but the last known value was '
                    f'{self._last} (moved from the Picomotor Application or the hand pad?). '
                    f'Not moving; reopen the axis to accept the new count.')
            if self._q('MD?') != 1:
                raise RuntimeError(f'{self.info.name}: a move is already in progress')
            if target == now:
                return
            self.controller.send(f'{self.channel}PA{target}')
        # poll without holding the lock, so stop() can get in
        deadline = self._now() + abs(target - now) / self._velocity_now + self.move_timeout_s
        while self._q('MD?') != 1:
            if self._now() > deadline:
                self.stop()
                self._last = self._q('TP?')
                raise RuntimeError(f'{self.info.name}: move to {target} did not finish in time; '
                                   f'aborted at {self._last}')
            self._sleep(POLL_S)
        landed = self._q('TP?')
        self._last = landed
        self.controller.check_error()
        if landed != target:        # aborted (E-stop) or refused
            raise RuntimeError(f'{self.info.name}: stopped at {landed}, target was {target}')

    def stop(self):
        """Abort motion on every channel of this controller, at once (AB)."""
        if self.is_open:
            self.controller.send('AB')
