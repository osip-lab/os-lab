"""Pure-Python interface to PicoScope 4000A-series oscilloscopes (picosdk),
built for continuous streaming of slow signals (chart-recorder style).

No GUI imports — this is the device layer for any interface (kalishlot web
GUI, scripts), per the lab's device/GUI decoupling rule.

Run directly for a connectivity self-test (streams 2 s from channel A):

    python pico_scope/ps4000a_scope.py

Design (same single-owner pattern as basler_cam CameraStreamer): after
open() a background thread owns the driver handle, continuously moving
samples from the driver into per-channel numpy ring buffers. Consumers call
read_window() at their own pace — the newest data is always there, nobody
ever waits on the device. Reconfiguration (channels, sample rate) is
submitted as commands executed by the thread between polls, because it
requires stopping and restarting the driver's streaming.

Only one program can own the scope: close PicoScope 7 before using this.
"""

import ctypes
import queue
import threading
import time

import numpy as np
from picosdk.constants import PICO_STATUS_LOOKUP
from picosdk.ps4000a import ps4000a as ps

CHANNEL_NAMES = ('A', 'B', 'C', 'D')
COUPLINGS = {'AC': ps.PS4000A_COUPLING['PS4000A_AC'],
             'DC': ps.PS4000A_COUPLING['PS4000A_DC']}
# range index -> full scale in volts (4000A analog ranges: +-10 mV .. +-50 V)
RANGES = {index: volts for index, volts in ps.PICO_VOLTAGE_RANGE.items()
          if index <= 11}

# statuses OpenUnit returns for USB-powered scopes on a USB-2 port / without
# the auxiliary supply; ChangePowerSource(status) accepts and continues
POWER_CHANGE_STATUSES = (282, 286)

DRIVER_BUFFER_SAMPLES = 65536  # per channel, handed to the driver
RING_CAPACITY_SECONDS = 70  # keep a bit more than the largest view window
RING_CAPACITY_MAX = 8_000_000  # samples per channel (16 MB of int16)
POLL_INTERVAL_S = 0.01

# Block mode. Directions are the PS4000A_THRESHOLD_DIRECTION enum; 'NONE' with
# a non-zero auto-trigger is how a block is made to start immediately, which is
# what a synchronized capture wants - there is nothing to trigger on, the two
# records are aligned afterwards from the light itself.
TRIGGER_DIRECTIONS = {'above': 0, 'below': 1, 'rising': 2, 'falling': 3,
                      'rising_or_falling': 4, 'none': 2}
BLOCK_POLL_INTERVAL_S = 0.002


def _check(status, what):
    if status != 0:
        name = PICO_STATUS_LOOKUP.get(status, status)
        raise RuntimeError(f'{what} failed: {name}')


class RingBuffer:
    """Fixed-capacity ring of int16 samples; thread-safe append and read."""

    def __init__(self, capacity):
        self.capacity = int(capacity)
        self._data = np.zeros(self.capacity, dtype=np.int16)
        self._written = 0  # total samples ever appended
        self._lock = threading.Lock()

    def append(self, block):
        n = len(block)
        with self._lock:
            if n >= self.capacity:
                self._data[:] = block[-self.capacity:]
            else:
                start = self._written % self.capacity
                end = start + n
                if end <= self.capacity:
                    self._data[start:end] = block
                else:
                    split = self.capacity - start
                    self._data[start:] = block[:split]
                    self._data[:end - self.capacity] = block[split:]
            self._written += n

    def last(self, n):
        """Return up to the n newest samples, oldest first."""
        with self._lock:
            n = min(int(n), self._written, self.capacity)
            end = self._written % self.capacity
            if n == 0:
                return np.empty(0, dtype=np.int16)
            if n <= end:
                return self._data[end - n:end].copy()
            return np.concatenate((self._data[-(n - end):], self._data[:end]))


class PicoScope4000A:
    """One 4000A-series scope in continuous streaming mode."""

    def __init__(self, serial=None):
        self.serial = serial
        self.variant = ''
        self._handle = ctypes.c_int16(0)
        self._max_adc = 32767
        # configuration; 'range_v' is the full scale of the +-range
        self.channels = {name: {'enabled': name == 'A', 'coupling': 'DC',
                                'range_v': 5.0}
                         for name in CHANNEL_NAMES}
        self.sample_rate_hz = 1000.0
        self._rings = {}
        self._driver_buffers = {}
        self._commands = queue.Queue()
        self._thread = None
        self._stopping = threading.Event()
        self.on_error = None  # optional callable(exception), from the thread
        self._block = None  # in-flight block capture, see start_block()

    # ---------------------------------------------------------------- device
    @staticmethod
    def list_devices():
        count = ctypes.c_int16(0)
        serials = ctypes.create_string_buffer(256)
        length = ctypes.c_int16(len(serials))
        status = ps.ps4000aEnumerateUnits(ctypes.byref(count), serials,
                                          ctypes.byref(length))
        if status != 0 or count.value == 0:
            return []
        return [{'serial': s} for s in serials.value.decode().split(',') if s]

    @property
    def is_open(self):
        return self._handle.value > 0

    def open(self):
        if self.is_open:
            return
        serial = self.serial.encode() if self.serial else None
        status = ps.ps4000aOpenUnit(ctypes.byref(self._handle), serial)
        if status in POWER_CHANGE_STATUSES:
            # USB-powered scope on a USB-2 port: accept and continue
            status = ps.ps4000aChangePowerSource(self._handle, status)
        if status != 0 or self._handle.value <= 0:
            name = PICO_STATUS_LOOKUP.get(status, status)
            raise RuntimeError(
                f'could not open PicoScope ({name}) — if it is open in '
                f'another program (e.g. PicoScope 7), close it there first')
        info = ctypes.create_string_buffer(64)
        needed = ctypes.c_int16()
        ps.ps4000aGetUnitInfo(self._handle, info, len(info),
                              ctypes.byref(needed), 3)  # variant
        self.variant = info.value.decode()
        if not self.serial:
            ps.ps4000aGetUnitInfo(self._handle, info, len(info),
                                  ctypes.byref(needed), 4)  # serial
            self.serial = info.value.decode()
        max_adc = ctypes.c_int16()
        _check(ps.ps4000aMaximumValue(self._handle, ctypes.byref(max_adc)),
               'MaximumValue')
        self._max_adc = max_adc.value

    def close(self):
        self.stop_streaming()
        if self.is_open:
            try:
                ps.ps4000aCloseUnit(self._handle)
            finally:
                self._handle = ctypes.c_int16(0)

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    # -------------------------------------------------------- configuration
    # Public setters store the wish and hand it to the streaming thread
    # (stop -> apply -> restart); when not streaming they apply directly.
    def configure_channel(self, name, enabled=None, coupling=None,
                          range_v=None):
        """Update one channel; returns the accepted configuration."""
        if name not in self.channels:
            raise ValueError(f'no channel {name!r}')
        config = dict(self.channels[name])
        if enabled is not None:
            config['enabled'] = bool(enabled)
        if coupling is not None:
            if coupling not in COUPLINGS:
                raise ValueError(f"coupling must be 'AC' or 'DC'")
            config['coupling'] = coupling
        if range_v is not None:
            # snap to the closest range that still contains the request
            candidates = [v for v in RANGES.values() if v >= float(range_v)]
            config['range_v'] = min(candidates) if candidates \
                else max(RANGES.values())
        if not config['enabled'] and not any(
                c['enabled'] for n, c in self.channels.items() if n != name):
            raise ValueError('at least one channel must stay enabled')
        self._reconfigure(lambda: self.channels.__setitem__(name, config))
        return dict(self.channels[name])

    def set_sample_rate(self, hertz):
        """Set the sample rate; returns the accepted value."""
        hertz = float(np.clip(hertz, 1.0, 1_000_000.0))
        self._reconfigure(lambda: setattr(self, 'sample_rate_hz', hertz))
        return self.sample_rate_hz

    def _reconfigure(self, apply):
        if self.is_streaming:
            done = threading.Event()

            def command():
                self._driver_stop()
                apply()
                self._driver_start()
                done.set()

            self._commands.put(command)
            if not done.wait(timeout=10):
                raise RuntimeError('scope reconfiguration timed out')
        else:
            apply()

    # ------------------------------------------------------------- streaming
    @property
    def is_streaming(self):
        return self._thread is not None and self._thread.is_alive()

    def start_streaming(self):
        if self.is_streaming:
            return
        self._driver_start()
        self._stopping.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name=f'pico-{self.serial}')
        self._thread.start()

    def stop_streaming(self):
        self._stopping.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
        if self.is_open:
            ps.ps4000aStop(self._handle)

    def _driver_start(self):
        """Apply the channel configuration and start driver streaming."""
        enabled = []
        for index, name in enumerate(CHANNEL_NAMES):
            config = self.channels[name]
            range_index = next(i for i, v in sorted(RANGES.items())
                               if v >= config['range_v'])
            _check(ps.ps4000aSetChannel(
                self._handle, index, int(config['enabled']),
                COUPLINGS[config['coupling']], range_index, 0.0),
                f'SetChannel {name}')
            if config['enabled']:
                enabled.append((index, name))

        capacity = min(int(self.sample_rate_hz * RING_CAPACITY_SECONDS),
                       RING_CAPACITY_MAX)
        self._rings = {}
        self._driver_buffers = {}
        for index, name in enabled:
            self._rings[name] = RingBuffer(capacity)
            buffer = np.zeros(DRIVER_BUFFER_SAMPLES, dtype=np.int16)
            self._driver_buffers[name] = buffer
            _check(ps.ps4000aSetDataBuffer(
                self._handle, index,
                buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
                DRIVER_BUFFER_SAMPLES, 0,
                ps.PS4000A_RATIO_MODE['PS4000A_RATIO_MODE_NONE']),
                f'SetDataBuffer {name}')

        interval = ctypes.c_int32(max(round(1e6 / self.sample_rate_hz), 1))
        _check(ps.ps4000aRunStreaming(
            self._handle, ctypes.byref(interval),
            ps.PS4000A_TIME_UNITS['PS4000A_US'],
            0, 0, 0,  # no pre/post trigger limit, no autostop
            1, ps.PS4000A_RATIO_MODE['PS4000A_RATIO_MODE_NONE'],
            DRIVER_BUFFER_SAMPLES), 'RunStreaming')
        # the driver may adjust the interval; keep the accepted rate
        self.sample_rate_hz = 1e6 / interval.value

    def _driver_stop(self):
        ps.ps4000aStop(self._handle)

    def _loop(self):
        def on_data(handle, n_samples, start_index, overflow, trigger_at,
                    triggered, auto_stop, param):
            for name, buffer in self._driver_buffers.items():
                self._rings[name].append(
                    buffer[start_index:start_index + n_samples])

        callback = ps.StreamingReadyType(on_data)
        while not self._stopping.is_set():
            try:
                while True:
                    try:
                        command = self._commands.get_nowait()
                    except queue.Empty:
                        break
                    command()
                ps.ps4000aGetStreamingLatestValues(self._handle, callback,
                                                   None)
            except Exception as error:
                if self._stopping.is_set():
                    break
                if self.on_error is not None:
                    self.on_error(error)
                break
            time.sleep(POLL_INTERVAL_S)

    # --------------------------------------------------------- block capture
    # Streaming is for watching a slow signal indefinitely; a block is for
    # capturing a known span at full rate, which is what a spectrum recording
    # is. The two cannot run at once - start_streaming() must be stopped first.
    def _range_index(self, name):
        return next(i for i, v in sorted(RANGES.items())
                    if v >= self.channels[name]['range_v'])

    def _apply_channels(self):
        """Push the channel configuration to the driver."""
        enabled = []
        for index, name in enumerate(CHANNEL_NAMES):
            config = self.channels[name]
            _check(ps.ps4000aSetChannel(
                self._handle, index, int(config['enabled']),
                COUPLINGS[config['coupling']], self._range_index(name), 0.0),
                f'SetChannel {name}')
            if config['enabled']:
                enabled.append((index, name))
        return enabled

    def find_timebase(self, sample_interval_s, n_samples):
        """Smallest timebase whose interval is at least `sample_interval_s`.

        The 4000A timebase is an integer code, not a rate: for this family
        interval = (timebase + 1) / 80 MHz. The driver is asked to confirm each
        candidate rather than trusting the formula, because the achievable set
        depends on how many channels are enabled.

        Returns (timebase, actual_interval_s, max_samples).
        """
        wanted_ns = sample_interval_s * 1e9
        interval_ns = ctypes.c_float()
        max_samples = ctypes.c_int32()
        best = None
        timebase = 0
        while timebase < 2 ** 30:
            status = ps.ps4000aGetTimebase2(
                self._handle, timebase, int(n_samples),
                ctypes.byref(interval_ns), ctypes.byref(max_samples), 0)
            if status == 0:
                best = (timebase, interval_ns.value, max_samples.value)
                if interval_ns.value >= wanted_ns:
                    return timebase, interval_ns.value * 1e-9, max_samples.value
                # step: intervals grow linearly, so jump straight to the target
                step = max(1, int(wanted_ns / max(interval_ns.value, 1e-9)) - 1)
                timebase += step
            else:
                timebase += 1
            if timebase > 1_000_000:
                break
        if best is None:
            raise RuntimeError(
                f'no usable timebase for {n_samples} samples - the request may '
                f'exceed the scope memory with these channels enabled')
        return best[0], best[1] * 1e-9, best[2]

    def configure_trigger(self, source='D', threshold_v=0.0, direction='rising',
                          delay=0, auto_trigger_ms=1000, enabled=True):
        """Set the simple trigger used by capture_block.

        `auto_trigger_ms` is how long the scope waits before giving up and
        capturing anyway; 0 means wait forever. For a synchronized capture pass
        enabled=False, which arms nothing and starts the block immediately.
        """
        if enabled:
            if source not in CHANNEL_NAMES:
                raise ValueError(f'no channel {source!r}')
            if direction not in TRIGGER_DIRECTIONS:
                raise ValueError(f'direction must be one of '
                                 f'{sorted(TRIGGER_DIRECTIONS)}')
            full_scale = self.channels[source]['range_v']
            adc = int(np.clip(threshold_v / full_scale, -1.0, 1.0) * self._max_adc)
            _check(ps.ps4000aSetSimpleTrigger(
                self._handle, 1, CHANNEL_NAMES.index(source), adc,
                TRIGGER_DIRECTIONS[direction], int(delay),
                int(auto_trigger_ms)), 'SetSimpleTrigger')
        else:
            _check(ps.ps4000aSetSimpleTrigger(self._handle, 0, 0, 0, 2, 0, 0),
                   'SetSimpleTrigger (disable)')
        self._trigger = {'enabled': bool(enabled), 'source': source,
                         'threshold_v': threshold_v, 'direction': direction,
                         'delay': int(delay),
                         'auto_trigger_ms': int(auto_trigger_ms)}
        return dict(self._trigger)

    def start_block(self, duration_s, sample_interval_s, pre_fraction=0.0):
        """Arm and start a block capture; returns immediately.

        Split from the collection step on purpose: a synchronized capture has
        to start the camera burst *while* the scope is recording, so the caller
        needs the scope running and control back. Follow with
        wait_block() and read_block(), or use capture_block() for both.
        """
        if self.is_streaming:
            raise RuntimeError('stop streaming before starting a block capture')
        n_samples = max(int(round(duration_s / sample_interval_s)), 2)
        enabled = self._apply_channels()
        if not enabled:
            raise RuntimeError('no channel is enabled')
        timebase, interval, max_samples = self.find_timebase(sample_interval_s,
                                                             n_samples)
        if n_samples > max_samples:
            raise RuntimeError(
                f'{n_samples} samples exceeds the {max_samples} the scope can '
                f'hold with {len(enabled)} channel(s) enabled at '
                f'{interval * 1e9:.0f} ns - shorten the capture, slow the '
                f'sample rate, or disable a channel')
        if not hasattr(self, '_trigger'):
            self.configure_trigger(enabled=False)

        pre = int(round(n_samples * float(np.clip(pre_fraction, 0.0, 1.0))))
        post = n_samples - pre
        self._block = {'pre': pre, 'post': post, 'n_samples': n_samples,
                       'interval_s': interval, 'timebase': timebase,
                       'enabled': enabled, 'host_start_s': time.time()}
        _check(ps.ps4000aRunBlock(self._handle, pre, post, timebase, None, 0,
                                  None, None), 'RunBlock')
        return dict(self._block)

    def wait_block(self, timeout_s=None):
        """Block until the capture is complete; returns the wait in seconds."""
        if getattr(self, '_block', None) is None:
            raise RuntimeError('no block capture is running')
        deadline = None if timeout_s is None else time.time() + timeout_s
        ready = ctypes.c_int16(0)
        tic = time.time()
        while not ready.value:
            _check(ps.ps4000aIsReady(self._handle, ctypes.byref(ready)),
                   'IsReady')
            if ready.value:
                break
            if deadline is not None and time.time() > deadline:
                ps.ps4000aStop(self._handle)
                raise TimeoutError(
                    f'block capture did not complete within {timeout_s:.1f} s'
                    + (' - the trigger never fired; pass auto_trigger_ms or '
                       'configure_trigger(enabled=False)'
                       if getattr(self, '_trigger', {}).get('enabled') else ''))
            time.sleep(BLOCK_POLL_INTERVAL_S)
        return time.time() - tic

    def read_block(self):
        """Collect the finished block: (t, {channel: volts}, info).

        `t` has **zero at the trigger**, so a pre-trigger fraction gives
        negative times. The legacy adc2mv path in pico_scope/__init__.py puts
        zero at the first sample instead; that difference is a trap worth not
        repeating.
        """
        info = getattr(self, '_block', None)
        if info is None:
            raise RuntimeError('no block capture to read')
        n = info['n_samples']
        buffers = {}
        for index, name in info['enabled']:
            maximum = np.zeros(n, dtype=np.int16)
            minimum = np.zeros(n, dtype=np.int16)
            buffers[name] = (maximum, minimum)
            _check(ps.ps4000aSetDataBuffers(
                self._handle, index,
                maximum.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
                minimum.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
                n, 0, ps.PS4000A_RATIO_MODE['PS4000A_RATIO_MODE_NONE']),
                f'SetDataBuffers {name}')
        collected = ctypes.c_int32(n)
        overflow = ctypes.c_int16()
        _check(ps.ps4000aGetValues(self._handle, 0, ctypes.byref(collected),
                                   1, ps.PS4000A_RATIO_MODE[
                                       'PS4000A_RATIO_MODE_NONE'], 0,
                                   ctypes.byref(overflow)), 'GetValues')
        ps.ps4000aStop(self._handle)

        count = collected.value
        t = (np.arange(count) - info['pre']) * info['interval_s']
        volts = {name: self.to_volts(name, maximum[:count])
                 for name, (maximum, _) in buffers.items()}
        overflowed = [name for index, name in info['enabled']
                      if overflow.value & (1 << index)]
        result = dict(info)
        result.update({'n_collected': count, 'overflow_channels': overflowed,
                       'host_end_s': time.time()})
        result.pop('enabled', None)
        if overflowed:
            print(f'  ! channel(s) {overflowed} overflowed their range during '
                  f'the block - widen them with configure_channel(range_v=...)')
        self._block = None
        return t, volts, result

    def capture_block(self, duration_s, sample_interval_s, pre_fraction=0.0,
                      timeout_s=None):
        """Arm, wait and collect in one call: (t, {channel: volts}, info)."""
        info = self.start_block(duration_s, sample_interval_s, pre_fraction)
        self.wait_block(timeout_s=timeout_s
                        if timeout_s is not None else duration_s * 3 + 10)
        return self.read_block()

    # ---------------------------------------------------------------- data
    def read_window(self, seconds):
        """Newest `seconds` of data: (dt, {channel: int16 array}).
        Arrays may be shorter right after a (re)start while the buffer
        fills; they are aligned at the newest sample."""
        dt = 1.0 / self.sample_rate_hz
        n = int(seconds / dt)
        return dt, {name: ring.last(n) for name, ring in self._rings.items()}

    def to_volts(self, name, adc):
        return adc.astype(np.float64) * (self.channels[name]['range_v']
                                         / self._max_adc)


def self_test():
    devices = PicoScope4000A.list_devices()
    print('devices:', devices or 'none found via enumeration')
    with PicoScope4000A() as scope:
        print(f'opened {scope.variant} s/n {scope.serial}, '
              f'max ADC {scope._max_adc}')
        print('ranges (V):', sorted(RANGES.values()))
        scope.configure_channel('A', enabled=True, coupling='DC', range_v=5.0)
        accepted = scope.set_sample_rate(10_000)
        print(f'sample rate accepted: {accepted:g} Hz')
        scope.start_streaming()
        time.sleep(2.0)
        dt, window = scope.read_window(1.5)
        for name, adc in window.items():
            volts = scope.to_volts(name, adc)
            print(f'channel {name}: {len(adc)} samples at dt = {dt * 1e6:.1f} us, '
                  f'min {volts.min():+.4f} V, max {volts.max():+.4f} V, '
                  f'mean {volts.mean():+.4f} V')
        scope.stop_streaming()
    print('self-test passed')


if __name__ == '__main__':
    self_test()
