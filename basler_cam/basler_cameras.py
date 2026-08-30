"""Reliable interface to Basler cameras via pypylon.

This module provides a thin, robust wrapper around pypylon that supports
several cameras open simultaneously. It is meant to be the camera backend
for a future GUI, but can also be run directly as a connectivity self-test:

    python basler_cameras.py

The self-test enumerates all connected Basler cameras, opens all of them at
the same time, sets exposure/gain, grabs a frame from each, prints image
statistics and closes everything cleanly.
"""

import sys
import time
from pathlib import Path

import numpy as np
from pypylon import genicam, pylon

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# burst_timing and CameraStreamer are camera-agnostic and now live in
# camera_core, shared with the XIMEA wrapper. Re-exported here because
# basler_gui.py, kalishlot/adapters/basler.py and mode_video_capture.py all
# import them from this module.
from camera_core import (CameraStreamer, burst_timing,  # noqa: E402,F401
                         check_camera_surface, sum_bin)


def _clip(node, value):
    """Set a float node to `value` clipped to its range; return what stuck."""
    node.SetValue(float(np.clip(value, node.Min, node.Max)))
    return node.GetValue()


def _snap(node, value):
    """Set an integer node, clipped to its range and snapped to its increment.

    Width, Height, OffsetX and the binning factors all carry increments, and a
    value that does not sit on the grid is rejected outright rather than
    rounded - so snap first and report back what the camera accepted.
    """
    low, high = node.Min, node.Max
    value = int(np.clip(value, low, high))
    try:
        increment = node.Inc or 1
    except Exception:
        increment = 1
    value = low + ((value - low) // increment) * increment
    node.SetValue(int(value))
    return node.GetValue()


def _entry_available(node, name):
    """Is `name` an available entry of this enumeration node right now?

    Availability is dynamic: LineSource only offers ExposureActive once the
    line is in Output mode, so this must be asked after LineMode is set.
    """
    try:
        entry = node.GetEntryByName(name)
        return entry is not None and bool(genicam.IsAvailable(entry))
    except Exception:
        return False


def _grab_timestamp(result):
    """Per-frame timestamp in nanoseconds, from the chunk if one is attached.

    The chunk stamp is applied by the camera at exposure; the transport-layer
    stamp is taken when the frame reaches the host, so it carries USB latency.
    Prefer the former and fall back to the latter.
    """
    for accessor in (lambda: result.ChunkTimestamp.GetValue(),
                     lambda: result.ChunkTimestamp.Value):
        try:
            return int(accessor())
        except Exception:
            continue
    return int(result.TimeStamp)


class BaslerCamera:
    """A single Basler camera, addressed by serial number.

    Usage:
        cam = BaslerCamera('24756778')
        cam.open()
        cam.exposure_us = 3000
        cam.gain_db = 0.0
        img = cam.grab()
        cam.close()

    or as a context manager:
        with BaslerCamera('24756778') as cam:
            img = cam.grab()
    """

    MAKE = 'basler'
    # acA2040 sensor pitch. The effective pixel is this times the binning
    # factor, which is what describe() records for later interpretation.
    PIXEL_SIZE_MM = 5.5 / 1000.0

    GRAB_TIMEOUT_MS = 10000
    MAX_FRAME_RATE = 10.0  # Hz, the default frame_rate_hz applied at open()
    THROUGHPUT_LIMIT_BPS = 150_000_000  # bytes/s per camera, two cameras fit in USB3
    GRAB_RETRIES = 3  # discarded-frame retries during streaming

    # Bytes put on the wire per pixel. Mono12p is the packed 12-bit format:
    # one and a half bytes per pixel, so it costs 25% less link than Mono12
    # while carrying the same levels.
    BYTES_PER_PIXEL = {'Mono8': 1.0, 'Mono12': 2.0, 'Mono12p': 1.5}
    # Highest value a pixel can take, per format. The optical synchronization
    # in pico_scope/mode_video_sync.py needs this to tell a saturated frame
    # from a merely bright one (see SYNCHRONIZED_VIDEO_SPECTRUM.md).
    SATURATION_LEVEL = {'Mono8': 255, 'Mono12': 4095, 'Mono12p': 4095}

    def __init__(self, serial_number):
        self.serial_number = str(serial_number)
        self._cam = None

    # ---------------------------------------------------------------- device
    @staticmethod
    def list_devices():
        """Return info dicts for all Basler cameras connected to this PC."""
        factory = pylon.TlFactory.GetInstance()
        return [{'serial_number': d.GetSerialNumber(),
                 'model': d.GetModelName(),
                 'friendly_name': d.GetFriendlyName()}
                for d in factory.EnumerateDevices()]

    @property
    def is_open(self):
        return self._cam is not None and self._cam.IsOpen()

    def open(self, pixel_format='Mono12'):
        """Connect to the camera and apply sane defaults."""
        if self.is_open:
            return
        factory = pylon.TlFactory.GetInstance()
        matches = [d for d in factory.EnumerateDevices()
                   if d.GetSerialNumber() == self.serial_number]
        if not matches:
            available = [d['serial_number'] for d in self.list_devices()]
            raise RuntimeError(f'camera s/n {self.serial_number} not found; '
                               f'available: {available}')
        self._cam = pylon.InstantCamera(factory.CreateDevice(matches[0]))
        self._cam.Open()

        self._cam.ExposureMode.SetValue('Timed')
        self._cam.ExposureAuto.SetValue('Off')
        self._cam.GainSelector.SetValue('All')
        self._cam.GainAuto.SetValue('Off')
        self._cam.PixelFormat.SetValue(pixel_format)

        # Two cameras free-running at full speed oversubscribe the USB3 bus
        # and frames get discarded ("payload data has been discarded").
        # Basler's recommended fix is to limit each camera's link throughput
        # so the sum over all cameras stays below the host controller
        # capacity (~380 MB/s practical for USB3). This paces the data on the
        # wire; capping the frame rate alone does not prevent burst collisions.
        self._cam.DeviceLinkThroughputLimitMode.SetValue('On')
        self._cam.DeviceLinkThroughputLimit.SetValue(self.THROUGHPUT_LIMIT_BPS)
        self._cam.AcquisitionFrameRateEnable.SetValue(True)
        self._cam.AcquisitionFrameRate.SetValue(self.MAX_FRAME_RATE)

    def close(self):
        if self._cam is not None:
            try:
                if self._cam.IsGrabbing():
                    self._cam.StopGrabbing()
                if self._cam.IsOpen():
                    self._cam.Close()
            finally:
                self._cam = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    # ------------------------------------------------------------- settings
    @property
    def pixel_size_mm(self):
        """Sensor pitch, before binning."""
        return self.PIXEL_SIZE_MM

    @property
    def model(self):
        return self._cam.DeviceModelName.GetValue()

    @property
    def frame_shape(self):
        """(height, width) of the frames the camera delivers."""
        return self._cam.Height.GetValue(), self._cam.Width.GetValue()

    @property
    def exposure_us(self):
        return self._cam.ExposureTime.GetValue()

    @exposure_us.setter
    def exposure_us(self, value):
        node = self._cam.ExposureTime
        self._cam.ExposureTime.SetValue(float(np.clip(value, node.Min, node.Max)))

    @property
    def gain_db(self):
        return self._cam.Gain.GetValue()

    @gain_db.setter
    def gain_db(self, value):
        node = self._cam.Gain
        self._cam.Gain.SetValue(float(np.clip(value, node.Min, node.Max)))

    @property
    def exposure_limits_us(self):
        node = self._cam.ExposureTime
        return node.Min, node.Max

    @property
    def gain_limits_db(self):
        node = self._cam.Gain
        return node.Min, node.Max

    @property
    def pixel_format(self):
        return self._cam.PixelFormat.GetValue()

    @property
    def saturation_level(self):
        """Highest value a pixel of the current format can take."""
        return self.SATURATION_LEVEL[self.pixel_format]

    def set_pixel_format(self, fmt):
        """Switch pixel format; returns the value the camera accepted.

        open() applies Mono12. Mono8 roughly halves the payload and is plenty
        for telling a round spot from a two-lobed pattern, which is all the
        mode video is asked to do.
        """
        self._cam.PixelFormat.SetValue(fmt)
        return self.pixel_format

    # -------------------------------------------------- acquisition timing
    @property
    def frame_rate_hz(self):
        """The requested frame rate. What you actually get is
        resulting_frame_rate, which also accounts for readout and exposure."""
        return self._cam.AcquisitionFrameRate.GetValue()

    @frame_rate_hz.setter
    def frame_rate_hz(self, value):
        self._cam.AcquisitionFrameRateEnable.SetValue(True)
        node = self._cam.AcquisitionFrameRate
        self._cam.AcquisitionFrameRate.SetValue(
            float(np.clip(value, node.Min, node.Max)))

    @property
    def resulting_frame_rate(self):
        """The camera's own answer for the rate it can sustain as configured.

        This is the number to trust: unlike the payload arithmetic in
        max_frame_rate_for() it also accounts for sensor readout time and for
        the exposure not fitting inside the frame period.
        """
        return self._cam.ResultingFrameRate.GetValue()

    @property
    def throughput_limit_bps(self):
        return self._cam.DeviceLinkThroughputLimit.GetValue()

    def set_throughput_limit(self, bytes_per_second):
        """Raise or lower the link throughput cap; returns the accepted value.

        THROUGHPUT_LIMIT_BPS is deliberately conservative so that two cameras
        can share the USB3 bus (see the comment in open()). Raise it only after
        confirming this camera is the only one streaming.
        """
        self._cam.DeviceLinkThroughputLimitMode.SetValue('On')
        node = self._cam.DeviceLinkThroughputLimit
        self._cam.DeviceLinkThroughputLimit.SetValue(
            int(np.clip(bytes_per_second, node.Min, node.Max)))
        return self.throughput_limit_bps

    @property
    def formats(self):
        """Mono formats this camera offers, shallowest first."""
        available = [name for name in ('Mono8', 'Mono12p', 'Mono12')
                     if _entry_available(self._cam.PixelFormat, name)]
        return tuple(available)

    @property
    def deepest_format(self):
        """The format with the most levels - what a capture wants by default.

        Headroom is the reason: as the laser warms the transmission climbs,
        and a clipped peak stops the camera tracking the photodiode.
        """
        available = self.formats
        for name in ('Mono12', 'Mono12p', 'Mono8'):
            if name in available:
                return name
        return self.pixel_format

    # ------------------------------------------------------ binning and ROI
    @property
    def max_binning(self):
        """Largest square binning the camera will accept, in firmware."""
        return int(min(self._cam.BinningHorizontal.Max,
                       self._cam.BinningVertical.Max))

    @property
    def binning_mode(self):
        """How binned pixels are combined: 'sum', 'average', or the fixed
        firmware mode when the camera has no node to ask.

        Measured, not assumed, on the acA2040-90umNIR: probe_binning_mode()
        found it sums, which is why binning here multiplies the signal.
        """
        for node_name in ('BinningVerticalMode', 'BinningHorizontalMode'):
            try:
                return str(getattr(self._cam, node_name).GetValue()).lower()
            except Exception:
                continue
        return 'firmware-sum'

    @property
    def binning(self):
        """(horizontal, vertical) binning factors."""
        return (self._cam.BinningHorizontal.GetValue(),
                self._cam.BinningVertical.GetValue())

    def set_binning(self, factor_x, factor_y=None, mode='Sum'):
        """Bin in the camera, before the data reaches the link.

        This is what makes binning reduce bandwidth - rebinning a full-size
        frame on the host saves nothing. Call this BEFORE set_roi(): binning
        changes what one pixel means, so Width/Height/OffsetX/OffsetY are
        expressed in binned pixels and their usable maximum shrinks by the
        binning factor.

        `mode` ('Sum' or 'Average') is not selectable on every model - the
        acA2040-90umNIR in this lab has no BinningHorizontalMode /
        BinningVerticalMode node at all and bins in a fixed firmware mode. That
        is not an error; the returned dict reports whether the mode was applied,
        so the caller can record what actually happened.
        """
        factor_y = factor_x if factor_y is None else factor_y
        # Offsets first: a non-zero offset can block the width/height change
        # that a binning change forces.
        _snap(self._cam.OffsetX, 0)
        _snap(self._cam.OffsetY, 0)
        applied_x = _snap(self._cam.BinningHorizontal, factor_x)
        applied_y = _snap(self._cam.BinningVertical, factor_y)

        modes = {}
        for axis, node_name in (('horizontal', 'BinningHorizontalMode'),
                                ('vertical', 'BinningVerticalMode')):
            try:
                node = getattr(self._cam, node_name)
                node.SetValue(mode)
                modes[axis] = node.GetValue()
            except Exception:
                modes[axis] = None  # node absent: fixed firmware mode
        return {'binning': (applied_x, applied_y),
                'mode_requested': mode,
                'mode_applied': modes,
                'mode_selectable': any(v is not None for v in modes.values()),
                'frame_shape': self.frame_shape}

    def set_roi(self, width, height, offset_x=None, offset_y=None):
        """Crop the transmitted frame; sizes are in BINNED pixels.

        Offsets default to centring the ROI. Every value is clipped to its
        node's range and snapped to its increment, and the accepted values come
        back in the returned dict - ask for 500 and you may well get 496.

        Apply set_binning() first; see its docstring for why the order matters.
        """
        _snap(self._cam.OffsetX, 0)
        _snap(self._cam.OffsetY, 0)
        applied_w = _snap(self._cam.Width, width)
        applied_h = _snap(self._cam.Height, height)
        centre_x = (self._cam.Width.Max - applied_w) // 2
        centre_y = (self._cam.Height.Max - applied_h) // 2
        applied_x = _snap(self._cam.OffsetX,
                          centre_x if offset_x is None else offset_x)
        applied_y = _snap(self._cam.OffsetY,
                          centre_y if offset_y is None else offset_y)
        return {'width': applied_w, 'height': applied_h,
                'offset_x': applied_x, 'offset_y': applied_y,
                'frame_shape': self.frame_shape}

    @property
    def max_frame_size(self):
        """(width, height) of the largest frame at the current binning.

        Shrinks by the binning factor, because Width and Height count binned
        pixels - which is the whole reason set_binning() has to come first.
        """
        return self._cam.Width.Max, self._cam.Height.Max

    def set_roi_full(self):
        """Uncrop: the largest frame the current binning allows."""
        width, height = self.max_frame_size
        return self.set_roi(width, height)

    def max_frame_rate_for(self, width, height, pixel_format=None, binning=1):
        """Link-limited frame rate for a given SENSOR-pixel ROI, in Hz.

        `width`/`height` are unbinned sensor pixels; the transmitted frame is
        that divided by `binning`, which is why binning buys a factor of
        binning**2 *on the link*.

        Payload arithmetic only, so it is an upper bound and frequently a very
        loose one: on the acA2040 the binding constraint is usually sensor
        readout, which is paced per row (~5.4 us) and which binning does NOT
        shorten. Measured on 25173136: 2048 rows -> 90 Hz, 1024 -> 178 Hz,
        512 -> 350 Hz, independent of width and of binning. Always cross-check
        against resulting_frame_rate once the camera is configured, or just use
        assert_frame_rate_reachable().
        """
        fmt = self.pixel_format if pixel_format is None else pixel_format
        try:
            bytes_per_pixel = self.BYTES_PER_PIXEL[fmt]
        except KeyError:
            raise ValueError(f'unknown pixel format {fmt!r}; known: '
                             f'{sorted(self.BYTES_PER_PIXEL)}')
        payload = (width // binning) * (height // binning) * bytes_per_pixel
        if payload <= 0:
            raise ValueError(f'empty ROI: {width}x{height} at binning {binning}')
        return self.throughput_limit_bps / payload

    def assert_frame_rate_reachable(self, requested_hz, tolerance=0.02):
        """Raise unless the camera can actually sustain `requested_hz`.

        Without this the camera quietly delivers fewer frames than asked for,
        which in a synchronized capture is silent data loss rather than a
        visible failure. Checks the camera's own ResultingFrameRate, which
        accounts for readout and exposure as well as for the link.
        """
        resulting = self.resulting_frame_rate
        if resulting >= requested_hz * (1 - tolerance):
            return resulting
        height, width = self.frame_shape
        binning_x, binning_y = self.binning
        wanted_period_us = 1e6 / requested_hz

        # Which of the three ceilings is the low one? Naming it saves the
        # caller from trying the remedies that cannot help. On this sensor
        # readout is paced per ROW and binning does not shorten it (it happens
        # after readout), so 'shrink the ROI' means shrink the HEIGHT.
        reasons = []
        if self.exposure_us >= wanted_period_us:
            reasons.append(
                f'the {self.exposure_us:.0f} us exposure alone caps the rate at '
                f'{1e6 / self.exposure_us:.1f} Hz - shorten it below '
                f'{wanted_period_us:.0f} us')
        link_max = self.max_frame_rate_for(width * binning_x, height * binning_y,
                                           self.pixel_format, binning_x)
        if link_max < requested_hz:
            reasons.append(
                f'the {self.throughput_limit_bps / 1e6:.0f} MB/s link cap allows '
                f'only {link_max:.1f} Hz - raise it with set_throughput_limit(), '
                f'bin harder, or use Mono8')
        if not reasons:
            reasons.append(
                f'sensor readout is the limit: it is paced per row, about '
                f'5.4 us per row, and binning does not shorten it. Reduce the '
                f'ROI HEIGHT ({height} binned rows = {height * binning_y} sensor '
                f'rows); the width is free')
        raise RuntimeError(
            f'camera {self.serial_number} cannot sustain {requested_hz:.1f} Hz: '
            f'ResultingFrameRate is {resulting:.1f} Hz with a {width}x{height} '
            f'frame in {self.pixel_format}, binning {binning_x}x{binning_y}, '
            f'exposure {self.exposure_us:.0f} us and a '
            f'{self.throughput_limit_bps / 1e6:.0f} MB/s link cap. '
            + '; '.join(reasons) + '.')

    # ----------------------------------------------------- digital I/O lines
    def set_exposure_active_output(self, line='Line3', source='ExposureActive'):
        """Emit a pulse on `line` for the duration of every exposure.

        Line3 and Line4 are the GPIO lines (open collector with an internal
        ~2 kOhm pull-up, so they drive a scope's 1 MOhm input unaided); Line2 is
        the opto-isolated output and needs an external pull-up. GPIO lines
        default to Input, and Basler warns that applying the wrong signal to a
        GPIO input can damage the camera - so LineMode is set to Output here,
        which is meant to happen before anything is connected.

        Falls back to Timer1 triggered on ExposureStart where ExposureActive is
        not offered; that produces the same pulse train.

        Not needed by the optical synchronization this repo actually uses (see
        SYNCHRONIZED_VIDEO_SPECTRUM.md) - kept because a hardware pulse train
        remains the best independent check if a cable is ever made up.
        """
        self._cam.LineSelector.SetValue(line)
        if genicam.IsWritable(self._cam.LineMode.Node):
            self._cam.LineMode.SetValue('Output')
        if _entry_available(self._cam.LineSource, source):
            self._cam.LineSource.SetValue(source)
            return {'line': line, 'source': self._cam.LineSource.GetValue(),
                    'via_timer': False}
        # fallback: a timer started by every exposure, lasting as long as it
        self._cam.TimerSelector.SetValue('Timer1')
        self._cam.TimerTriggerSource.SetValue('ExposureStart')
        _clip(self._cam.TimerDuration, self.exposure_us)
        self._cam.LineSource.SetValue('Timer1Active')
        return {'line': line, 'source': self._cam.LineSource.GetValue(),
                'via_timer': True,
                'timer_duration_us': self._cam.TimerDuration.GetValue()}

    def enable_chunks(self, selectors=('Timestamp',)):
        """Attach per-frame metadata to every grab.

        'Timestamp' is the one that matters: it stamps each frame with an exact
        time in the camera's own clock. That is what lets record_burst() report
        real frame times rather than assuming they are evenly spaced, and what
        makes a dropped frame visible without any help from the scope.
        """
        self._cam.ChunkModeActive.SetValue(True)
        enabled = []
        for selector in selectors:
            self._cam.ChunkSelector.SetValue(selector)
            self._cam.ChunkEnable.SetValue(True)
            enabled.append(selector)
        return enabled

    # ------------------------------------------------------------- grabbing
    def grab(self):
        """Grab a single frame and return it as a numpy array."""
        result = self._cam.GrabOne(self.GRAB_TIMEOUT_MS)
        try:
            if not result.GrabSucceeded():
                raise RuntimeError(f'grab failed on camera {self.serial_number}: '
                                   f'{result.ErrorCode} {result.ErrorDescription}')
            return result.Array.copy()
        finally:
            result.Release()

    def start_streaming(self):
        """Start continuous acquisition; retrieve frames with get_frame()."""
        if not self._cam.IsGrabbing():
            self._cam.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

    def get_frame(self):
        """Retrieve the latest frame while streaming.

        A frame the camera discarded for lack of USB bandwidth is skipped and
        the next one is awaited, up to GRAB_RETRIES times.
        """
        last_error = ''
        for _ in range(self.GRAB_RETRIES + 1):
            result = self._cam.RetrieveResult(self.GRAB_TIMEOUT_MS,
                                              pylon.TimeoutHandling_ThrowException)
            try:
                if result.GrabSucceeded():
                    return result.Array.copy()
                last_error = f'{result.ErrorCode} {result.ErrorDescription}'
            finally:
                result.Release()
        raise RuntimeError(f'grab failed on camera {self.serial_number}: {last_error}')

    def record_burst(self, n_frames, progress=None):
        """Grab exactly `n_frames` consecutive frames; return (frames, meta).

        `frames` is an (n, height, width) array in the camera's native dtype;
        `meta` has one row per frame with `block_id`, `camera_timestamp_ns` and
        `host_time_s`. Pass the result through burst_timing() to check it.

        Uses GrabStrategy_OneByOne. The streaming path in this module uses
        GrabStrategy_LatestImageOnly, which silently discards frames whenever
        the consumer falls behind - harmless for a live view, fatal here, where
        every frame has to be accounted for.

        Call enable_chunks() first if you want real camera timestamps;
        without chunks this falls back to the transport layer's stamp, which is
        taken on arrival rather than at exposure.
        """
        if n_frames < 1:
            raise ValueError(f'n_frames must be at least 1, got {n_frames}')
        if self._cam.IsGrabbing():
            raise RuntimeError(
                f'camera {self.serial_number} is already grabbing - stop the '
                f'live stream before recording a burst')

        height, width = self.frame_shape
        dtype = np.uint8 if self.pixel_format == 'Mono8' else np.uint16
        frames = np.empty((n_frames, height, width), dtype=dtype)
        meta = []

        self._cam.StartGrabbingMax(n_frames, pylon.GrabStrategy_OneByOne)
        try:
            for index in range(n_frames):
                result = self._cam.RetrieveResult(
                    self.GRAB_TIMEOUT_MS, pylon.TimeoutHandling_ThrowException)
                try:
                    if not result.GrabSucceeded():
                        raise RuntimeError(
                            f'camera {self.serial_number} failed on burst frame '
                            f'{index}/{n_frames}: {result.ErrorCode} '
                            f'{result.ErrorDescription}')
                    frames[index] = result.Array
                    meta.append({
                        'block_id': int(result.BlockID),
                        'camera_timestamp_ns': _grab_timestamp(result),
                        'host_time_s': time.time(),
                    })
                finally:
                    result.Release()
                if progress is not None:
                    progress(index + 1, n_frames)
        finally:
            self._cam.StopGrabbing()
        return frames, meta

    def describe(self):
        """Every setting a capture needs to record about itself.

        Written verbatim into the session file so that a frame stack can still
        be interpreted later - in particular the effective pixel size, which is
        the sensor pitch times the binning factor.
        """
        height, width = self.frame_shape
        binning_x, binning_y = self.binning
        return {
            'make': self.MAKE,
            'serial_number': self.serial_number,
            'model': self.model,
            'pixel_format': self.pixel_format,
            'pixel_size_mm': self.pixel_size_mm,
            'binning_mode': self.binning_mode,
            'saturation_level': self.saturation_level,
            'width': width,
            'height': height,
            'offset_x': self._cam.OffsetX.GetValue(),
            'offset_y': self._cam.OffsetY.GetValue(),
            'binning_x': binning_x,
            'binning_y': binning_y,
            'exposure_us': self.exposure_us,
            'gain_db': self.gain_db,
            'frame_rate_hz': self.frame_rate_hz,
            'resulting_frame_rate_hz': self.resulting_frame_rate,
            'throughput_limit_bps': self.throughput_limit_bps,
        }

    def stop_streaming(self):
        if self._cam.IsGrabbing():
            self._cam.StopGrabbing()


def self_test():
    """Open all connected cameras simultaneously and grab a frame from each."""
    devices = BaslerCamera.list_devices()
    if not devices:
        print('no Basler cameras found')
        return

    print(f'found {len(devices)} camera(s):')
    for d in devices:
        print(f"  s/n {d['serial_number']} - {d['model']}")

    cameras = [BaslerCamera(d['serial_number']) for d in devices]
    try:
        # open all cameras at the same time
        for cam in cameras:
            cam.open()
            print(f'\nopened {cam.serial_number} ({cam.model})')
            print(f'  exposure range: {cam.exposure_limits_us[0]:.0f} - '
                  f'{cam.exposure_limits_us[1]:.0f} us')
            print(f'  gain range: {cam.gain_limits_db[0]:.1f} - '
                  f'{cam.gain_limits_db[1]:.1f} dB')

        # set and read back exposure / gain on every camera
        for cam in cameras:
            cam.exposure_us = 3000
            cam.gain_db = 0.0
            print(f'{cam.serial_number}: set exposure -> read back '
                  f'{cam.exposure_us:.1f} us, gain -> {cam.gain_db:.1f} dB')

        # grab a single frame from each camera while all are open
        for cam in cameras:
            tic = time.time()
            img = cam.grab()
            dt = time.time() - tic
            print(f'{cam.serial_number}: grabbed {img.shape} {img.dtype} in {dt:.2f} s, '
                  f'min {img.min()}, max {img.max()}, mean {img.mean():.1f}')

        # short burst of continuous grabbing from all cameras interleaved
        for cam in cameras:
            cam.start_streaming()
        tic = time.time()
        n_frames = 5
        for i in range(n_frames):
            for cam in cameras:
                img = cam.get_frame()
            print(f'streaming frame {i + 1}/{n_frames} from all cameras ok')
        dt = time.time() - tic
        print(f'streamed {n_frames} frames from {len(cameras)} camera(s) '
              f'in {dt:.2f} s ({n_frames * len(cameras) / dt:.1f} frames/s total)')
        for cam in cameras:
            cam.stop_streaming()

        print('\nself-test passed')
    finally:
        for cam in cameras:
            cam.close()
        print('all cameras closed')


def probe_binning_mode(cam, exposure_us=5000, factor=2):
    """Is this camera's binning a Sum or an Average? Ask the sensor.

    acA2040 cameras have no BinningHorizontalMode / BinningVerticalMode node,
    so the mode cannot be read - but it can be measured. Grab the same scene at
    binning 1 and at binning `factor` with identical exposure and gain: summing
    multiplies the mean level by factor**2, averaging leaves it alone.

    Point the camera at something dim and unchanging. A scene bright enough to
    clip after summing makes the ratio meaningless, which is why the saturated
    fraction is reported alongside it.
    """
    print(f'\n--- binning mode probe (binning 1 vs {factor}) ---')
    original_format = cam.pixel_format
    original_binning = cam.binning
    # Mono12 for headroom: a x4 sum clips almost anything in Mono8.
    cam.set_pixel_format('Mono12')
    cam.exposure_us = exposure_us
    cam.gain_db = 0.0

    results = {}
    for binning in (1, factor):
        cam.set_binning(binning)
        cam.set_roi_full()
        image = cam.grab()
        saturated = float((image >= cam.saturation_level).mean())
        results[binning] = {'mean': float(image.mean()),
                            'max': int(image.max()),
                            'saturated_fraction': saturated,
                            'shape': image.shape}
        print(f'  binning {binning}: shape {image.shape}, mean '
              f'{image.mean():8.2f}, max {image.max():5d}, '
              f'saturated {saturated:.3%}')

    ratio = results[factor]['mean'] / max(results[1]['mean'], 1e-9)
    expected_sum = factor ** 2
    if results[factor]['saturated_fraction'] > 0.001:
        verdict = 'inconclusive - the binned frame is clipping, use a dimmer scene'
    elif ratio > (1 + expected_sum) / 2:
        verdict = f'Sum (ratio ~{expected_sum})'
    else:
        verdict = 'Average (ratio ~1)'
    print(f'  mean ratio {ratio:.2f} vs {expected_sum} for Sum, 1 for Average')
    print(f'  verdict: {verdict}')

    cam.set_binning(*original_binning)
    cam.set_roi_full()
    cam.set_pixel_format(original_format)
    return {'ratio': ratio, 'verdict': verdict, 'levels': results}


def burst_self_test(serial_number=None, n_frames=50, frame_rate_hz=100.0,
                    binning=2, pixel_format='Mono8'):
    """Record one burst the way a synchronized capture would, and report on it.

    This is the Phase 1a end-to-end check: it exercises binning, ROI, the
    frame rate, chunk timestamps and record_burst together, and prints the two
    numbers that the optical synchronization actually depends on -

      * the frame-to-frame brightness scatter, which decides whether the offset
        fit lands in its sub-0.15 ms regime or its failing one, and
      * the saturated pixel fraction, whose growth is the fit's one real
        failure mode.

    See pico_scope/SYNCHRONIZED_VIDEO_SPECTRUM.md. Run it against a live scene:
    on a static one the scatter is a noise floor, which is the useful number,
    but the brightness sequence itself will be featureless.
    """
    devices = BaslerCamera.list_devices()
    if not devices:
        print('no Basler cameras found')
        return
    if serial_number is None:
        serial_number = devices[0]['serial_number']
        print(f'no serial given, using the first camera: {serial_number}')

    cam = BaslerCamera(serial_number)
    cam.open()
    try:
        print(f'opened {cam.serial_number} ({cam.model})')
        probe_binning_mode(cam)

        print(f'\n--- configuring for {frame_rate_hz:g} Hz ---')
        cam.set_pixel_format(pixel_format)
        binning_info = cam.set_binning(binning)
        print(f'  binning {binning_info["binning"]}, mode selectable: '
              f'{binning_info["mode_selectable"]}')
        roi = cam.set_roi_full()
        print(f'  frame {roi["width"]}x{roi["height"]} in {cam.pixel_format}')

        period_us = 1e6 / frame_rate_hz
        cam.exposure_us = period_us  # exposure = frame period, no dead time
        cam.frame_rate_hz = frame_rate_hz
        sensor_h, sensor_w = roi['height'] * binning, roi['width'] * binning
        link_max = cam.max_frame_rate_for(sensor_w, sensor_h,
                                          cam.pixel_format, binning)
        print(f'  exposure {cam.exposure_us:.0f} us, link allows '
              f'{link_max:.1f} Hz, camera reports '
              f'{cam.resulting_frame_rate:.1f} Hz')
        cam.assert_frame_rate_reachable(frame_rate_hz)

        chunks = cam.enable_chunks()
        print(f'  chunks enabled: {chunks}')

        print(f'\n--- recording {n_frames} frames ---')
        tic = time.time()
        frames, meta = cam.record_burst(n_frames)
        wall = time.time() - tic
        print(f'  {frames.shape} {frames.dtype} in {wall:.2f} s '
              f'({n_frames / wall:.1f} frames/s wall clock)')

        report = burst_timing(meta, expected_rate_hz=frame_rate_hz)
        print(f'  dropped frames: {report["n_dropped"]} {report["dropped"]}')
        print(f'  frame period from camera timestamps: '
              f'{report["period_s_median"] * 1e3:.4f} ms '
              f'+- {report["period_s_std"] * 1e3:.4f} ms '
              f'(asked for {1e3 / frame_rate_hz:.4f} ms)')
        print(f'  timestamps look like nanoseconds: '
              f'{report.get("timestamps_look_like_ns")}')

        brightness = frames.reshape(n_frames, -1).mean(axis=1)
        scatter = brightness.std() / max(brightness.mean(), 1e-9)
        saturated = float((frames >= cam.saturation_level).mean())
        print(f'\n--- what the optical synchronization cares about ---')
        print(f'  frame brightness: mean {brightness.mean():.2f}, '
              f'min {brightness.min():.2f}, max {brightness.max():.2f}')
        print(f'  frame-to-frame scatter: {scatter:.2%} of the mean')
        print(f'  saturated pixels: {saturated:.3%} '
              f'(keep below ~1%; 5% breaks the offset fit)')

        print('\nburst self-test passed')
        return {'describe': cam.describe(), 'timing': report,
                'brightness_scatter': float(scatter),
                'saturated_fraction': saturated}
    finally:
        cam.close()
        print('camera closed')


if __name__ == '__main__':
    if '--burst-test' in sys.argv:
        index = sys.argv.index('--burst-test') + 1
        serial = (sys.argv[index]
                  if index < len(sys.argv) and not sys.argv[index].startswith('-')
                  else None)
        burst_self_test(serial)
    elif '--binning-mode' in sys.argv:
        index = sys.argv.index('--binning-mode') + 1
        serial = (sys.argv[index]
                  if index < len(sys.argv) and not sys.argv[index].startswith('-')
                  else BaslerCamera.list_devices()[0]['serial_number'])
        with BaslerCamera(serial) as camera:
            probe_binning_mode(camera)
    else:
        self_test()
