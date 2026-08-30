"""Reliable interface to XIMEA cameras via xiapi, to the shared camera contract.

The counterpart of `basler_cam/basler_cameras.py`. Both wrap a different SDK
and present the one surface written down in `camera_core.py`, so the kalishlot
adapters and `pico_scope/mode_video_capture.py` work with either camera and
never ask which one they are holding.

    python ximea_cam/ximea_cameras.py

The self-test enumerates the connected XIMEA cameras, opens one, records a
burst and checks the frame timing - which needs no light, so a capped camera in
a dark room is a perfectly good test of it.

## Why this module is not `ximea_cam/__init__.py`

That one imports PyQt6 and numba at module scope, which makes it unusable from
a web server or a headless capture script. Device control has to be plain
Python; consumers reach this module by putting `ximea_cam/` on `sys.path` and
importing `ximea_cameras` flat, which also sidesteps executing the package
`__init__`.

## Where this camera differs from the Basler, and what is done about it

Measured on the MQ042MG-CM-S7-TG in this lab, not assumed:

- **No binning of any kind.** `downsampling_maximum` is `XI_DWN_1x1`, so there
  is no firmware 2x2 to ask for. `set_binning(2)` therefore sums 2x2 on the
  host as frames arrive. The caller gets the same 4x signal and the same binned
  coordinates as the Basler's firmware Sum; `binning_mode` reports which one
  actually happened, and `describe()` records it.
- **10-bit sensor, and 10 is the maximum.** `XI_MONO16` carries 10 bits
  right-aligned, so full scale is 1023, not 65535 - getting this wrong would
  make every frame look unsaturated. Exposed as `Mono10` rather than `Mono16`
  for that reason.
- **Bandwidth is quoted in Mbit/s**, where pylon uses bytes/s. Converted here,
  so `throughput_limit_bps` means the same thing on both cameras.
- **Timestamps arrive as `tsSec`/`tsUSec`** on the frame rather than as chunk
  data, and are converted to the nanoseconds the contract asks for. There is no
  enable step.
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from camera_core import (CameraStreamer, burst_timing,  # noqa: E402,F401
                         check_burst_meta, check_camera_surface, sum_bin)

try:
    from ximea import xiapi
except Exception as error:                                # pragma: no cover
    raise ImportError(
        "Missing dependency 'ximea' required by ximea_cameras.\n\n"
        "It is not on PyPI: install the XIMEA Software Package from "
        "ximea.com, then run python ximea_cam/install_ximea_package.py\n"
    ) from error


def _text(value):
    """xiapi returns C strings as bytes; addresses and labels want str."""
    return value.decode('ascii', 'replace') if isinstance(value, bytes) else str(value)


def _snap(value, minimum, maximum, increment):
    """Clip to a range and round down to a legal increment.

    Down rather than to-nearest: rounding an ROI up can push it past the
    sensor edge, and xiapi rejects the whole call rather than clamping.
    """
    value = int(np.clip(value, minimum, maximum))
    return int(minimum + ((value - minimum) // increment) * increment)


class XimeaCamera:
    """A single XIMEA camera, addressed by serial number.

    Usage:
        cam = XimeaCamera('QXNAA2506000')
        cam.open()
        cam.exposure_us = 3000
        cam.gain_db = 0.0
        img = cam.grab()
        cam.close()

    or as a context manager:
        with XimeaCamera('QXNAA2506000') as cam:
            img = cam.grab()
    """

    MAKE = 'ximea'
    # MQ042MG (CMV4000) sensor pitch, the same as the acA2040's.
    PIXEL_SIZE_MM = 5.5 / 1000.0

    GRAB_TIMEOUT_MS = 10000
    MAX_FRAME_RATE = 10.0  # Hz, the default frame_rate_hz applied at open()
    GRAB_RETRIES = 3
    # Frames the driver may hold while the consumer is busy, capped at what
    # the driver reports it can do. The default of 4 is 40 ms of slack at
    # 100 Hz - one busy moment on the host and the burst loses frames, which
    # has been seen. Deeper buffers cost only memory.
    BUFFER_QUEUE_SIZE = 64

    # Our names for the formats, mapped to xiapi's. 'Mono10' rather than
    # 'Mono16' because the wire format is 16-bit but the data is 10 - and it is
    # the number of levels, not the container, that says when a pixel clips.
    FORMATS = {'Mono8': 'XI_MONO8', 'Mono10': 'XI_MONO16'}
    BYTES_PER_PIXEL = {'Mono8': 1.0, 'Mono10': 2.0}

    def __init__(self, serial_number):
        self.serial_number = str(serial_number)
        self._cam = None
        self._binning = 1
        self._streaming = False
        self._image = None

    # ---------------------------------------------------------------- device
    @staticmethod
    def list_devices():
        """Return info dicts for all XIMEA cameras connected to this PC.

        Reads the serial without opening the camera, so enumerating does not
        disturb a camera another program is already using.
        """
        try:
            probe = xiapi.Camera()
            count = probe.get_number_devices()
        except Exception:
            return []
        devices = []
        for index in range(count):
            try:
                handle = xiapi.Camera(dev_id=index)
                serial = _text(handle.get_device_info_string('device_sn'))
                model = _text(handle.get_device_info_string('device_name'))
                devices.append({'serial_number': serial, 'model': model,
                                'friendly_name': f'{model} s/n {serial}'})
            except Exception:
                continue        # a camera held open elsewhere, not a failure
        return devices

    @property
    def is_open(self):
        return self._cam is not None and bool(getattr(self._cam, 'CAM_OPEN', False))

    def open(self, pixel_format='Mono10'):
        """Connect to the camera and apply sane defaults."""
        if self.is_open:
            return
        cam = xiapi.Camera()
        try:
            cam.open_device_by_SN(self.serial_number)
        except Exception as error:
            available = [d['serial_number'] for d in self.list_devices()]
            raise RuntimeError(f'camera s/n {self.serial_number} could not be '
                               f'opened ({error}); available: {available}') from error
        self._cam = cam
        self._image = xiapi.Image()

        self.set_pixel_format(pixel_format)
        cam.set_gain(0.0)
        # Pace the camera from its own clock rather than free-running, so a
        # requested frame rate means something and get_framerate_maximum()
        # reports the ceiling the current settings actually allow.
        cam.set_acq_timing_mode('XI_ACQ_TIMING_MODE_FRAME_RATE')
        self.frame_rate_hz = self.MAX_FRAME_RATE
        # SAFE copies each frame out of the driver's buffer before returning
        # it. UNSAFE hands back a pointer that the next grab may overwrite,
        # which is not survivable for a burst held in memory.
        cam.set_buffer_policy('XI_BP_SAFE')
        # Clipped to the reported maximum rather than asked for blind: a value
        # over the maximum is rejected outright, and a rejected queue size
        # leaves the shallow default in place while looking like it worked.
        cam.set_buffers_queue_size(min(self.BUFFER_QUEUE_SIZE,
                                       cam.get_buffers_queue_size_maximum()))

    def close(self):
        """Stop acquisition and release the camera. Safe to call twice."""
        if self._cam is None:
            return
        try:
            if self._streaming:
                self.stop_streaming()
        except Exception:
            pass
        try:
            self._cam.close_device()
        except Exception:
            pass
        self._cam = None
        self._image = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *exc_info):
        self.close()
        return False

    @property
    def model(self):
        return _text(self._cam.get_device_name())

    @property
    def pixel_size_mm(self):
        """Sensor pitch, before binning."""
        return self.PIXEL_SIZE_MM

    @property
    def frame_shape(self):
        """(height, width) of a delivered frame, in binned pixels."""
        return (self._cam.get_height() // self._binning,
                self._cam.get_width() // self._binning)

    # ------------------------------------------------- exposure, gain, depth
    @property
    def exposure_us(self):
        return float(self._cam.get_exposure())

    @exposure_us.setter
    def exposure_us(self, value):
        low, high = self.exposure_limits_us
        self._cam.set_exposure(float(np.clip(value, low, high)))

    @property
    def exposure_limits_us(self):
        return (float(self._cam.get_exposure_minimum()),
                float(self._cam.get_exposure_maximum()))

    @property
    def gain_db(self):
        return float(self._cam.get_gain())

    @gain_db.setter
    def gain_db(self, value):
        low, high = self.gain_limits_db
        self._cam.set_gain(float(np.clip(value, low, high)))

    @property
    def gain_limits_db(self):
        return (float(self._cam.get_gain_minimum()),
                float(self._cam.get_gain_maximum()))

    @property
    def pixel_format(self):
        """Our name for the current format: 'Mono8' or 'Mono10'."""
        current = _text(self._cam.get_imgdataformat())
        for name, xi_name in self.FORMATS.items():
            if xi_name == current:
                return name
        return current

    @property
    def formats(self):
        """Mono formats this camera offers, shallowest first."""
        return ('Mono8', 'Mono10')

    @property
    def deepest_format(self):
        """The format with the most levels - what a capture wants by default."""
        return 'Mono10'

    @property
    def bit_depth(self):
        """Bits actually carried by a pixel, which is not the container size.

        `XI_MONO16` on this sensor delivers 10 significant bits right-aligned,
        so a pixel clips at 1023 while looking like a 16-bit value.
        """
        if self.pixel_format == 'Mono8':
            return 8
        depth = _text(self._cam.get_output_bit_depth())   # e.g. 'XI_BPP_10'
        return int(depth.rsplit('_', 1)[-1])

    @property
    def _sum_dtype(self):
        """Accumulator for host-side binning: the narrowest that cannot clip.

        Left to itself sum_bin has to assume a 16-bit container really holds
        16 bits and widen to uint32. This camera knows better - 10 bits summed
        4 or 16 at a time still fits uint16 - and the narrower accumulator is
        four times faster. That is not a micro-optimisation here: binning runs
        once per frame inside the burst loop, and at 100 Hz the wide version
        cost 9.5 ms of a 10 ms period and dropped frames.
        """
        return (np.uint16 if self.saturation_level <= np.iinfo(np.uint16).max
                else np.uint32)

    @property
    def saturation_level(self):
        """Highest value a delivered pixel can take, binning included.

        Binning sums, so N x N binning raises full scale by N**2 - the same
        arithmetic as the Basler's firmware Sum.
        """
        return (2 ** self.bit_depth - 1) * self._binning ** 2

    def set_pixel_format(self, fmt):
        """Set the format and return the accepted one."""
        if fmt not in self.FORMATS:
            raise ValueError(f'unknown pixel format {fmt!r}; '
                             f'this camera offers {", ".join(self.formats)}')
        self._cam.set_imgdataformat(self.FORMATS[fmt])
        return self.pixel_format

    # ------------------------------------------------------------ frame rate
    @property
    def frame_rate_hz(self):
        """The rate the camera has been asked to keep."""
        return float(self._cam.get_framerate())

    @frame_rate_hz.setter
    def frame_rate_hz(self, value):
        low = float(self._cam.get_framerate_minimum())
        high = float(self._cam.get_framerate_maximum())
        self._cam.set_framerate(float(np.clip(value, low, high)))

    @property
    def frame_rate_limits_hz(self):
        """(min, max) rate the camera can keep as currently configured.

        The maximum moves with exposure, ROI and link, so it is the honest
        ceiling for a rate control - unlike `resulting_frame_rate`, which
        reports what is happening now and would pin a slider to its own value.
        """
        return (float(self._cam.get_framerate_minimum()),
                float(self._cam.get_framerate_maximum()))

    @property
    def resulting_frame_rate(self):
        """The rate the camera can actually sustain as configured.

        The counterpart of pylon's ResultingFrameRate, and load-bearing for the
        same reason: `choose_roi()` in the capture script sizes the ROI purely
        by probing this. `get_framerate_maximum()` already accounts for
        exposure, ROI and link, so the achievable rate is the smaller of what
        was asked for and what is possible.
        """
        return min(float(self._cam.get_framerate()),
                   float(self._cam.get_framerate_maximum()))

    @property
    def throughput_limit_bps(self):
        """Current link limit in BYTES per second, as pylon quotes it."""
        return float(self._cam.get_limit_bandwidth()) * 1e6 / 8.0

    def set_throughput_limit(self, bytes_per_second):
        """Cap the link in bytes/s; returns the accepted value.

        xiapi speaks Mbit/s, so the unit is converted here rather than leaving
        every caller to remember which camera it is talking to.
        """
        mbit = bytes_per_second * 8.0 / 1e6
        low = float(self._cam.get_limit_bandwidth_minimum())
        high = float(self._cam.get_limit_bandwidth_maximum())
        try:
            self._cam.set_limit_bandwidth_mode('XI_ON')
        except Exception:
            pass
        self._cam.set_limit_bandwidth(int(np.clip(mbit, low, high)))
        return self.throughput_limit_bps

    @property
    def available_bandwidth_bps(self):
        """What the link measured itself as being able to carry, in bytes/s."""
        return float(self._cam.get_available_bandwidth()) * 1e6 / 8.0

    # ------------------------------------------------------ binning and ROI
    @property
    def binning(self):
        """(horizontal, vertical) binning factors."""
        return (self._binning, self._binning)

    @property
    def max_binning(self):
        """Largest square binning available, firmware or host.

        Host-side summing has no hard ceiling, but past 4 the frames get small
        enough that the mode is no longer resolved, which is the point of the
        video.
        """
        return 4

    @property
    def binning_mode(self):
        return 'host-sum' if self._binning > 1 else 'none'

    def set_binning(self, factor_x, factor_y=None, mode='Sum'):
        """Bin by summing factor x factor blocks on the host.

        This camera has no firmware downsampling at all - `downsampling_maximum`
        is XI_DWN_1x1 - so unlike the Basler this saves no link bandwidth. It
        buys the other two things binning is wanted for: factor**2 more signal
        per pixel, and a quarter of the data to hold and store, which is what
        makes long records affordable.

        Call this BEFORE set_roi(): binning changes what one pixel means, so
        ROI sizes are expressed in binned pixels and their maximum shrinks by
        the factor, exactly as on the Basler.
        """
        factor_y = factor_x if factor_y is None else factor_y
        if factor_x != factor_y:
            raise ValueError(f'host-side binning is square only, not '
                             f'{factor_x}x{factor_y}')
        if mode.lower() != 'sum':
            raise ValueError(f'host-side binning sums; {mode!r} is not offered')
        if not 1 <= factor_x <= self.max_binning:
            raise ValueError(f'binning must be 1..{self.max_binning}, '
                             f'not {factor_x}')
        self._binning = int(factor_x)
        return {'binning': (self._binning, self._binning),
                'mode_requested': mode,
                'mode_applied': {'horizontal': 'Sum', 'vertical': 'Sum'},
                'mode_selectable': False,
                'binning_mode': self.binning_mode,
                'frame_shape': self.frame_shape}

    @property
    def max_frame_size(self):
        """(width, height) at the current binning, in binned pixels."""
        return (self._cam.get_width_maximum() // self._binning,
                self._cam.get_height_maximum() // self._binning)

    def set_roi(self, width, height, offset_x=None, offset_y=None):
        """Crop the sensor. Sizes are in BINNED pixels, offsets default to centred.

        Every value is clipped and snapped to the increments the camera
        enforces (measured here: width 4, height 2, offsets 4 and 2), and to a
        multiple of the binning factor so that summing leaves no remainder.
        """
        cam = self._binning
        # Offsets to zero first: a non-zero offset caps the width/height that
        # can be requested, so a grow-then-move would be rejected.
        cam_obj = self._cam
        cam_obj.set_offsetX(0)
        cam_obj.set_offsetY(0)

        step_w = np.lcm(cam_obj.get_width_increment(), cam)
        step_h = np.lcm(cam_obj.get_height_increment(), cam)
        sensor_w = _snap(width * cam, cam_obj.get_width_minimum(),
                         cam_obj.get_width_maximum(), step_w)
        sensor_h = _snap(height * cam, cam_obj.get_height_minimum(),
                         cam_obj.get_height_maximum(), step_h)
        cam_obj.set_width(sensor_w)
        cam_obj.set_height(sensor_h)

        full_w, full_h = cam_obj.get_width_maximum(), cam_obj.get_height_maximum()
        want_x = (full_w - sensor_w) // 2 if offset_x is None else offset_x * cam
        want_y = (full_h - sensor_h) // 2 if offset_y is None else offset_y * cam
        step_ox = np.lcm(cam_obj.get_offsetX_increment(), cam)
        step_oy = np.lcm(cam_obj.get_offsetY_increment(), cam)
        cam_obj.set_offsetX(_snap(want_x, 0, cam_obj.get_offsetX_maximum(), step_ox))
        cam_obj.set_offsetY(_snap(want_y, 0, cam_obj.get_offsetY_maximum(), step_oy))

        return {'width': cam_obj.get_width() // cam,
                'height': cam_obj.get_height() // cam,
                'offset_x': cam_obj.get_offsetX() // cam,
                'offset_y': cam_obj.get_offsetY() // cam,
                'frame_shape': self.frame_shape}

    def set_roi_full(self):
        """Uncrop to the whole sensor at the current binning."""
        width, height = self.max_frame_size
        return self.set_roi(width, height, 0, 0)

    def max_frame_rate_for(self, width, height, pixel_format=None, binning=1):
        """Upper bound on frame rate from link bandwidth alone, in SENSOR pixels.

        Host-side binning does not reduce what crosses the wire, so unlike on
        the Basler the `binning` argument does not lower the cost here - it is
        accepted so that callers need not know which camera they hold.
        """
        pixel_format = pixel_format or self.pixel_format
        if pixel_format not in self.BYTES_PER_PIXEL:
            raise ValueError(f'unknown pixel format {pixel_format!r}')
        if width <= 0 or height <= 0:
            raise ValueError('empty ROI')
        bytes_per_frame = width * height * self.BYTES_PER_PIXEL[pixel_format]
        return self.available_bandwidth_bps / bytes_per_frame

    def assert_frame_rate_reachable(self, requested_hz, tolerance=0.02):
        """Return the achievable rate, or explain which ceiling is binding."""
        achievable = self.resulting_frame_rate
        if achievable >= requested_hz * (1.0 - tolerance):
            return achievable
        width, height = self._cam.get_width(), self._cam.get_height()
        link = self.max_frame_rate_for(width, height, self.pixel_format)
        exposure_ceiling = 1e6 / self.exposure_us
        raise RuntimeError(
            f'{requested_hz:g} Hz is not reachable: the camera reports '
            f'{achievable:.1f} Hz at {width}x{height} {self.pixel_format}. '
            f'The exposure alone caps it at {exposure_ceiling:.1f} Hz and the '
            f'link at {link:.1f} Hz - whichever is lower is what to change '
            f'(shorten the exposure, or cut ROI pixels).')

    # ------------------------------------------------------------- capturing
    def _frame_from(self, image, raw_peak=False):
        """Copy a delivered frame out of the driver buffer, binning if asked.

        `raw_peak` also returns the highest value any *sensor* pixel reached.
        Binning hides clipping - four summed pixels reach full scale only if
        all four clipped - so a burst that wants to know whether it saturated
        has to look before the sum, and the conversion is only worth doing once.
        """
        raw = image.get_image_data_numpy()
        if self._binning > 1:
            frame = sum_bin(raw, self._binning, dtype=self._sum_dtype)
        else:
            frame = np.array(raw, copy=True)
        return (frame, int(raw.max())) if raw_peak else frame

    @staticmethod
    def _stamp_ns(image):
        """The camera's own timestamp for a frame, in nanoseconds.

        xiapi splits it into whole seconds and microseconds; the contract in
        camera_core asks for nanoseconds on an arbitrary but stable epoch.
        """
        return int(image.tsSec) * 1_000_000_000 + int(image.tsUSec) * 1000

    def enable_timestamps(self):
        """Nothing to do: every xiapi frame arrives already stamped.

        Present so that callers need not know which camera stamps frames for
        free and which has to be asked (the Basler needs chunk data enabled).
        """
        return ()

    def grab(self):
        """Grab a single frame, starting and stopping acquisition around it."""
        self._cam.start_acquisition()
        try:
            self._cam.get_image(self._image, timeout=self.GRAB_TIMEOUT_MS)
            return self._frame_from(self._image)
        finally:
            self._cam.stop_acquisition()

    def start_streaming(self):
        """Begin continuous acquisition, newest frame first."""
        if self._streaming:
            return
        # The counterpart of pylon's LatestImageOnly: a live view wants the
        # most recent frame, not the oldest queued one.
        self._cam.enable_recent_frame()
        self._cam.start_acquisition()
        self._streaming = True

    def get_frame(self):
        """Latest frame while streaming."""
        for attempt in range(self.GRAB_RETRIES):
            try:
                self._cam.get_image(self._image, timeout=self.GRAB_TIMEOUT_MS)
                return self._frame_from(self._image)
            except xiapi.Xi_error:
                if attempt == self.GRAB_RETRIES - 1:
                    raise
        raise RuntimeError('no frame after retries')          # pragma: no cover

    def stop_streaming(self):
        if not self._streaming:
            return
        self._cam.stop_acquisition()
        self._streaming = False

    def record_burst(self, n_frames, progress=None):
        """Record exactly `n_frames` consecutive frames with their timing.

        Every frame is kept, in order - the point of a burst, as against the
        live view, is that no frame is silently skipped. `acq_nframe` is the
        camera's own count of frames it captured, so a gap in it is the only
        honest evidence that one went missing.

        Returns `(frames, meta)` per the contract in camera_core.
        """
        if self._streaming:
            raise RuntimeError('stop the live stream before recording a burst')
        self._cam.disable_recent_frame()   # FIFO: every frame, in order

        height, width = self.frame_shape
        dtype = np.uint8 if (self.pixel_format == 'Mono8'
                             and self._binning == 1) else np.uint16
        frames = np.empty((n_frames, height, width), dtype=dtype)
        meta = []
        raw_peak = 0

        self._cam.start_acquisition()
        try:
            for index in range(n_frames):
                self._cam.get_image(self._image, timeout=self.GRAB_TIMEOUT_MS)
                frames[index], peak = self._frame_from(self._image, raw_peak=True)
                raw_peak = max(raw_peak, peak)
                meta.append({
                    'block_id': int(self._image.acq_nframe),
                    'camera_timestamp_ns': self._stamp_ns(self._image),
                    'host_time_s': time.time(),
                })
                if progress is not None:
                    progress(index + 1, n_frames)
        finally:
            self._cam.stop_acquisition()

        if meta:
            # What the sensor peaked at before binning summed the evidence
            # away, so a caller can still tell a clipped burst from a bright one.
            meta[0]['raw_peak'] = raw_peak
            meta[0]['raw_full_scale'] = 2 ** self.bit_depth - 1
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
            'bit_depth': self.bit_depth,
            'width': width,
            'height': height,
            'offset_x': self._cam.get_offsetX() // self._binning,
            'offset_y': self._cam.get_offsetY() // self._binning,
            'binning_x': binning_x,
            'binning_y': binning_y,
            'exposure_us': self.exposure_us,
            'gain_db': self.gain_db,
            'frame_rate_hz': self.frame_rate_hz,
            'resulting_frame_rate_hz': self.resulting_frame_rate,
            'throughput_limit_bps': self.throughput_limit_bps,
        }


def self_test(serial_number=None, n_frames=200, frame_rate_hz=100.0,
              binning=2, pixel_format='Mono10'):
    """Open a camera, record a burst, and check the timing that sync rests on.

    Needs no light: what is being tested is whether the camera paces itself at
    the rate it was asked for, whether it drops frames, and whether its
    timestamps are its own rather than the host's. A dark frame answers all
    three.
    """
    check_camera_surface(XimeaCamera)
    print('ximea_cameras self-test')

    devices = XimeaCamera.list_devices()
    print(f'  {len(devices)} camera(s) connected')
    for device in devices:
        print(f"    {device['serial_number']}  {device['model']}")
    if not devices:
        print('  nothing to test against')
        return None
    serial_number = serial_number or devices[0]['serial_number']

    cam = XimeaCamera(serial_number)
    cam.open()
    try:
        check_camera_surface(cam)
        cam.set_pixel_format(pixel_format)
        cam.set_binning(binning)
        width, height = cam.max_frame_size
        roi = cam.set_roi(width // 2, height // 4)
        # The exposure has to stay under the frame period or it becomes the
        # cap on the rate itself.
        cam.exposure_us = 1e6 / frame_rate_hz - 100.0
        cam.gain_db = 0.0
        cam.frame_rate_hz = frame_rate_hz
        print(f"  {roi['width']}x{roi['height']} binned pixels at "
              f"({roi['offset_x']}, {roi['offset_y']}), {cam.pixel_format}, "
              f"full scale {cam.saturation_level}, binning {cam.binning_mode}")
        print(f'  asked {frame_rate_hz:g} Hz, camera can sustain '
              f'{cam.resulting_frame_rate:.1f} Hz')
        cam.assert_frame_rate_reachable(frame_rate_hz)

        tic = time.time()
        frames, meta = cam.record_burst(n_frames)
        wall = time.time() - tic
        check_burst_meta(meta)
        timing = burst_timing(meta, expected_rate_hz=frame_rate_hz)

        print(f'  recorded {len(frames)} frames in {wall:.2f} s, '
              f'{frames.nbytes / 1e6:.0f} MB, dtype {frames.dtype}')
        print(f"  period {timing['period_s_median'] * 1e3:.3f} ms "
              f"(asked {1e3 / frame_rate_hz:.3f}), jitter "
              f"{timing['period_s_std'] * 1e6:.0f} us, "
              f"{timing['n_dropped']} dropped")
        print(f"  timestamps look like nanoseconds: "
              f"{timing['timestamps_look_like_ns']}")
        print(f'  frame levels {frames.min()}-{frames.max()} of '
              f'{cam.saturation_level}, sensor peaked at '
              f"{meta[0]['raw_peak']} of {meta[0]['raw_full_scale']}")

        assert timing['n_dropped'] == 0, timing['dropped']
        assert timing['timestamps_look_like_ns'], timing
        # A host-side timestamp would carry the USB delivery jitter; a
        # camera-side one is as steady as the sensor's own clock. A tenth of a
        # frame period is far looser than a camera clock and far tighter than
        # USB scheduling, so it separates the two cases.
        assert timing['period_s_std'] < 0.1 / frame_rate_hz, (
            f"period jitter {timing['period_s_std'] * 1e6:.0f} us is too large "
            f"for a camera-side clock - the timestamps may be stamped on "
            f"arrival at the host, which would carry USB latency into the "
            f"synchronization")
        print('self-test passed')
        return {'describe': cam.describe(), 'timing': timing}
    finally:
        cam.set_binning(1)
        try:
            cam.set_roi_full()
        except Exception:
            pass
        cam.close()


if __name__ == '__main__':
    self_test()
