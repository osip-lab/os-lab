"""Manufacturer-independent camera adapter base.

The frontend camera box (static/boxes/camera.js) is shared by ALL cameras —
Basler, Ximea, synthetic, whatever comes next — and only speaks the generic
vocabulary implemented here: commands play / pause / snap / set_setting plus
the Gaussian-fit and ROI commands, events status / setting_applied /
fit_status / fit / guess / roi / error, and a describe() with a settings
schema.

Adding a camera brand therefore means:
  1. a pure device-layer module for the SDK (no GUI imports),
  2. a subclass of CameraAdapterBase implementing the hardware hooks below,
  3. one line in server.py DEVICE_TYPES and one line in app.js
     BOX_RENDERERS pointing the new type_name at createCameraBox.
Nothing else — the box, the fit pipeline and the command plumbing are
inherited. See ADDING_DEVICES.md.

Region of interest
------------------
An ROI crops what the camera delivers. Where the hardware can do it (Basler,
XIMEA: fewer rows read out means a higher frame rate and less USB bandwidth)
the crop is pushed into the camera — mix in StreamerROIMixin. Where it cannot
(the synthetic camera, or any SDK without the feature) the base crops every
frame here instead, so the two are indistinguishable to the browser apart
from the frame rate.

Once an ROI is set, the cropped frame *is* the image: fit results, the guess
circle and everything else the box draws are in pixels of the current frame,
not of the whole sensor. describe() carries the ROI's position on the sensor
('roi') and the uncropped size ('sensor_full') for the readouts that want
absolute coordinates.
"""

import time

from .base import DeviceAdapter
from .camera_fit import CameraFitMixin

CAMERA_COMMANDS = ['play', 'pause', 'snap', 'set_setting',
                   'fit_on', 'fit_off', 'set_guess', 'clear_guess',
                   'set_fit_threshold', 'set_roi', 'clear_roi']


class CameraAdapterBase(CameraFitMixin, DeviceAdapter):
    """Shared camera behavior; subclasses provide only the hardware hooks.

    Hooks to implement (besides type_name / display_name / list_available):
      _open()                     connect and start delivering frames
      _close()                    release the hardware; safe to call twice
      _play() / _pause()          resume / suspend the live stream
      _snap()                     deliver one frame while paused
      _apply_setting(name, value) apply a setting; must emit — possibly later,
                                  from the device thread — 'setting_applied'
                                  with the value the HARDWARE accepted
      _settings_schema()          list of setting dicts for describe()
      _sensor_shape()             [height, width] of the UNCROPPED sensor
                                  (the ROI is applied on top of it here)

    From the frame-producing thread, call _store_camera_frame(frame, display)
    with the full-resolution frame (fed to the fit) and a display-ready
    uint8 grayscale version (JPEG-streamed to the browsers). Both are cropped
    here when a software ROI is active.

    Class attributes to override where the hardware differs:
      LEVELS_MAX          full scale of the raw data (4095 for 12-bit)
      PIXEL_SIZE_MM       physical pixel pitch, for mm readouts in the box
      DISPLAY_DOWNSAMPLE  full-res -> display-stream reduction factor
    """

    LEVELS_MAX = 4095
    DISPLAY_DOWNSAMPLE = 1
    # how long restore_settings() waits for asynchronously-applied settings
    # to land, so the describe() right after open() shows the restored values
    # (override where _apply_setting hands off to a device thread)
    RESTORE_SETTLE_S = 0.0

    # True for subclasses that mix in StreamerROIMixin (or otherwise crop in
    # the camera); False means every frame is cropped here instead.
    HAS_HARDWARE_ROI = False
    # Smallest ROI worth having: the fit rebins by 4, and anything tinier is
    # a mis-drag rather than a region of interest.
    MIN_ROI_PX = 16

    def __init__(self, address):
        super().__init__(address)
        self._init_fit()
        self._playing = True
        # None, or {'x', 'y', 'width', 'height'} in UNCROPPED sensor pixels
        self._roi = None

    # ------------------------------------------------------ hardware hooks
    def _open(self):
        raise NotImplementedError

    def _close(self):
        raise NotImplementedError

    def _play(self):
        raise NotImplementedError

    def _pause(self):
        raise NotImplementedError

    def _snap(self):
        raise NotImplementedError

    def _apply_setting(self, name, value):
        raise NotImplementedError

    def _settings_schema(self):
        raise NotImplementedError

    def _sensor_shape(self):
        raise NotImplementedError

    def _label(self):
        return f'{self.display_name} — {self.address}'

    # ROI hooks — only for HAS_HARDWARE_ROI subclasses (see StreamerROIMixin).
    # Both apply asynchronously (the SDK is owned by the streaming thread) and
    # must call _roi_applied() from there with what the camera accepted.
    def _apply_hardware_roi(self, x, y, width, height):
        raise NotImplementedError

    def _clear_hardware_roi(self):
        raise NotImplementedError

    # -------------------------------------------------- shared implementation
    def open(self):
        self._open()
        self._playing = True

    def close(self):
        self._stop_fit()
        self._close()

    def _frame_shape(self):
        """[height, width] of the frames delivered NOW, i.e. after the ROI."""
        if self._roi is not None:
            return [self._roi['height'], self._roi['width']]
        return list(self._sensor_shape())

    def _store_camera_frame(self, frame, display):
        """Call per frame from the producing thread."""
        roi = self._roi
        if roi is not None and not self.HAS_HARDWARE_ROI:
            # software crop: the camera still delivers the whole sensor.
            # The ROI was snapped to DISPLAY_DOWNSAMPLE multiples when it was
            # set, so the display crop below is the same region exactly.
            x, y = roi['x'], roi['y']
            width, height = roi['width'], roi['height']
            frame = frame[y:y + height, x:x + width]
            step = self.DISPLAY_DOWNSAMPLE
            display = display[y // step:(y + height) // step,
                              x // step:(x + width) // step]
        self._store_display_frame(display)
        self._store_fit_frame(frame)

    # ------------------------------------------------------------------ ROI
    def _set_roi(self, x, y, width, height):
        """Crop to a rectangle given in pixels of the CURRENT (cropped) frame.

        That is what the browser can offer: the user drags the rectangle on
        the live image, which may already be an ROI. Translating it onto the
        sensor here keeps the frontend free of the absolute/relative
        distinction, and makes a second drag zoom further in.
        """
        sensor_h, sensor_w = self._sensor_shape()
        origin_x = self._roi['x'] if self._roi else 0
        origin_y = self._roi['y'] if self._roi else 0
        frame_h, frame_w = self._frame_shape()

        x = int(min(max(round(x), 0), frame_w - self.MIN_ROI_PX))
        y = int(min(max(round(y), 0), frame_h - self.MIN_ROI_PX))
        width = int(min(max(round(width), self.MIN_ROI_PX), frame_w - x))
        height = int(min(max(round(height), self.MIN_ROI_PX), frame_h - y))
        x += origin_x
        y += origin_y

        if self.HAS_HARDWARE_ROI:
            # the camera snaps to its own increments and reports back
            self._apply_hardware_roi(x, y, width, height)
            return {'ok': True}

        # Software crop: snap to the display downsampling grid so that the
        # cropped display frame is exactly the cropped full-res frame.
        step = self.DISPLAY_DOWNSAMPLE
        x -= x % step
        y -= y % step
        width -= width % step
        height -= height % step
        width = min(width, sensor_w - x)
        height = min(height, sensor_h - y)
        self._roi_applied({'x': x, 'y': y, 'width': width, 'height': height})
        return {'ok': True}

    def _clear_roi(self):
        if self.HAS_HARDWARE_ROI:
            self._clear_hardware_roi()
        else:
            self._roi_applied(None)
        return {'ok': True}

    def _roi_applied(self, roi):
        """Record the ROI the hardware (or the crop) actually ended up with.

        Called from whichever thread applied it — the streaming thread for a
        hardware ROI. Everything the box draws is in current-frame pixels, so
        the fit's initial guess is carried across with the crop rather than
        silently pointing at a different place.
        """
        previous = self._roi
        self._roi = roi
        shift_x = (previous['x'] if previous else 0) - (roi['x'] if roi else 0)
        shift_y = (previous['y'] if previous else 0) - (roi['y'] if roi else 0)
        if self._fit_guess is not None:
            height, width = self._frame_shape()
            moved = {'x_0': self._fit_guess['x_0'] + shift_x,
                     'y_0': self._fit_guess['y_0'] + shift_y,
                     'sigma': self._fit_guess['sigma']}
            # a crop can also leave the guess behind without moving it (same
            # corner, fewer pixels), so the bounds are checked either way
            inside = 0 <= moved['x_0'] < width and 0 <= moved['y_0'] < height
            moved = moved if inside else None
            if moved != self._fit_guess:
                self._fit_guess = moved
                if self._fit_loop is not None:
                    self._fit_loop.guess = moved
                self.emit({'type': 'guess', 'guess': moved})
        self.emit({'type': 'roi', 'roi': roi,
                   'sensor_shape': self._frame_shape(),
                   'sensor_full': list(self._sensor_shape())})

    # -------------------------------------------------- settings persistence
    def settings_snapshot(self):
        return {'settings': {setting['name']: setting['value']
                             for setting in self._settings_schema()},
                'roi': self._roi}

    def restore_settings(self, snapshot):
        applied = False
        for name, value in (snapshot.get('settings') or {}).items():
            try:
                self._apply_setting(name, float(value))
                applied = True
            except Exception:
                pass  # setting no longer exists / out of range: skip it
        roi = snapshot.get('roi')
        if roi:
            try:
                # stored in sensor pixels, and nothing is cropped yet right
                # after open(), so it goes in as a current-frame rectangle
                self._set_roi(roi['x'], roi['y'], roi['width'], roi['height'])
                applied = True
            except Exception:
                pass  # sensor/binning changed under the stored ROI: skip it
        if applied and self.RESTORE_SETTLE_S:
            time.sleep(self.RESTORE_SETTLE_S)

    def describe(self):
        return {'type': self.type_name,
                'label': self._label(),
                'frame_shape': [s // self.DISPLAY_DOWNSAMPLE
                                for s in self._frame_shape()],
                'sensor_shape': self._frame_shape(),
                'sensor_full': list(self._sensor_shape()),
                'roi': self._roi,
                'roi_hardware': self.HAS_HARDWARE_ROI,
                'levels_max': self.LEVELS_MAX,
                'commands': list(CAMERA_COMMANDS),
                'playing': self._playing,
                'settings': self._settings_schema(),
                **self.fit_describe()}

    def command(self, name, args):
        result = self.fit_command(name, args)
        if result is not None:
            return result
        if name == 'play':
            self._play()
            self._playing = True
            self.emit({'type': 'status', 'playing': True})
            return {'ok': True}
        if name == 'pause':
            self._pause()
            self._playing = False
            self.emit({'type': 'status', 'playing': False})
            return {'ok': True}
        if name == 'snap':
            self._snap()
            return {'ok': True}
        if name == 'set_setting':
            self._apply_setting(args['name'], float(args['value']))
            return {'ok': True}  # accepted value arrives as an event
        if name == 'set_roi':
            return self._set_roi(float(args['x']), float(args['y']),
                                 float(args['width']), float(args['height']))
        if name == 'clear_roi':
            return self._clear_roi()
        raise ValueError(f'unknown command {name!r}')


class StreamerROIMixin:
    """Hardware ROI for adapters driving a camera_core.CameraStreamer.

    Expects the two attributes those adapters already have: `self.streamer`
    and a device-layer camera with set_roi()/set_roi_full() (the signature
    both basler_cam and ximea_cam expose). Mix it in BEFORE CameraAdapterBase.

    The geometry nodes are locked while the camera is grabbing, so the change
    goes through submit_offline(): applied in the streaming thread with
    acquisition stopped, and reported back with the values the camera snapped
    to (its width/height/offset increments are never exactly what was asked).
    """

    HAS_HARDWARE_ROI = True

    def _apply_hardware_roi(self, x, y, width, height):
        def apply(camera):
            applied = camera.set_roi(width, height, x, y)
            self._roi_applied({'x': applied['offset_x'],
                               'y': applied['offset_y'],
                               'width': applied['width'],
                               'height': applied['height']})
            self._after_roi_applied(camera)

        self.streamer.submit_offline(apply)

    def _clear_hardware_roi(self):
        def apply(camera):
            camera.set_roi_full()
            self._roi_applied(None)
            self._after_roi_applied(camera)

        self.streamer.submit_offline(apply)

    def _after_roi_applied(self, camera):
        """Hook, called in the streaming thread after the ROI changed.

        Override where a limit moves with the ROI — fewer rows to read out
        raises the frame rate a camera can sustain, and the box must be told.
        """
