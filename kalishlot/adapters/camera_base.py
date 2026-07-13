"""Manufacturer-independent camera adapter base.

The frontend camera box (static/boxes/camera.js) is shared by ALL cameras —
Basler, Ximea, synthetic, whatever comes next — and only speaks the generic
vocabulary implemented here: commands play / pause / snap / set_setting plus
the Gaussian-fit commands, events status / setting_applied / fit_status /
fit / guess / error, and a describe() with a settings schema.

Adding a camera brand therefore means:
  1. a pure device-layer module for the SDK (no GUI imports),
  2. a subclass of CameraAdapterBase implementing the hardware hooks below,
  3. one line in server.py DEVICE_TYPES and one line in app.js
     BOX_RENDERERS pointing the new type_name at createCameraBox.
Nothing else — the box, the fit pipeline and the command plumbing are
inherited. See ADDING_DEVICES.md.
"""

from .base import DeviceAdapter
from .camera_fit import CameraFitMixin

CAMERA_COMMANDS = ['play', 'pause', 'snap', 'set_setting',
                   'fit_on', 'fit_off', 'set_guess', 'clear_guess']


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
      _sensor_shape()             [height, width] of full-resolution frames

    From the frame-producing thread, call _store_camera_frame(frame, display)
    with the full-resolution frame (fed to the fit) and a display-ready
    uint8 grayscale version (JPEG-streamed to the browsers).

    Class attributes to override where the hardware differs:
      LEVELS_MAX          full scale of the raw data (4095 for 12-bit)
      PIXEL_SIZE_MM       physical pixel pitch, for mm readouts in the box
      DISPLAY_DOWNSAMPLE  full-res -> display-stream reduction factor
    """

    LEVELS_MAX = 4095
    DISPLAY_DOWNSAMPLE = 1

    def __init__(self, address):
        super().__init__(address)
        self._init_fit()
        self._playing = True

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

    # -------------------------------------------------- shared implementation
    def open(self):
        self._open()
        self._playing = True

    def close(self):
        self._stop_fit()
        self._close()

    def _store_camera_frame(self, frame, display):
        """Call per frame from the producing thread."""
        self._store_display_frame(display)
        self._store_fit_frame(frame)

    def describe(self):
        sensor = list(self._sensor_shape())
        return {'type': self.type_name,
                'label': self._label(),
                'frame_shape': [s // self.DISPLAY_DOWNSAMPLE for s in sensor],
                'sensor_shape': sensor,
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
        raise ValueError(f'unknown command {name!r}')
