"""Basler camera adapter: wraps the pure device layer in basler_cam/.

All hardware access goes through BaslerCamera + CameraStreamer
(basler_cam/basler_cameras.py). The pylon camera object is not thread-safe,
so setting changes are submitted to the streaming thread via
CameraStreamer.submit() and the accepted (clamped) value comes back to the
browser as a 'setting_applied' event.
"""

import sys
import threading
from pathlib import Path

import numpy as np

# the device layer lives at the repo root, outside kalishlot/
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'basler_cam'))
from basler_cameras import BaslerCamera, CameraStreamer  # noqa: E402

from .base import DeviceAdapter  # noqa: E402
from .camera_fit import CameraFitMixin  # noqa: E402


class BaslerCameraAdapter(CameraFitMixin, DeviceAdapter):
    type_name = 'basler_camera'
    display_name = 'Basler camera'

    MAX_LIVE = 2  # USB3 bandwidth: more cameras drop frames
    DOWNSAMPLE = 2  # 2048x2048 sensor -> 1024x1024 display stream

    _open_serials = set()
    _open_lock = threading.Lock()

    @staticmethod
    def list_available():
        return [{'address': d['serial_number'],
                 'label': f"{d['model']} s/n {d['serial_number']}"}
                for d in BaslerCamera.list_devices()]

    def __init__(self, address):
        super().__init__(address)
        self.camera = BaslerCamera(address)
        self.streamer = None
        self._settings = {}
        self._limits = {}
        self._playing = True
        self._sensor_shape = [2048, 2048]  # actual value read at open()
        self._init_fit()

    # ------------------------------------------------------------ lifecycle
    def open(self):
        with self._open_lock:
            if len(self._open_serials) >= self.MAX_LIVE:
                raise RuntimeError(
                    f'at most {self.MAX_LIVE} Basler cameras can stream at '
                    f'once (USB3 bandwidth); close another camera box first')
            try:
                self.camera.open()
            except Exception as error:
                raise RuntimeError(
                    f'{error} — if the camera is open in another program '
                    f'(e.g. the pylon Viewer or the desktop GUI), close it '
                    f'there first') from error
            self._open_serials.add(self.address)

        # read settings/limits/shape before streaming starts; afterwards all
        # camera access must go through streamer.submit()
        self._settings = {'exposure': self.camera.exposure_us,
                          'gain': self.camera.gain_db}
        self._limits = {'exposure': self.camera.exposure_limits_us,
                        'gain': self.camera.gain_limits_db}
        self._sensor_shape = list(self.camera.frame_shape)
        self.streamer = CameraStreamer(self.camera,
                                       on_frame=self._on_frame,
                                       on_error=self._on_error)
        self.streamer.start()
        self._playing = True

    def close(self):
        self._stop_fit()
        if self.streamer is not None:
            self.streamer.stop()
            self.streamer = None
        self.camera.close()
        with self._open_lock:
            self._open_serials.discard(self.address)

    def describe(self):
        exposure_min, exposure_max = self._limits.get('exposure', (28, 1e7))
        gain_min, gain_max = self._limits.get('gain', (0.0, 23.0))
        return {'type': self.type_name,
                'label': f'{self.display_name} — s/n {self.address}',
                'frame_shape': [s // self.DOWNSAMPLE for s in self._sensor_shape],
                'sensor_shape': list(self._sensor_shape),
                'commands': ['play', 'pause', 'snap', 'set_setting',
                             'fit_on', 'fit_off', 'set_guess', 'clear_guess'],
                'playing': self._playing,
                **self.fit_describe(),
                'settings': [
                    {'name': 'exposure', 'label': 'exposure', 'unit': 'μs',
                     'min': exposure_min, 'max': exposure_max, 'decimals': 0,
                     'value': self._settings.get('exposure', 0.0)},
                    {'name': 'gain', 'label': 'gain', 'unit': 'dB',
                     'min': gain_min, 'max': gain_max, 'decimals': 1,
                     'value': self._settings.get('gain', 0.0)},
                ]}

    # ------------------------------------------------------------- commands
    def command(self, name, args):
        result = self.fit_command(name, args)
        if result is not None:
            return result
        if name == 'play':
            self.streamer.resume()
            self._playing = True
            self.emit({'type': 'status', 'playing': True})
            return {'ok': True}
        if name == 'pause':
            self.streamer.pause()
            self._playing = False
            self.emit({'type': 'status', 'playing': False})
            return {'ok': True}
        if name == 'snap':
            self.streamer.snap()
            return {'ok': True}
        if name == 'set_setting':
            setting = args['name']
            if setting not in ('exposure', 'gain'):
                raise ValueError(f'unknown setting {setting!r}')
            value = float(args['value'])

            def apply(camera):
                if setting == 'exposure':
                    camera.exposure_us = value
                    accepted = camera.exposure_us
                else:
                    camera.gain_db = value
                    accepted = camera.gain_db
                self._settings[setting] = accepted
                self.emit({'type': 'setting_applied', 'name': setting,
                           'value': accepted})

            self.streamer.submit(apply)
            return {'ok': True}  # accepted value arrives as an event
        raise ValueError(f'unknown command {name!r}')

    # ---------------------------------------------- streaming-thread callbacks
    def _on_frame(self, frame):
        small = frame[::self.DOWNSAMPLE, ::self.DOWNSAMPLE]
        self._store_display_frame((small >> 4).astype(np.uint8))
        self._store_fit_frame(frame)

    def _on_error(self, error):
        self.emit({'type': 'error', 'message': str(error)})
