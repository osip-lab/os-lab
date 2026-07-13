"""Basler camera adapter: only the Basler-specific hardware hooks.

Everything camera-generic (commands, describe, fit pipeline, frame plumbing)
lives in CameraAdapterBase — this file wires it to the pure device layer in
basler_cam/basler_cameras.py. The pylon camera object is not thread-safe, so
setting changes are submitted to the streaming thread via
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

from .camera_base import CameraAdapterBase  # noqa: E402


class BaslerCameraAdapter(CameraAdapterBase):
    type_name = 'basler_camera'
    display_name = 'Basler camera'

    MAX_LIVE = 2  # USB3 bandwidth: more cameras drop frames
    DISPLAY_DOWNSAMPLE = 2  # 2048x2048 sensor -> 1024x1024 display stream
    LEVELS_MAX = 4095  # Mono12
    PIXEL_SIZE_MM = 5.5 / 1000.0  # acA2040 pixel pitch

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
        self._shape = [2048, 2048]  # actual value read at _open()

    def _label(self):
        return f'{self.display_name} — s/n {self.address}'

    # ------------------------------------------------------ hardware hooks
    def _open(self):
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
        self._shape = list(self.camera.frame_shape)
        self.streamer = CameraStreamer(self.camera,
                                       on_frame=self._on_frame,
                                       on_error=self._on_error)
        self.streamer.start()

    def _close(self):
        if self.streamer is not None:
            self.streamer.stop()
            self.streamer = None
        self.camera.close()
        with self._open_lock:
            self._open_serials.discard(self.address)

    def _play(self):
        self.streamer.resume()

    def _pause(self):
        self.streamer.pause()

    def _snap(self):
        self.streamer.snap()

    def _sensor_shape(self):
        return self._shape

    def _settings_schema(self):
        exposure_min, exposure_max = self._limits.get('exposure', (28, 1e7))
        gain_min, gain_max = self._limits.get('gain', (0.0, 23.0))
        return [{'name': 'exposure', 'label': 'exposure', 'unit': 'μs',
                 'min': exposure_min, 'max': exposure_max, 'decimals': 0,
                 'value': self._settings.get('exposure', 0.0)},
                {'name': 'gain', 'label': 'gain', 'unit': 'dB',
                 'min': gain_min, 'max': gain_max, 'decimals': 1,
                 'value': self._settings.get('gain', 0.0)}]

    def _apply_setting(self, name, value):
        if name not in ('exposure', 'gain'):
            raise ValueError(f'unknown setting {name!r}')

        def apply(camera):
            if name == 'exposure':
                camera.exposure_us = value
                accepted = camera.exposure_us
            else:
                camera.gain_db = value
                accepted = camera.gain_db
            self._settings[name] = accepted
            self.emit({'type': 'setting_applied', 'name': name,
                       'value': accepted})

        self.streamer.submit(apply)

    # ---------------------------------------------- streaming-thread callbacks
    def _on_frame(self, frame):
        small = frame[::self.DISPLAY_DOWNSAMPLE, ::self.DISPLAY_DOWNSAMPLE]
        self._store_camera_frame(frame, (small >> 4).astype(np.uint8))

    def _on_error(self, error):
        self.emit({'type': 'error', 'message': str(error)})
