"""XIMEA camera adapter: only the XIMEA-specific hardware hooks.

Everything camera-generic (commands, describe, fit pipeline, frame plumbing)
lives in CameraAdapterBase — this file wires it to the pure device layer in
ximea_cam/ximea_cameras.py. xiAPI is not thread-safe, so setting changes are
submitted to the streaming thread via CameraStreamer.submit() and the accepted
(clamped) value comes back to the browser as a 'setting_applied' event.

The browser side is the same manufacturer-agnostic camera box the Basler uses;
this adapter differs from that one in three measured particulars: the sensor is
10-bit (full scale 1023, so the display shift is 2 bits, not 4), the frame rate
is a settable parameter rather than a consequence, and the camera has no
firmware binning.

Cropping to an ROI is what the frame rate responds to most strongly here (the
sensor is read row by row), so _after_roi_applied re-reads the rate and its
ceiling and pushes both to the box.
"""

import sys
import threading
from pathlib import Path

import numpy as np

# the device layer lives at the repo root, outside kalishlot/. Importing the
# module flat rather than as ximea_cam.ximea_cameras also avoids executing that
# package's __init__, which pulls in PyQt6.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'ximea_cam'))
from ximea_cameras import CameraStreamer, XimeaCamera  # noqa: E402

from .camera_base import CameraAdapterBase, StreamerROIMixin  # noqa: E402
from .exposure_rate import exposure_for_rate, rate_for_exposure  # noqa: E402


class XimeaCameraAdapter(StreamerROIMixin, CameraAdapterBase):
    type_name = 'ximea_camera'
    display_name = 'XIMEA camera'

    MAX_LIVE = 2  # USB3 bandwidth: more cameras drop frames
    DISPLAY_DOWNSAMPLE = 2  # 2048x2048 sensor -> 1024x1024 display stream
    LEVELS_MAX = 1023  # 10-bit sensor; XI_MONO16 is the container, not the depth
    PIXEL_SIZE_MM = 5.5 / 1000.0  # MQ042MG (CMV4000) pixel pitch
    RESTORE_SETTLE_S = 0.5  # settings apply on the streamer thread

    # 10 bits of data in a 16-bit word: shift by 2 to get the display byte,
    # where the Basler's 12-bit data shifts by 4.
    DISPLAY_SHIFT = 2

    _open_serials = set()
    _open_lock = threading.Lock()

    @staticmethod
    def list_available():
        return [{'address': d['serial_number'],
                 'label': f"{d['model']} s/n {d['serial_number']}"}
                for d in XimeaCamera.list_devices()]

    def __init__(self, address):
        super().__init__(address)
        self.camera = XimeaCamera(address)
        self.streamer = None
        self._settings = {}
        self._limits = {}
        self._shape = [2048, 2048]  # uncropped sensor; read at _open()

    def _label(self):
        return f'{self.display_name} — s/n {self.address}'

    # ------------------------------------------------------ hardware hooks
    def _open(self):
        with self._open_lock:
            if len(self._open_serials) >= self.MAX_LIVE:
                raise RuntimeError(
                    f'at most {self.MAX_LIVE} XIMEA cameras can stream at '
                    f'once (USB3 bandwidth); close another camera box first')
            try:
                self.camera.open()
            except Exception as error:
                raise RuntimeError(
                    f'{error} — if the camera is open in another program '
                    f'(e.g. xiCamTool or the desktop GUI), close it '
                    f'there first') from error
            self._open_serials.add(self.address)

        # read settings/limits/shape before streaming starts; afterwards all
        # camera access must go through streamer.submit()
        self._settings = {'exposure': self.camera.exposure_us,
                          'gain': self.camera.gain_db,
                          'framerate': self.camera.frame_rate_hz}
        self._limits = {'exposure': self.camera.exposure_limits_us,
                        'gain': self.camera.gain_limits_db,
                        'framerate': self.camera.frame_rate_limits_hz}
        # Start uncropped whatever the camera was left set to (xiCamTool or an
        # earlier capture script may have left an ROI behind): 'no ROI' in the
        # box must mean the whole sensor. A stored ROI of our own is
        # re-applied a moment later by restore_settings(). Not guarded: a
        # camera whose geometry cannot be normalised would go on to describe
        # itself as delivering frames it is not, so failing to open (with the
        # reason shown to the user) is the honest outcome.
        self.camera.set_roi_full()
        width, height = self.camera.max_frame_size
        self._shape = [height, width]
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
        exposure_min, exposure_max = self._limits.get('exposure', (107, 1e6))
        gain_min, gain_max = self._limits.get('gain', (-1.5, 6.0))
        rate_min, rate_max = self._limits.get('framerate', (1.0, 100.0))
        return [{'name': 'exposure', 'label': 'exposure', 'unit': 'μs',
                 'min': exposure_min, 'max': exposure_max, 'decimals': 0,
                 'value': self._settings.get('exposure', 0.0)},
                {'name': 'gain', 'label': 'gain', 'unit': 'dB',
                 'min': gain_min, 'max': gain_max, 'decimals': 1,
                 'value': self._settings.get('gain', 0.0)},
                # Unlike the Basler, this camera is paced from its own clock,
                # so the frame rate is asked for rather than inferred - and a
                # live view left at the 10 Hz default looks broken.
                {'name': 'framerate', 'label': 'frame rate', 'unit': 'Hz',
                 'min': rate_min, 'max': rate_max, 'decimals': 1,
                 'value': self._settings.get('framerate', 0.0)}]

    def restore_settings(self, snapshot):
        # the saved exposure and rate go back exactly as they were, not one
        # of them re-derived from the other (see _apply_setting)
        self._restoring = True
        try:
            super().restore_settings(snapshot)
        finally:
            self._restoring = False

    def _apply_setting(self, name, value):
        if name not in ('exposure', 'gain', 'framerate'):
            raise ValueError(f'unknown setting {name!r}')
        # From the box, exposure and frame rate follow each other: a rate
        # gets the longest exposure that keeps it, an exposure the fastest
        # rate it allows (exposure_rate.py). Read now, not on the streaming
        # thread, which runs the change later.
        coupled = not getattr(self, '_restoring', False)

        def apply(camera):
            if name == 'gain':
                camera.gain_db = value
                self._settings['gain'] = camera.gain_db
                self.emit({'type': 'setting_applied', 'name': 'gain',
                           'value': self._settings['gain']})
                return
            if name == 'exposure':
                if coupled:
                    rate_for_exposure(camera, value)
                else:
                    camera.exposure_us = value
            elif coupled:
                exposure_for_rate(camera, value)
            else:
                camera.frame_rate_hz = value
            # either one moves the other (and the ceiling on the rate), so
            # both go back to the box
            self._limits['framerate'] = camera.frame_rate_limits_hz
            self._settings['exposure'] = camera.exposure_us
            self._settings['framerate'] = camera.frame_rate_hz
            self.emit({'type': 'setting_applied', 'name': 'exposure',
                       'value': self._settings['exposure']})
            self.emit({'type': 'setting_applied', 'name': 'framerate',
                       'value': self._settings['framerate'],
                       'max': self._limits['framerate'][1]})

        self.streamer.submit(apply)

    def _after_roi_applied(self, camera):
        # Fewer rows to read out means a higher rate the camera can sustain;
        # the rate itself is re-read because the camera drops it when the old
        # value no longer fits the new geometry.
        self._limits['framerate'] = camera.frame_rate_limits_hz
        self._settings['framerate'] = camera.frame_rate_hz
        self.emit({'type': 'setting_applied', 'name': 'framerate',
                   'value': self._settings['framerate']})

    # ---------------------------------------------- streaming-thread callbacks
    def _on_frame(self, frame):
        small = frame[::self.DISPLAY_DOWNSAMPLE, ::self.DISPLAY_DOWNSAMPLE]
        self._store_camera_frame(frame,
                                 (small >> self.DISPLAY_SHIFT).astype(np.uint8))

    def _on_error(self, error):
        self.emit({'type': 'error', 'message': str(error)})
