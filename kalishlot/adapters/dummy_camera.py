"""Synthetic camera adapter: a moving Gaussian beam with noise.

Lets the whole web GUI (registry, boxes, streaming, commands, settings, fit)
be developed and tested without touching lab hardware — and doubles as the
minimal reference implementation of the CameraAdapterBase hardware hooks.
"""

import threading
import time

import numpy as np

from .camera_base import CameraAdapterBase


class DummyCameraAdapter(CameraAdapterBase):
    type_name = 'dummy_camera'
    display_name = 'Dummy camera (synthetic beam)'

    SHAPE = (1024, 1024)
    PIXEL_SIZE_MM = 5.5 / 1000.0  # pretend to be the lab's Basler sensor
    FRAME_RATE = 10.0  # Hz
    ORBIT_PERIOD_S = 20.0  # the beam center circles the frame center

    @staticmethod
    def list_available():
        return [{'address': 'synthetic-0', 'label': 'synthetic beam 0'},
                {'address': 'synthetic-1', 'label': 'synthetic beam 1'}]

    def __init__(self, address):
        super().__init__(address)
        self.settings = {'exposure': 3000.0, 'gain': 0.0}
        self._running = threading.Event()
        self._stopping = threading.Event()
        self._thread = None
        height, width = self.SHAPE
        self._xx, self._yy = np.meshgrid(np.arange(width, dtype=np.float64),
                                         np.arange(height, dtype=np.float64))
        self._rng = np.random.default_rng()

    # ------------------------------------------------------ hardware hooks
    def _open(self):
        if self._thread is not None and self._thread.is_alive():
            return
        self._stopping.clear()
        self._running.set()
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name=f'dummy-cam-{self.address}')
        self._thread.start()

    def _close(self):
        self._stopping.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    def _play(self):
        self._running.set()

    def _pause(self):
        self._running.clear()

    def _snap(self):
        if not self._running.is_set():
            self._make_frame()

    def _sensor_shape(self):
        return list(self.SHAPE)

    def _settings_schema(self):
        return [{'name': 'exposure', 'label': 'exposure', 'unit': 'μs',
                 'min': 28, 'max': 10_000_000, 'decimals': 0,
                 'value': self.settings['exposure']},
                {'name': 'gain', 'label': 'gain', 'unit': 'dB',
                 'min': 0.0, 'max': 23.0, 'decimals': 1,
                 'value': self.settings['gain']}]

    def _apply_setting(self, name, value):
        if name not in self.settings:
            raise ValueError(f'unknown setting {name!r}')
        self.settings[name] = float(value)
        self.emit({'type': 'setting_applied', 'name': name,
                   'value': self.settings[name]})

    # --------------------------------------------------------------- frames
    def _make_frame(self):
        height, width = self.SHAPE
        phase = 2 * np.pi * time.time() / self.ORBIT_PERIOD_S
        x0 = width / 2 + 0.25 * width * np.cos(phase)
        y0 = height / 2 + 0.25 * height * np.sin(phase)
        sigma = 60.0
        # brightness responds to the dummy exposure/gain like a real camera
        amplitude = 1500.0 * self.settings['exposure'] / 3000.0
        amplitude *= 10 ** (self.settings['gain'] / 20.0)
        amplitude = min(amplitude, 4000.0)
        frame = (100.0 + amplitude * np.exp(-(((self._xx - x0) ** 2)
                                              + ((self._yy - y0) ** 2))
                                            / (2 * sigma ** 2)))
        frame += self._rng.normal(0, 30, self.SHAPE)
        frame = np.clip(frame, 0, 4095).astype(np.uint16)
        self._store_camera_frame(frame, (frame >> 4).astype(np.uint8))

    def _loop(self):
        period = 1.0 / self.FRAME_RATE
        while not self._stopping.is_set():
            tic = time.time()
            if self._running.is_set():
                self._make_frame()
            time.sleep(max(0.0, period - (time.time() - tic)))
