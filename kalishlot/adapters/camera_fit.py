"""Gaussian-fit support shared by camera adapters.

Mixin adding the fit vocabulary on top of DeviceAdapter: commands fit_on /
fit_off / set_guess / clear_guess, and 'fit' / 'fit_status' / 'guess' events.
The fitting itself runs in the device layer's FitLoop (newest-frame-only, so
a slow fit skips frames and never lags behind the camera).

Coordinates in commands, events and fit parameters are always full-resolution
sensor pixels; the frontend scales them onto the (possibly downsampled)
display stream using describe()['sensor_shape'].
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'basler_cam'))
from gaussian_fit import FitLoop, beam_brightness  # noqa: E402

CROSS_SECTION_STEP = 4  # send every 4th pixel of the row/column cuts
BRIGHTNESS_EMIT_INTERVAL_S = 0.2  # live readout rate for the trigger control


class CameraFitMixin:
    FIT_REBINNING = 4
    # physical pixel pitch for the mm readouts in the box; every camera
    # adapter overrides this with its sensor's value
    PIXEL_SIZE_MM = 1.0 / 1000.0

    def _init_fit(self):
        self._fit_loop = None
        self._fitting = False
        self._fit_guess = None  # {'x_0', 'y_0', 'sigma'} in sensor px, or None
        self._fit_threshold = 0.0  # counts above background; 0 = fit every frame
        self._last_frame = None  # newest full-resolution frame
        self._last_brightness_emit = 0.0

    def _store_fit_frame(self, frame):
        """Call from the frame-producing thread with each full-res frame.

        The trigger threshold is checked HERE, not in the fit loop: the loop
        keeps only the newest submitted frame, so with a blinking beam a
        below-threshold frame arriving right after a bright one would
        overwrite it before the fit thread wakes. Gating at submit time lets
        the bright frame sit in the loop until fitted, no matter how many
        dark frames follow.
        """
        self._last_frame = frame
        if not self._fitting or self._fit_loop is None:
            return
        brightness = beam_brightness(frame, self._fit_guess)
        now = time.monotonic()
        if now - self._last_brightness_emit >= BRIGHTNESS_EMIT_INTERVAL_S:
            self._last_brightness_emit = now
            self.emit({'type': 'brightness', 'value': round(brightness, 1)})
        if self._fit_threshold <= 0 or brightness >= self._fit_threshold:
            self._fit_loop.submit(frame)

    def fit_describe(self):
        """Merge into describe() so re-attaching viewers restore fit state."""
        return {'fitting': self._fitting,
                'guess': self._fit_guess,
                'fit_threshold': self._fit_threshold,
                'pixel_size_mm': self.PIXEL_SIZE_MM}

    def _stop_fit(self):
        self._fitting = False
        if self._fit_loop is not None:
            self._fit_loop.stop()
            self._fit_loop = None

    def _refit_now(self):
        """Fit the newest frame right away (also when paused)."""
        if self._fitting and self._fit_loop is not None \
                and self._last_frame is not None:
            self._fit_loop.submit(self._last_frame)

    def fit_command(self, name, args):
        """Handle a fit command; return None if `name` is not one of them."""
        if name == 'fit_on':
            self._fitting = True
            if self._fit_loop is None:
                self._fit_loop = FitLoop(on_result=self._on_fit_result,
                                         rebinning=self.FIT_REBINNING)
            self._fit_loop.guess = self._fit_guess
            self._fit_loop.start()
            self._refit_now()
            self.emit({'type': 'fit_status', 'enabled': True})
            return {'ok': True}
        if name == 'fit_off':
            self._fitting = False
            if self._fit_loop is not None:
                self._fit_loop.stop()
            self.emit({'type': 'fit_status', 'enabled': False})
            return {'ok': True}
        if name == 'set_guess':
            guess = {'x_0': float(args['x_0']), 'y_0': float(args['y_0']),
                     'sigma': max(float(args['sigma']), 1.0)}
            self._fit_guess = guess
            if self._fit_loop is not None:
                self._fit_loop.guess = guess
            self._refit_now()
            self.emit({'type': 'guess', 'guess': guess})
            return {'ok': True}
        if name == 'clear_guess':
            self._fit_guess = None
            if self._fit_loop is not None:
                self._fit_loop.guess = None
            self._refit_now()
            self.emit({'type': 'guess', 'guess': None})
            return {'ok': True}
        if name == 'set_fit_threshold':
            self._fit_threshold = max(float(args['value']), 0.0)
            self.emit({'type': 'fit_threshold', 'value': self._fit_threshold})
            return {'ok': True}
        return None

    # ------------------------------------------------------ FitLoop callback
    def _on_fit_result(self, success, parameters):
        if not self._fitting:
            return  # result of a frame submitted just before fit_off
        if not success:
            self.emit({'type': 'fit', 'success': False,
                       'reason': parameters.get('reason', 'did not converge')})
            return
        event = {'type': 'fit', 'success': True,
                 'params': {key: float(value)
                            for key, value in parameters.items()}}
        # row/column cuts through the fit center, for the cross-section plots
        frame = self._last_frame
        if frame is not None:
            height, width = frame.shape
            row = int(np.clip(round(parameters['y_0']), 0, height - 1))
            column = int(np.clip(round(parameters['x_0']), 0, width - 1))
            event['cross'] = {'step': CROSS_SECTION_STEP,
                              'row': frame[row, ::CROSS_SECTION_STEP].tolist(),
                              'col': frame[::CROSS_SECTION_STEP, column].tolist()}
        self.emit(event)
