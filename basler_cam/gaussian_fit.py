"""2D Gaussian fitting of beam profiles.

Pure computation plus a small threaded fit loop; no GUI imports, so any
interface (desktop GUI, web server) can use it. The fitting itself is
replicated from the well-tested routine in mode_position_capture_gui.py.
"""

import threading
import time

import numpy as np
from numba import njit, float64
from scipy.optimize import curve_fit
from scipy.ndimage import center_of_mass


@njit(float64[:](float64[:, :, ::0], float64, float64, float64, float64, float64, float64, float64),
      locals={'x': float64[:, ::0], 'y': float64[:, ::0], 'g': float64[:, ::1],
              'a': float64, 'b': float64, 'c': float64})
def gaussian2d(xy, ampl, xo, yo, sigma_x, sigma_y, theta, offset):
    x = xy[0]
    y = xy[1]
    a = np.cos(theta)**2 / (2 * sigma_x**2) + np.sin(theta)**2 / (2 * sigma_y**2)
    b = -np.sin(2 * theta) / (4 * sigma_x**2) + np.sin(2 * theta) / (4 * sigma_y**2)
    c = np.sin(theta)**2 / (2 * sigma_x**2) + np.cos(theta)**2 / (2 * sigma_y**2)
    g = offset + ampl * np.exp(-(a * (x - xo)**2 + 2 * b * (x - xo) * (y - yo) + c * (y - yo)**2))
    return np.ravel(g)


def rebin_image(img, factor):
    h, w = img.shape
    h_crop = (h // factor) * factor
    w_crop = (w // factor) * factor
    img_crop = img[:h_crop, :w_crop]
    return img_crop.reshape(h_crop // factor, factor, w_crop // factor, factor).mean(axis=(1, 3))


def fit_gaussian(arr, rebinning=1, manual_guess=None):
    """Fit a 2D Gaussian to `arr`; return (success, parameters).

    Parameters are in full-resolution pixel coordinates:
    amplitude, offset, angle, x_0, y_0, s_x, s_y, w_x, w_y, time.
    The camera measures intensity I = exp(-2 r^2 / w^2) while the fit uses
    exp(-r^2 / (2 sigma^2)), so the beam radius is w = 2 sigma.

    If `manual_guess` is given, it is a dict with 'x_0', 'y_0', 'sigma' in
    full-resolution pixels, overriding the automatic initial guess.
    """
    sy0, sx0 = np.shape(arr)
    arr = rebin_image(arr, rebinning)
    sy, sx = np.shape(arr)

    xx = np.linspace(0, sx - 1, sx)
    yy = np.linspace(0, sy - 1, sy)
    xx, yy = np.meshgrid(xx, yy)

    background = np.percentile(arr, 15)
    thresh = np.percentile(arr - background, (1 - 100 / sx0 / sy0) * 100)
    mh = (arr - background) >= thresh
    if not mh.any():
        # saturated / flat-top frame: fall back to the brightest pixels
        mh = arr >= arr.max()
    amplitude = np.mean(arr[mh]) - background
    y0, x0 = center_of_mass(np.array(mh, dtype=np.float64))
    mc = arr > amplitude / np.e**0.5
    radius = max((np.sum(mc) / np.pi) ** 0.5, 1)
    # a rebinned pixel at index i covers full-resolution pixels
    # [i * rebinning, (i + 1) * rebinning), so its center sits at
    # i * rebinning + (rebinning - 1) / 2
    bin_offset = (rebinning - 1) / 2
    if manual_guess is not None:
        x0 = (manual_guess['x_0'] - bin_offset) / rebinning
        y0 = (manual_guess['y_0'] - bin_offset) / rebinning
        radius = max(manual_guess['sigma'] / rebinning, 1)
    initial_guess = (amplitude, x0, y0, radius, radius, 0.0, background)

    tic = time.time()
    success = True
    try:
        p = curve_fit(gaussian2d, np.array((xx, yy)), arr.ravel(), p0=initial_guess,
                      bounds=((0.0, 0.0, 0.0, 0.0, 0.0, -np.pi / 4, 0.0),
                              (4095, sx, sy, np.inf, np.inf, np.pi / 4, 4095)),
                      ftol=1e-3, xtol=1e-3)
        pars = p[0]
    except RuntimeError:
        success = False
        pars = initial_guess
    dt = time.time() - tic

    parameters = {'amplitude': pars[0], 'offset': pars[6], 'angle': pars[5], 'time': dt,
                  'x_0': pars[1] * rebinning + bin_offset,
                  'y_0': pars[2] * rebinning + bin_offset,
                  's_x': pars[3] * rebinning, 's_y': pars[4] * rebinning,
                  'w_x': pars[3] * 2 * rebinning, 'w_y': pars[4] * 2 * rebinning}
    return success, parameters


class FitLoop:
    """Fit frames in a background thread, always the newest one only.

    submit() overwrites any not-yet-fitted frame, so when fitting is slower
    than the frame rate, intermediate frames are skipped and the loop never
    builds a backlog. Results are delivered through on_result(success, pars),
    called from the fitting thread. Frames whose signal is below `min_signal`
    counts above background are not fitted; on_result gets success=False and
    pars={'reason': 'low signal'}.
    """

    def __init__(self, on_result, rebinning=4, min_signal=50):
        self.on_result = on_result
        self.rebinning = rebinning
        self.min_signal = min_signal
        self._pending = None
        self._new = threading.Event()
        self._stopping = threading.Event()
        self._thread = None

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return
        self._stopping.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name='gauss-fit')
        self._thread.start()

    def stop(self):
        self._stopping.set()
        self._new.set()
        if self._thread is not None:
            self._thread.join(timeout=30)
            self._thread = None

    def submit(self, frame):
        self._pending = frame
        self._new.set()

    def _loop(self):
        while not self._stopping.is_set():
            if not self._new.wait(timeout=0.1):
                continue
            self._new.clear()
            if self._stopping.is_set():
                break
            frame = self._pending
            if frame is None:
                continue
            try:
                background = np.percentile(frame, 15)
                peak = np.percentile(frame, 99.95)
                if peak - background < self.min_signal:
                    self.on_result(False, {'reason': 'low signal'})
                    continue
                success, parameters = fit_gaussian(frame, rebinning=self.rebinning)
                self.on_result(success, parameters)
            except Exception as error:
                self.on_result(False, {'reason': str(error)})
