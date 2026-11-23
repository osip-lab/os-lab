"""Beam centre of mass, shared by camera adapters.

Mixin adding the centre-of-mass vocabulary on top of DeviceAdapter: commands
com_on / com_off / set_com_cut, the 'com' event, and 'com' / 'com_cut' in
describe(). A fast alternative to the Gaussian fit when only the position
matters, e.g. for steering spots onto markers with motors: milliseconds per
frame instead of the fit's fraction of a second, and no model to fail.

The estimate (center_of_mass below):
  background  the median of a subsample of the frame (the beam is small
              against the frame, so the median is the dark level)
  peak        the maximum of a 4x4-binned copy, so a hot pixel cannot set it
              (nor place the window: a real spot wins the binned maximum)
  cut         background + cut_fraction * (peak - background), but at least
              NOISE_CUT_SIGMAS noise std above the background
  window      a square around the peak, WINDOW_RADII times the radius of the
              area above the cut; doubled while the spot touches its edge
  weights     (pixel - cut) where positive, 0 elsewhere, inside the window
The centre is the weighted mean pixel position, the size its weighted std.
The noise floor and the window keep stray noise pixels and hot pixels far
from the spot out of the sum: across a megapixel, a 4-sigma cut alone still
lets dozens through, and each pulls the centre by its distance.
Without a cut, the noise of the whole frame pulls the centre towards the
middle of the image, the more so the smaller the spot; the cut is what makes
the centre of mass usable. It needs the spot to be the brightest thing in the
frame (or in the ROI: crop to the spot when there is other light).

Validity: no spot (peak less than MIN_SNR noise std above the background),
saturated (the raw maximum at full scale: the centre is still measured but
the weights are clipped), clipped (light above the cut touches the frame
edge: the centre is pulled inwards).

Coordinates: x, y and sigma_x, sigma_y are pixels of the current frame (for
drawing, like the fit); sensor_x, sensor_y are uncropped sensor pixels, the
markers' units (camera_markers.py), for comparing with a target.

The work runs on its own thread, newest frame only (like the fit's FitLoop),
so a slow computation skips frames and never lags behind the camera, and at
most COM_MAX_RATE_HZ, so two fast cameras do not take two cores. The last
results are kept with their frame times, for in-process consumers that need
fresh positions after a move (com_samples); browsers get a 'com' event a few
times a second. Scripts get the same results through the command com_samples
{n, since, timeout} (alignment/com_measure.py uses it for the alignment loop).
"""

import collections
import threading
import time

import numpy as np

COM_DEFAULT_CUT = 0.2         # fraction of the peak height above background
COM_MAX_RATE_HZ = 30.0
COM_EMIT_INTERVAL_S = 0.1     # 10 Hz to the browsers
COM_HISTORY = 300             # results kept for com_samples (10 s at 30 Hz)
COM_SAMPLES_MAX_TIMEOUT_S = 10.0   # the command blocks an HTTP request: stay under its 15 s
MIN_SNR = 8.0                 # peak height in noise std for "there is a spot"
NOISE_CUT_SIGMAS = 5.0        # the cut is never closer to the background than this
WINDOW_RADII = 4.0            # sum within +-4 spot radii of the peak ...
MIN_WINDOW_HALF_PX = 16       # ... but at least +-16 px
PEAK_BINNING = 4              # the peak is found on a 4x4-binned copy (hot pixels / 16)
MIN_SPOT_PIXELS = 3           # fewer pixels above the cut = not a spot


def center_of_mass(frame, cut_fraction=COM_DEFAULT_CUT, levels_max=None):
    """Thresholded centre of mass of a 2-D frame. Returns a dict:
    valid, reason, x, y, sigma_x, sigma_y (frame px), background, peak, cut,
    noise, counts (summed weights), n_pixels, saturated, clipped."""
    frame = np.asarray(frame)
    height, width = frame.shape
    sub = frame[::4, ::4].astype(np.float32)
    background = float(np.median(sub))
    noise = float(1.4826 * np.median(np.abs(sub - background))) or 1.0
    b = PEAK_BINNING
    hb, wb = (height // b) * b, (width // b) * b
    if hb and wb:
        binned = frame[:hb, :wb].astype(np.float32).reshape(hb // b, b, wb // b, b).mean(axis=(1, 3))
        peak = float(binned.max())
    else:
        peak = float(frame.max())
    raw_max = float(frame.max())
    result = {'valid': False, 'reason': '', 'x': float('nan'), 'y': float('nan'),
              'sigma_x': float('nan'), 'sigma_y': float('nan'),
              'background': background, 'peak': peak, 'noise': noise,
              'cut': float('nan'), 'counts': 0.0, 'n_pixels': 0,
              'saturated': levels_max is not None and raw_max >= levels_max,
              'clipped': False}
    contrast = peak - background
    if contrast < MIN_SNR * noise:
        result['reason'] = f'no spot (peak {contrast:.0f} above background, noise {noise:.1f})'
        return result
    cut = background + max(float(cut_fraction) * contrast, NOISE_CUT_SIGMAS * noise)

    # window around the peak, sized from the spot's area above the cut
    if hb and wb:
        iy, ix = np.unravel_index(int(np.argmax(binned)), binned.shape)
        cy, cx = b * iy + (b - 1) / 2, b * ix + (b - 1) / 2
        radius = np.sqrt(b * b * np.count_nonzero(binned > cut) / np.pi)
    else:
        cy, cx, radius = height / 2, width / 2, max(height, width)
    half = max(WINDOW_RADII * radius, MIN_WINDOW_HALF_PX)
    while True:
        y0, y1 = max(int(cy - half), 0), min(int(np.ceil(cy + half)) + 1, height)
        x0, x1 = max(int(cx - half), 0), min(int(np.ceil(cx + half)) + 1, width)
        weights = frame[y0:y1, x0:x1].astype(np.float32) - np.float32(cut)
        np.maximum(weights, 0, out=weights)
        cols = weights.sum(axis=0, dtype=np.float64)       # profile along x
        rows = weights.sum(axis=1, dtype=np.float64)       # profile along y
        at_window_edge = ((cols[0] > 0 and x0 > 0) or (cols[-1] > 0 and x1 < width)
                          or (rows[0] > 0 and y0 > 0) or (rows[-1] > 0 and y1 < height))
        if not at_window_edge or (x0 == 0 and y0 == 0 and x1 == width and y1 == height):
            break
        half *= 2                                          # the spot is bigger: widen
    total = float(cols.sum())
    n_pixels = int(np.count_nonzero(weights))
    if n_pixels < MIN_SPOT_PIXELS:
        result['reason'] = (f'no spot ({n_pixels} pixel{"s" * (n_pixels != 1)} above the cut: '
                            f'a hot pixel?)')
        return result
    xs = np.arange(x0, x1, dtype=np.float64)
    ys = np.arange(y0, y1, dtype=np.float64)
    x = float(cols @ xs / total)
    y = float(rows @ ys / total)
    result.update(
        valid=True, x=x, y=y,
        sigma_x=float(np.sqrt(max(cols @ (xs - x) ** 2 / total, 0.0))),
        sigma_y=float(np.sqrt(max(rows @ (ys - y) ** 2 / total, 0.0))),
        cut=cut, counts=total, n_pixels=n_pixels,
        clipped=bool((cols[0] > 0 and x0 == 0) or (cols[-1] > 0 and x1 == width)
                     or (rows[0] > 0 and y0 == 0) or (rows[-1] > 0 and y1 == height)))
    if result['saturated']:
        result['reason'] = 'saturated'
    elif result['clipped']:
        result['reason'] = 'spot touches the frame edge'
    return result


def _json_safe(result):
    """A result for JSON: NaN/inf -> None, numpy scalars -> Python."""
    out = {}
    for key, value in result.items():
        if isinstance(value, (np.floating, float)):
            value = float(value)
            out[key] = value if np.isfinite(value) else None
        elif isinstance(value, (np.integer, np.bool_)):
            out[key] = value.item()
        else:
            out[key] = value
    return out


class ComLoop:
    """Newest-frame-only worker: submit() from the camera thread, results go
    to on_result(result) on the worker thread."""

    def __init__(self, compute, on_result, max_rate_hz=COM_MAX_RATE_HZ):
        self._compute = compute
        self._on_result = on_result
        self._period = 1.0 / max_rate_hz
        self._pending = None
        self._cond = threading.Condition()
        self._running = False
        self._thread = None

    def start(self):
        with self._cond:
            if self._running:
                return
            self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True, name='com-loop')
        self._thread.start()

    def stop(self):
        with self._cond:
            self._running = False
            self._pending = None
            self._cond.notify_all()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=2)
        self._thread = None

    def submit(self, item):
        with self._cond:
            if self._running:
                self._pending = item
                self._cond.notify_all()

    def _run(self):
        while True:
            with self._cond:
                while self._running and self._pending is None:
                    self._cond.wait()
                if not self._running:
                    return
                item, self._pending = self._pending, None
            tic = time.monotonic()
            try:
                self._on_result(self._compute(item))
            except Exception as error:              # never kill the worker
                self._on_result({'valid': False, 'reason': f'error: {error}'})
            rest = self._period - (time.monotonic() - tic)
            if rest > 0:
                time.sleep(rest)


class CameraComMixin:
    """Needs LEVELS_MAX, emit() and _frame_meta() from the camera base."""

    def _init_com(self):
        self._com_on = False
        self._com_cut = COM_DEFAULT_CUT
        self._com_loop = None
        self._com_history = collections.deque(maxlen=COM_HISTORY)
        self._com_cond = threading.Condition()
        self._com_last_emit = 0.0

    # ------------------------------------------------------------- pipeline
    def _store_com_frame(self, frame, roi):
        """Call from the frame-producing thread with each frame that passed
        the trigger (the same frames the fit sees), and the ROI it was cut to."""
        if self._com_on and self._com_loop is not None:
            meta = self._frame_meta()
            self._com_loop.submit((frame, roi, meta.get('host_time_s', time.time()),
                                   self._com_cut))

    def _compute_com(self, item):
        frame, roi, frame_time, cut = item
        result = center_of_mass(frame, cut, getattr(self, 'LEVELS_MAX', None))
        ox, oy = (roi['x'], roi['y']) if roi else (0, 0)
        result.update(sensor_x=result['x'] + ox, sensor_y=result['y'] + oy,
                      frame_time_s=frame_time, cut_fraction=cut)
        return result

    def _on_com_result(self, result):
        if not self._com_on:
            return              # a frame submitted just before com_off
        result.setdefault('frame_time_s', time.time())
        with self._com_cond:
            self._com_history.append(result)
            self._com_cond.notify_all()
        now = time.monotonic()
        if now - self._com_last_emit >= COM_EMIT_INTERVAL_S:
            self._com_last_emit = now
            self.emit({'type': 'com', **{k: (round(v, 3) if isinstance(v, float) else v)
                                         for k, v in result.items()}})

    def _stop_com(self):
        self._com_on = False
        if self._com_loop is not None:
            self._com_loop.stop()
            self._com_loop = None

    # --------------------------------------------------- in-process readers
    def com_samples(self, n=1, since=None, timeout=5.0):
        """The next `n` centre-of-mass results of frames taken after host time
        `since` (default: now), waiting up to `timeout` s. Invalid results are
        included (check 'valid'). Raises TimeoutError with what arrived."""
        if not self._com_on:
            raise RuntimeError('the centre of mass is off on this camera')
        since = time.time() if since is None else since
        deadline = time.monotonic() + timeout
        with self._com_cond:
            while True:
                fresh = [r for r in self._com_history if r['frame_time_s'] > since]
                if len(fresh) >= n:
                    return fresh[:n]
                left = deadline - time.monotonic()
                if left <= 0 or not self._com_on:
                    raise TimeoutError(f'{len(fresh)} of {n} centre-of-mass results in '
                                       f'{timeout:g} s')
                self._com_cond.wait(left)

    # ------------------------------------------------------------- commands
    def com_describe(self):
        return {'com': self._com_on, 'com_cut': self._com_cut}

    def com_command(self, name, args):
        """Handle a centre-of-mass command; None if `name` is not one of them."""
        if name == 'com_on':
            if self._com_loop is None:
                self._com_loop = ComLoop(self._compute_com, self._on_com_result)
            self._com_on = True
            self._com_loop.start()
            self.emit({'type': 'com_status', 'enabled': True, 'cut': self._com_cut})
            return {'ok': True}
        if name == 'com_off':
            self._stop_com()
            with self._com_cond:
                self._com_history.clear()
                self._com_cond.notify_all()
            self.emit({'type': 'com_status', 'enabled': False, 'cut': self._com_cut})
            return {'ok': True}
        if name == 'com_samples':
            n = int(args.get('n', 1))
            if not 1 <= n <= COM_HISTORY:
                raise ValueError(f'n must be 1 to {COM_HISTORY}')
            timeout = min(float(args.get('timeout', 5.0)), COM_SAMPLES_MAX_TIMEOUT_S)
            since = args.get('since')
            try:
                samples = self.com_samples(n, None if since is None else float(since), timeout)
            except (RuntimeError, TimeoutError) as error:
                raise ValueError(str(error)) from None
            return {'samples': [_json_safe(r) for r in samples]}
        if name == 'set_com_cut':
            value = float(args['value'])
            if not 0.0 <= value < 1.0:
                raise ValueError('the cut is a fraction of the peak height, 0 to <1')
            self._com_cut = value
            self.emit({'type': 'com_status', 'enabled': self._com_on, 'cut': value})
            return {'ok': True}
        return None
