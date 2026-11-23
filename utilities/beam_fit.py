"""2D Gaussian beam fitting, along the beam's own axes - numpy and scipy only.

Kept apart from utilities/utils.py (which pulls in matplotlib, cv2, the
clipboard and local_config) so the device server can import it: the kalishlot
camera boxes fit with it live. utils.py re-exports everything here, so
`from utilities.utils import fit_gaussian_beam` keeps working.

    python -m utilities.beam_fit     # the self-test
"""
import time

import numpy as np
from scipy.optimize import curve_fit


# --------------------------------------------------- 2D Gaussian beam fitting
# A beam's own fit, independent of any camera package: basler_cam/gaussian_fit.py
# and basler_cam/mode_position_capture_gui.py each carry a copy of an older
# routine that this one replaces for the offline analysis scripts. Three things
# are different, and each of them was a way of reading the old result wrong:
#
#   1. The widths come back named after the beam's own axes - 'minor' and
#      'major' - not after the image's. The old routine fitted sigma_x and
#      sigma_y along a rotated frame but kept the image's names for them, and
#      since its angle was bounded to +/- 45 deg, 'w_x' silently meant "the
#      principal width whose axis is nearest the image horizontal". Either of
#      the two could be the larger, so neither name said what it measured.
#   2. The angle is unbounded, and the initial guess comes from the image's own
#      second moments rather than from zero. The old +/- 45 deg bound left a
#      seam there: starting round and unturned, the optimizer had no gradient
#      to tell it which way to go, and a beam tilted within a degree or two of
#      45 deg could settle for a circle instead - a true 90 x 24 px spot at
#      44 deg came back 46 x 46. It took a clean frame to show (sensor noise
#      happens to break the symmetry and rescue the fit), which is exactly what
#      makes it the kind of thing to design out rather than watch for.
#   3. The rotation is written as an explicit change of axes, so the angle is
#      the direction of the major axis, measured from +x toward +y - the same
#      convention matplotlib's Ellipse takes. The old a/b/c form put the
#      sigma_x axis at MINUS theta, and the two overlays in this repo that drew
#      it disagreed about the sign.
#
# Distances are in pixels throughout; the caller multiplies by its own pixel
# size. As everywhere here, w is the 1/e^2 intensity radius, twice sigma.

def gaussian_2d(xy, amplitude, x_0, y_0, sigma_u, sigma_v, angle, offset):
    """A rotated 2D Gaussian, evaluated on the grid `xy` = (x, y) and raveled.

    `angle` [rad] is the direction of the sigma_u axis, measured from +x toward
    +y - that is, toward increasing ROW index on an image shown with
    imshow(origin='upper'). Rotating the coordinates explicitly, rather than
    folding the rotation into the usual a/b/c quadratic coefficients, is what
    keeps that sign readable: the very same angle, in degrees, is what
    matplotlib's Ellipse wants for an overlay that lies along the data.
    """
    x, y = xy[0], xy[1]
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    dx, dy = x - x_0, y - y_0
    u = dx * cos_a + dy * sin_a     # along the sigma_u axis
    v = -dx * sin_a + dy * cos_a    # across it
    return np.ravel(offset + amplitude
                    * np.exp(-0.5 * ((u / sigma_u) ** 2 + (v / sigma_v) ** 2)))


def rebin_image(image, factor):
    """Average `factor` x `factor` blocks of pixels (the edge remainder is dropped)."""
    if factor <= 1:
        return np.asarray(image, dtype=float)
    height, width = np.shape(image)
    h_crop, w_crop = (height // factor) * factor, (width // factor) * factor
    cropped = np.asarray(image, dtype=float)[:h_crop, :w_crop]
    return cropped.reshape(h_crop // factor, factor,
                           w_crop // factor, factor).mean(axis=(1, 3))


# The brightest pixels the moment guess is taken over, as a fraction of the
# peak above background. exp(-2) is the 1/e^2 level - the beam's own edge - so
# the guess is computed over the spot and not over the sensor around it.
_MOMENT_GUESS_LEVEL = np.exp(-2.0)
# Moments of a Gaussian truncated at that level are narrower than the Gaussian
# itself; this puts the sigma guess back on scale. It only has to be roughly
# right - the fit refines it - but starting on scale is what keeps a strongly
# elliptical beam from converging onto a round local minimum.
_MOMENT_GUESS_WIDENING = 1.52


def _moment_guess(image):
    """(x_0, y_0, sigma_major, sigma_minor, angle) from the image's own moments.

    The starting point for the fit. Taking the orientation from the data - and
    not from zero, as the older routine did - is what lets a steeply tilted
    beam be fitted at all: from a round, unrotated start the optimizer has no
    gradient telling it which way to turn, and settles for a circle.
    """
    image = np.asarray(image, dtype=float)
    background = np.percentile(image, 15)
    weights = np.clip(image - background, 0.0, None)
    peak = weights.max()
    if peak <= 0:  # a blank frame has no beam and no orientation
        height, width = image.shape
        return width / 2.0, height / 2.0, max(width, height) / 4.0, \
            max(width, height) / 4.0, 0.0
    weights = np.where(weights >= _MOMENT_GUESS_LEVEL * peak, weights, 0.0)

    yy, xx = np.mgrid[:image.shape[0], :image.shape[1]]
    total = weights.sum()
    x_0 = float((weights * xx).sum() / total)
    y_0 = float((weights * yy).sum() / total)
    dx, dy = xx - x_0, yy - y_0
    c_xx = float((weights * dx * dx).sum() / total)
    c_yy = float((weights * dy * dy).sum() / total)
    c_xy = float((weights * dx * dy).sum() / total)

    # eigh returns ascending eigenvalues, so the second eigenvector is the major axis
    values, vectors = np.linalg.eigh(np.array([[c_xx, c_xy], [c_xy, c_yy]]))
    values = np.clip(values, 1e-6, None)
    sigma_minor, sigma_major = (np.sqrt(values) * _MOMENT_GUESS_WIDENING)
    major = vectors[:, 1]
    angle = float(np.arctan2(major[1], major[0]))
    return x_0, y_0, float(sigma_major), float(sigma_minor), angle


def _canonical_axes(sigma_u, sigma_v, angle):
    """Sort a fitted (sigma_u, sigma_v, angle) into (major, minor, major angle).

    The fit is free to describe one ellipse two ways - swapping the two sigmas
    and turning by 90 deg - so the answer is only unambiguous once the larger
    axis has been named the major one and the reported angle made to be its
    direction. The result is wrapped into [-90, +90) deg, the half turn an
    ellipse's orientation actually lives in.
    """
    if sigma_v > sigma_u:
        sigma_u, sigma_v = sigma_v, sigma_u
        angle += np.pi / 2
    angle_deg = (np.degrees(angle) + 90.0) % 180.0 - 90.0
    return float(sigma_u), float(sigma_v), float(angle_deg)


def fit_gaussian_beam(image, rebinning=1, manual_guess=None):
    """Fit a rotated 2D Gaussian to `image`; return (success, parameters).

    The beam is reported by its own principal axes, in full-resolution pixels:

        x_0, y_0            centre
        sigma_major/_minor  the two principal standard deviations
        w_major, w_minor    the 1/e^2 intensity radii, twice those sigmas
        major_axis_deg      direction of the MAJOR axis, from +x toward +y
                            (increasing row index); in [-90, +90)
        minor_axis_deg      the same for the minor axis, 90 deg away
        ellipticity         w_minor / w_major, 1.0 for a round spot
        amplitude, offset   peak above background, and the background
        angle_rad           major_axis_deg in radians, for the model function
        time                seconds the fit took

    `major_axis_deg` is ready to hand to matplotlib's Ellipse alongside
    width=2*w_major, height=2*w_minor - no sign to flip.

    With `rebinning` > 1 the fit runs on blocks of that many pixels averaged
    together (much faster, and harmless for a beam far wider than a block);
    every reported length is converted back to full-resolution pixels, the
    centre included.

    `manual_guess` is an optional {'x_0', 'y_0', 'sigma'} in full-resolution
    pixels - a circle the user dragged - which replaces the centre and width
    of the automatic guess. The orientation still comes from the image.

    `success` is False when the optimizer did not converge; `parameters` then
    holds the initial guess, so a caller can still draw something.
    """
    image = np.asarray(image, dtype=float)
    binned = rebin_image(image, rebinning)
    height, width = binned.shape
    yy, xx = np.mgrid[:height, :width]
    xx, yy = xx.astype(float), yy.astype(float)

    # A rebinned pixel at index i covers full-resolution pixels
    # [i * rebinning, (i + 1) * rebinning), so its centre sits at
    # i * rebinning + (rebinning - 1) / 2.
    bin_offset = (rebinning - 1) / 2.0

    x_0, y_0, sigma_major, sigma_minor, angle = _moment_guess(binned)
    if manual_guess is not None:
        x_0 = (manual_guess['x_0'] - bin_offset) / rebinning
        y_0 = (manual_guess['y_0'] - bin_offset) / rebinning
        # The drag gives one radius - the beam's width across its widest way -
        # so it replaces the scale, while the shape and the orientation stay
        # with the moments. Starting from a perfect circle instead, as an
        # earlier version of this did, leaves the angle with no gradient to
        # follow, and an elliptical beam then sometimes converges onto a
        # wrongly turned minimum.
        aspect = sigma_minor / sigma_major if sigma_major else 1.0
        sigma_major = max(manual_guess['sigma'] / rebinning, 1.0)
        sigma_minor = max(sigma_major * aspect, 1e-2)

    offset = float(np.percentile(binned, 15))
    amplitude = max(float(binned.max()) - offset, 1e-9)
    initial = (amplitude, x_0, y_0, sigma_major, sigma_minor, angle, offset)

    span = float(binned.max() - binned.min()) or 1.0
    # Bounds taken from the data, never from a fixed full-scale value: an
    # amplitude capped at some bit depth is a cap the caller then has to scale
    # its frames around. The angle spans a full turn - twice the half turn an
    # ellipse needs - so no orientation sits against a wall.
    lower = (0.0, -width, -height, 1e-3, 1e-3, -np.pi, binned.min() - span)
    upper = (10 * span, 2 * width, 2 * height, 10 * width, 10 * height,
             np.pi, binned.max() + span)

    started = time.time()
    success = True
    try:
        fitted, _ = curve_fit(gaussian_2d, np.array((xx, yy)), binned.ravel(),
                              p0=initial, bounds=(lower, upper),
                              ftol=1e-3, xtol=1e-3, maxfev=20000)
    except (RuntimeError, ValueError):
        success = False
        fitted = initial
    elapsed = time.time() - started

    amplitude, x_0, y_0, sigma_u, sigma_v, angle, offset = fitted
    sigma_major, sigma_minor, major_axis_deg = _canonical_axes(sigma_u, sigma_v, angle)
    sigma_major *= rebinning
    sigma_minor *= rebinning
    return success, {
        'amplitude': float(amplitude), 'offset': float(offset),
        'x_0': float(x_0) * rebinning + bin_offset,
        'y_0': float(y_0) * rebinning + bin_offset,
        'sigma_major': sigma_major, 'sigma_minor': sigma_minor,
        'w_major': 2 * sigma_major, 'w_minor': 2 * sigma_minor,
        'major_axis_deg': major_axis_deg,
        'minor_axis_deg': (major_axis_deg + 90.0 + 90.0) % 180.0 - 90.0,
        'angle_rad': np.radians(major_axis_deg),
        'ellipticity': sigma_minor / sigma_major if sigma_major else float('nan'),
        'time': elapsed,
    }


def gaussian_beam_image(parameters, shape):
    """Evaluate a fit_gaussian_beam() result over a full-resolution `shape` grid.

    For overlaying the fitted model on the frame it came from - a cross-section
    through it, or a residual. `parameters` is the dict the fit returns, whose
    lengths are already in full-resolution pixels.
    """
    height, width = shape
    yy, xx = np.mgrid[:height, :width]
    model = gaussian_2d(np.array((xx.astype(float), yy.astype(float))),
                        parameters['amplitude'], parameters['x_0'],
                        parameters['y_0'], parameters['sigma_major'],
                        parameters['sigma_minor'], parameters['angle_rad'],
                        parameters['offset'])
    return model.reshape(height, width)


# ------------------------------------------------------------------ self-test
def _fit_gaussian_beam_self_test():
    """Synthetic beams through fit_gaussian_beam - no camera, no files.

        python -m utilities.beam_fit
    """
    size = 241
    xx, yy = np.meshgrid(np.arange(size, dtype=float), np.arange(size, dtype=float))

    def beam(sigma_major, sigma_minor, major_deg, x_0=118.4, y_0=125.7, noise=5.0,
             seed=0):
        grid = np.array((xx, yy))
        image = gaussian_2d(grid, 800.0, x_0, y_0, sigma_major, sigma_minor,
                            np.radians(major_deg), 20.0).reshape(size, size)
        return image + np.random.default_rng(seed).normal(0.0, noise, image.shape)

    # --- every orientation, the old routine's +/- 45 deg seam included -------
    for major_deg in range(-89, 90, 7):
        success, pars = fit_gaussian_beam(beam(40.0, 11.0, major_deg), rebinning=2)
        assert success, major_deg
        wanted = (major_deg + 90) % 180 - 90
        turned = abs(((pars['major_axis_deg'] - wanted + 90) % 180) - 90)
        assert turned < 3.0, (major_deg, pars['major_axis_deg'])
        assert abs(pars['w_major'] - 80.0) < 4.0, (major_deg, pars['w_major'])
        assert abs(pars['w_minor'] - 22.0) < 4.0, (major_deg, pars['w_minor'])
        # the major axis is the wider one, by construction of the report
        assert pars['w_major'] >= pars['w_minor']
    print('fit ok: every tilt from -89 to +89 deg recovered, widths and angle')

    # --- the centre survives rebinning, offsets and all --------------------
    for rebinning in (1, 2, 4, 8):
        _, pars = fit_gaussian_beam(beam(40.0, 11.0, 35.0), rebinning=rebinning)
        assert abs(pars['x_0'] - 118.4) < 0.5, (rebinning, pars['x_0'])
        assert abs(pars['y_0'] - 125.7) < 0.5, (rebinning, pars['y_0'])
    print('centre ok: unmoved by rebinning 1, 2, 4 and 8')

    # --- a dragged guess sets the scale, not the shape ---------------------
    _, pars = fit_gaussian_beam(beam(40.0, 11.0, -50.0), rebinning=2,
                                manual_guess={'x_0': 120.0, 'y_0': 124.0,
                                              'sigma': 36.0})
    assert abs(((pars['major_axis_deg'] + 50.0 + 90) % 180) - 90) < 3.0, pars
    assert abs(pars['w_major'] - 80.0) < 4.0, pars
    print('manual guess ok: a circle dragged over a tilted beam still fits it')

    # --- a round beam is round, whatever angle comes back ------------------
    _, pars = fit_gaussian_beam(beam(30.0, 30.0, 0.0), rebinning=2)
    assert abs(pars['ellipticity'] - 1.0) < 0.02, pars['ellipticity']
    print(f"round beam ok: ellipticity {pars['ellipticity']:.4f}")

    # --- the angle is the one matplotlib's Ellipse takes, unflipped --------
    from matplotlib.patches import Ellipse
    _, pars = fit_gaussian_beam(beam(40.0, 11.0, -37.0), rebinning=2)
    ellipse = Ellipse((pars['x_0'], pars['y_0']), 2 * pars['w_major'],
                      2 * pars['w_minor'], angle=pars['major_axis_deg'])
    tip = ellipse.get_patch_transform().transform([(1.0, 0.0)])[0]
    drawn = np.degrees(np.arctan2(tip[1] - pars['y_0'], tip[0] - pars['x_0']))
    assert abs((((drawn + 37.0) + 90) % 180) - 90) < 3.0, drawn
    print(f"overlay ok: Ellipse(angle=major_axis_deg) lies at "
          f"{(drawn + 90) % 180 - 90:+.1f} deg, the beam at -37 deg")

    # --- the model can be put back on the frame it came from ---------------
    image = beam(40.0, 11.0, 22.0, noise=5.0)
    _, pars = fit_gaussian_beam(image, rebinning=2)
    model = gaussian_beam_image(pars, image.shape)
    assert model.shape == image.shape
    residual = float(np.sqrt(((model - image) ** 2).mean()))
    assert residual < 7.0, residual   # the noise it was given was 5 counts
    print(f'model ok: rms residual {residual:.2f} counts against 5.0 of noise')

    # --- a blank frame reports something rather than raising ---------------
    success, pars = fit_gaussian_beam(np.full((48, 48), 7.0))
    assert {'w_minor', 'w_major', 'major_axis_deg'} <= set(pars), sorted(pars)
    print('blank frame ok: parameters returned, nothing raised')
    print('fit_gaussian_beam self-test passed')


if __name__ == '__main__':
    _fit_gaussian_beam_self_test()
