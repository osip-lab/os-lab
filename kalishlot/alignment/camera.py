"""Camera-side metrics: how Gaussian a frame is, and is a frame a first-order mode.

A frame integrates its whole exposure (70 ms in the 2026-10 sessions, ~15% of
an FSR), so besides the target mode it holds the background and any other mode
that passed during the exposure. `exposure_purity` measures that from the scope.

Scores are scale-free on purpose: R^2/MSE improve with brightness, so a bright
but distorted mode could beat a dim clean one.
- overlap  = (sum I*G)^2 / (sum I^2 * sum G^2) inside a 2.5-sigma ellipse, with
  I the background-subtracted frame and G the fitted Gaussian (1 = perfect).
- ellipticity = min(sigma) / max(sigma) (1 = round; astigmatism lowers it).
"""

import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'basler_cam'))
from gaussian_fit import fit_gaussian, gaussian2d  # noqa: E402
from scipy.optimize import curve_fit  # noqa: E402

FIT_FULL_SCALE = 4095.0      # fit_gaussian's amplitude bound (12-bit); frames are scaled to it
ROI_SIGMAS = 2.5             # overlap is computed inside this ellipse
MIN_SNR = 8.0                # fitted amplitude / background noise for a valid score
FIRST_ORDER_MAX_OVERLAP = 0.8   # a frame matching the 0th-order Gaussian less than this
                                # is "not fundamental-like" (calibrate in validate_marks)
FIRST_ORDER_MIN_LIKENESS = 0.65  # HG10/LG01 template overlap needed to confirm a first order;
                                 # 2026-10-12 sessions: +s family 0.73-0.76, -s family 0.54-0.66


@dataclass
class GaussianScore:
    valid: bool
    reason: str
    overlap: float = float('nan')
    ellipticity: float = float('nan')
    angle_rad: float = float('nan')
    x0: float = float('nan')
    y0: float = float('nan')
    sigma_x: float = float('nan')
    sigma_y: float = float('nan')
    snr: float = float('nan')

    def to_dict(self):
        return asdict(self)


def _prepare(frame, mask=None):
    f = np.asarray(frame, dtype=float)
    if mask is not None:
        f = np.where(mask, f, np.median(f[~mask]) if (~mask).any() else 0.0)
    scale = FIT_FULL_SCALE / max(float(f.max()), 1e-12)
    return f * scale


def _gaussian(shape, p):
    yy, xx = np.indices(shape, dtype=float)
    th = p['angle']
    sx, sy = max(p['s_x'], 1e-6), max(p['s_y'], 1e-6)
    a = np.cos(th) ** 2 / (2 * sx ** 2) + np.sin(th) ** 2 / (2 * sy ** 2)
    b = -np.sin(2 * th) / (4 * sx ** 2) + np.sin(2 * th) / (4 * sy ** 2)
    c = np.sin(th) ** 2 / (2 * sx ** 2) + np.cos(th) ** 2 / (2 * sy ** 2)
    dx, dy = xx - p['x_0'], yy - p['y_0']
    q = a * dx ** 2 + 2 * b * dx * dy + c * dy ** 2
    return np.exp(-q), q


def overlap_with(frame_bg_sub, params):
    """Normalized overlap of a background-subtracted frame with the Gaussian
    `params` (fit_gaussian's dict), inside the ROI ellipse."""
    g, q = _gaussian(frame_bg_sub.shape, params)
    roi = q <= ROI_SIGMAS ** 2 / 2
    if roi.sum() < 10:
        return float('nan')
    i = frame_bg_sub[roi]
    gg = g[roi]
    den = float(np.sum(i * i) * np.sum(gg * gg))
    return float(np.sum(i * gg) ** 2 / den) if den > 0 else float('nan')


def refine_fit(f, p, crop_sigmas=4.0):
    """Tighten fit_gaussian's result (its tolerances are loose, tuned for a
    live display) with a full-tolerance fit on a crop around the beam."""
    h, w = f.shape
    r = crop_sigmas * max(p['s_x'], p['s_y'], 2.0)
    x0, x1 = int(max(p['x_0'] - r, 0)), int(min(p['x_0'] + r + 1, w))
    y0, y1 = int(max(p['y_0'] - r, 0)), int(min(p['y_0'] + r + 1, h))
    crop = f[y0:y1, x0:x1]
    if crop.size < 50:
        return False, p
    yy, xx = np.indices(crop.shape, dtype=float)
    p0 = [p['amplitude'], p['x_0'] - x0, p['y_0'] - y0, p['s_x'], p['s_y'],
          float(np.clip(p['angle'], -np.pi / 4, np.pi / 4)), p['offset']]
    try:
        popt, _ = curve_fit(gaussian2d, np.array((xx, yy)), crop.ravel(), p0=p0,
                            bounds=((0, 0, 0, 0.5, 0.5, -np.pi / 2, -np.inf),
                                    (np.inf, crop.shape[1], crop.shape[0], np.inf, np.inf,
                                     np.pi / 2, np.inf)), maxfev=4000)
    except (RuntimeError, ValueError):
        return False, p
    a, xc, yc, sx, sy, th, off = popt
    return True, {'amplitude': a, 'offset': off, 'angle': th, 'x_0': xc + x0,
                  'y_0': yc + y0, 's_x': sx, 's_y': sy}


def gaussian_score(frame, mask=None, rebinning=2):
    """Fit a 2D Gaussian to a frame and score how Gaussian it is."""
    f = _prepare(frame, mask)
    ok, p = fit_gaussian(f, rebinning=rebinning)
    if ok:
        ok, p = refine_fit(f, p)
    if not ok:
        return GaussianScore(False, 'fit did not converge')
    return _score(f, p)


FAST_CROP_SIGMAS = 6.0      # fit ROI half-size, in moment widths (which underestimate sigma;
                            # 4 clipped the tails: widths 5% small, noise taken from the tails)
FAST_TARGET_PX = 48         # the fit ROI is rebinned to about this many pixels across
FAST_MIN_HALF_PX = 16


def _moments(f):
    """Rough centre and width from intensity moments (background = median,
    pixels under 20% of the peak ignored). Biased by stray light, so only a
    starting point: ~7 px off on a live frame where the fit agrees to 0.3 px."""
    g = f - np.median(f)
    g[g < 0.2 * g.max()] = 0.0
    total = float(g.sum())
    if total <= 0:
        return None
    ys, xs = np.indices(g.shape, dtype=float)
    cx, cy = float((g * xs).sum() / total), float((g * ys).sum() / total)
    sx = float(np.sqrt((g * (xs - cx) ** 2).sum() / total))
    sy = float(np.sqrt((g * (ys - cy) ** 2).sum() / total))
    return cx, cy, max(sx, sy, 2.0)


def fast_gaussian_score(frame, mask=None, crop_sigmas=FAST_CROP_SIGMAS,
                        target_px=FAST_TARGET_PX):
    """gaussian_score, ~10-50x faster: fit only a region around the spot, rebinned.

    The fit ROI is +-crop_sigmas moment widths around a moments estimate of the
    spot; it is rebinned to ~target_px across, fitted (fit_gaussian, then the
    full-tolerance refine_fit on the small rebinned crop), and scored on the
    full-resolution crop. Centre and widths are in pixels of `frame`. Live
    frame (2026-10-19): 2.7 s for gaussian_score; rebinning alone 0.37 s."""
    f = _prepare(frame, mask)
    h, w = f.shape
    mom = _moments(f)
    if mom is None:
        return GaussianScore(False, 'empty frame')
    cx, cy, s = mom
    half = max(crop_sigmas * s, FAST_MIN_HALF_PX)
    x0, x1 = int(max(cx - half, 0)), int(min(cx + half + 1, w))
    y0, y1 = int(max(cy - half, 0)), int(min(cy + half + 1, h))
    crop = f[y0:y1, x0:x1]
    b = max(1, int(round(max(crop.shape) / target_px)))
    small = crop[:crop.shape[0] // b * b, :crop.shape[1] // b * b]
    small = small.reshape(small.shape[0] // b, b, small.shape[1] // b, b).mean(axis=(1, 3))
    ok, p = fit_gaussian(small)
    if ok:
        ok, p = refine_fit(small, p)
    if not ok:
        return GaussianScore(False, 'fit did not converge')
    # rebinned pixel i covers crop pixels [i*b, (i+1)*b): centre at i*b + (b-1)/2
    p = dict(p, x_0=p['x_0'] * b + (b - 1) / 2, y_0=p['y_0'] * b + (b - 1) / 2,
             s_x=p['s_x'] * b, s_y=p['s_y'] * b)
    score = _score(crop, p)
    score.x0 += x0
    score.y0 += y0
    if not (0 <= score.x0 < w and 0 <= score.y0 < h) and score.valid:
        score.valid, score.reason = False, 'centre outside the frame'
    return score


def _score(f, p):
    """GaussianScore of fitted parameters `p` on the (prepared) image `f`."""
    h, w = f.shape
    border = f.copy()
    g, q = _gaussian(f.shape, p)
    outside = q > (2 * ROI_SIGMAS) ** 2 / 2
    noise = float(np.std(border[outside])) if outside.sum() > 50 else float(np.std(f))
    snr = p['amplitude'] / max(noise, 1e-9)
    base = dict(angle_rad=float(p['angle']), x0=float(p['x_0']), y0=float(p['y_0']),
                sigma_x=float(p['s_x']), sigma_y=float(p['s_y']), snr=float(snr))
    if not (0 <= p['x_0'] < w and 0 <= p['y_0'] < h):
        return GaussianScore(False, 'centre outside the frame', **base)
    if min(p['s_x'], p['s_y']) < 1.5 or max(p['s_x'], p['s_y']) > max(h, w) / 2:
        return GaussianScore(False, 'implausible width', **base)
    if snr < MIN_SNR:
        return GaussianScore(False, f'too dim (snr {snr:.1f})', **base)
    ov = overlap_with(f - p['offset'], p)
    ell = min(p['s_x'], p['s_y']) / max(p['s_x'], p['s_y'])
    return GaussianScore(True, '', ov, float(ell), **base)


def fundamental_likeness(frame, reference, mask=None):
    """Overlap of a frame with a reference 0th-order Gaussian (a GaussianScore).

    A first-order mode (two lobes / ring) overlaps poorly with the 0th-order
    Gaussian at the same place. Returns nan when the reference is invalid."""
    if not reference.valid:
        return float('nan')
    f = _prepare(frame, mask)
    params = {'x_0': reference.x0, 'y_0': reference.y0, 's_x': reference.sigma_x,
              's_y': reference.sigma_y, 'angle': reference.angle_rad}
    return overlap_with(f - np.percentile(f, 15), params)


def _envelope_coords(shape, ref):
    """Coordinates in units of the reference beam's sigmas, in its own axes."""
    yy, xx = np.indices(shape, dtype=float)
    dx, dy = xx - ref.x0, yy - ref.y0
    c, s = np.cos(ref.angle_rad), np.sin(ref.angle_rad)
    u = (c * dx + s * dy) / max(ref.sigma_x, 1e-6)
    v = (-s * dx + c * dy) / max(ref.sigma_y, 1e-6)
    return u, v


FIRST_ORDER_ANGLES = 12      # HG10 template orientations tried over 180 degrees


def first_order_likeness(frame, reference, mask=None):
    """Best overlap of a frame with a first-order mode built on the 0th-order
    envelope: HG10 at any orientation, or the LG01 donut (degenerate pair).

    Intensity of HG10 along unit direction n: (n.r)^2 exp(-r^2/2) in sigma
    units (the camera sees exp(-r^2/(2 sigma^2)) for the 0th order, and the
    first-order field carries one extra factor of the coordinate).
    Returns (overlap, best template name)."""
    if not reference.valid:
        return float('nan'), ''
    f = _prepare(frame, mask)
    f = f - np.percentile(f, 15)
    u, v = _envelope_coords(f.shape, reference)
    r2 = u ** 2 + v ** 2
    env = np.exp(-r2 / 2)
    roi = r2 <= 3.5 ** 2
    if roi.sum() < 20:
        return float('nan'), ''
    i = f[roi]

    def ov(template):
        g = template[roi]
        den = float(np.sum(i * i) * np.sum(g * g))
        return float(np.sum(i * g) ** 2 / den) if den > 0 else float('nan')

    best, name = ov(r2 * env), 'LG01'
    for k in range(FIRST_ORDER_ANGLES):
        a = np.pi * k / FIRST_ORDER_ANGLES
        o = ov((np.cos(a) * u + np.sin(a) * v) ** 2 * env)
        if o > best:
            best, name = o, f'HG10@{np.degrees(a):.0f}deg'
    return best, name


def exposure_purity(t, y, window, t_target, half_width, baseline=None):
    """Share of the scope signal inside a frame's exposure window that comes
    from the target peak (t_target +- half_width). 1 means the frame saw only
    that mode; low values mean other modes/background dominate the frame."""
    t = np.asarray(t)
    y = np.asarray(y, dtype=float)
    m = (t >= window[0]) & (t < window[1])
    if not m.any():
        return float('nan')
    b = np.percentile(y[m], 5) if baseline is None else baseline
    v = np.clip(y[m] - b, 0, None)
    total = float(v.sum())
    target = float(v[np.abs(t[m] - t_target) <= half_width].sum())
    return target / total if total > 0 else float('nan')


def _self_test():
    yy, xx = np.indices((120, 140), dtype=float)
    rng = np.random.default_rng(1)
    round_beam = 1000 * np.exp(-((xx - 70) ** 2 + (yy - 60) ** 2) / (2 * 8 ** 2)) + 50
    s = gaussian_score(round_beam + rng.normal(0, 5, round_beam.shape))
    assert s.valid and s.overlap > 0.98 and s.ellipticity > 0.95, s
    astig = 1000 * np.exp(-((xx - 70) ** 2 / (2 * 14 ** 2) + (yy - 60) ** 2 / (2 * 6 ** 2))) + 50
    s2 = gaussian_score(astig + rng.normal(0, 5, astig.shape))
    assert s2.valid and s2.ellipticity < 0.5, s2
    # HG10: two lobes; poor overlap with the round reference
    hg10 = 1000 * ((xx - 70) / 8) ** 2 * np.exp(-((xx - 70) ** 2 + (yy - 60) ** 2) / (2 * 8 ** 2)) + 50
    assert fundamental_likeness(hg10, s) < FIRST_ORDER_MAX_OVERLAP
    assert fundamental_likeness(round_beam, s) > 0.95
    fo, name = first_order_likeness(hg10, s)
    assert fo > 0.9 and name.startswith('HG10'), (fo, name)
    hg11 = 1000 * ((xx - 70) * (yy - 60) / 64) ** 2 * np.exp(-((xx - 70) ** 2 + (yy - 60) ** 2) / (2 * 8 ** 2)) + 50
    assert first_order_likeness(hg11, s)[0] < fo - 0.2
    dim = 50 + rng.normal(0, 5, round_beam.shape)
    assert not gaussian_score(dim).valid
    t = np.arange(0, 0.1, 1e-4)
    y = 1.0 / (1 + ((t - 0.05) / 1e-3) ** 2)
    assert exposure_purity(t, y, (0.0, 0.1), 0.05, 0.01) > 0.9
    # fast fit: same centre and width as the full fit on a live-sized frame with stray light
    rng = np.random.default_rng(3)
    yy, xx = np.indices((546, 720), dtype=float)
    spot = 40 + 600 * np.exp(-((xx - 351.3) ** 2 / (2 * 50 ** 2) + (yy - 236.8) ** 2 / (2 * 62 ** 2)))
    spot += 80 * np.exp(-((xx - 600) ** 2 + (yy - 80) ** 2) / (2 * 15 ** 2))   # stray light
    spot += rng.normal(0, 4, spot.shape)
    fast, full = fast_gaussian_score(spot), gaussian_score(spot)
    assert fast.valid and full.valid, (fast, full)
    assert abs(fast.x0 - 351.3) < 0.5 and abs(fast.y0 - 236.8) < 0.5, fast
    assert abs(fast.sigma_x - 50) < 1.5 and abs(fast.sigma_y - 62) < 1.5, fast
    assert abs(fast.x0 - full.x0) < 0.5 and abs(fast.overlap - full.overlap) < 0.01, (fast, full)
    print('camera self-test OK')


if __name__ == '__main__':
    _self_test()
