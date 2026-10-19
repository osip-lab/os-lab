"""Wrong-mode detection: the first order taller than the 0th order.

    python alignment/test_wrong_mode.py        (from kalishlot/)

Seen live on 2026-10-19 (Y piezo above ~47 V): the tilt made the first order
the tallest peak, every 0th-order pick was a first order, and the position
jumped by 35 px. A synthetic capture reproduces it: a rising sweep, the first
order at s = +0.31 and 1.5x taller, Gaussian frames at the 0th-order blinks,
two-lobe (HG10) frames at the first-order blinks.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alignment.metrics import MetricsConfig, evaluate_capture  # noqa: E402

RATE = 20_000.0
FSR_S = 0.8
S_FIRST = 0.31
GAMMA_S = 0.001
PERIOD_S, EXPOSURE_S = 0.0088, 0.008
H0, H1 = 0.4, 0.6                       # first order taller than the 0th
SHAPE = (120, 160)
CENTRE0 = (80.0, 60.0)                  # (x, y) of the 0th-order spot


def _lorentz(t, t0, h):
    return h / (1 + ((t - t0) / GAMMA_S) ** 2)


def synthetic_capture(h0=H0, h1=H1, distorted_zeroth=False):
    rng = np.random.default_rng(7)
    t = np.arange(0, 3.4, 1 / RATE)
    zeroth = 0.25 + FSR_S * np.arange(4)
    first = zeroth + S_FIRST * FSR_S                # rising sweep: first order after the 0th
    y = 0.01 + sum(_lorentz(t, a, h0) for a in zeroth) + sum(_lorentz(t, b, h1) for b in first)
    y += rng.normal(0, 0.002, t.size)
    aux = -1.0 + 0.15 * t                           # monotonic rising ramp, no turnaround

    yy, xx = np.indices(SHAPE, dtype=float)
    gauss = np.exp(-((xx - CENTRE0[0]) ** 2 / (2 * 12 ** 2) + (yy - CENTRE0[1]) ** 2 / (2 * 15 ** 2)))
    u = (xx - 90) / 12
    hg10 = u ** 2 * np.exp(-(u ** 2 + ((yy - 50) / 15) ** 2))
    hg10 /= hg10.max()
    if distorted_zeroth:    # far from paraxial: a 0th order a Gaussian fits only poorly
        u0 = (xx - CENTRE0[0]) / 12
        gauss = u0 ** 2 * np.exp(-(u0 ** 2 + ((yy - CENTRE0[1]) / 15) ** 2))
        gauss /= gauss.max()
    starts = np.arange(0, t[-1] - EXPOSURE_S, PERIOD_S)
    windows = np.column_stack([starts, starts + EXPOSURE_S])
    frames = np.empty((len(starts),) + SHAPE, dtype=np.float32)
    for i, (a, b) in enumerate(windows):
        sel = (t >= a) & (t < b)
        e0 = sum(_lorentz(t[sel], c, h0).sum() for c in zeroth)
        e1 = sum(_lorentz(t[sel], c, h1).sum() for c in first)
        frames[i] = 20 + 0.5 * (e0 * gauss + e1 * hg10) + rng.normal(0, 1.0, SHAPE)
    return t, y, aux, frames, windows


def _metrics(cfg, **capture):
    return evaluate_capture(*synthetic_capture(**capture), cfg=cfg)


def test_without_guard_picks_the_first_order():
    res = _metrics(MetricsConfig(fast=True))
    cy = res.metrics['centroid_y']
    assert cy.valid and abs(cy.value - 50) < 3, cy        # the impostor's centre: the failure seen live


def test_guard_rejects_it():
    res = _metrics(MetricsConfig(fast=True, min_fundamental_overlap=0.85))
    assert not res.metrics['centroid_y'].valid, res.metrics['centroid_y']
    assert not res.metrics['intensity'].valid             # invalid, not the first order's height


def test_recovery_finds_the_0th_order():
    res = _metrics(MetricsConfig(fast=True, min_fundamental_overlap=0.85,
                                 recover_first_order_s=S_FIRST))
    m = res.metrics
    assert m['centroid_y'].valid and abs(m['centroid_y'].value - CENTRE0[1]) < 1, m['centroid_y']
    assert abs(m['centroid_x'].value - CENTRE0[0]) < 1, m['centroid_x']
    assert m['intensity'].valid and abs(m['intensity'].value - H0) < 0.05, m['intensity']
    assert sum(sw.get('recovered', 0) for sw in res.sweeps) >= 2


def test_non_gaussian_zeroth_order_needs_the_guard_off():
    # beyond the paraxial regime: a distorted 0th order (a Gaussian fits it at overlap ~0.66),
    # much brighter than the first order. (A shape no Gaussian fits at all, e.g. a ring, never
    # trips the guard: the photodiode pick stands and only the camera metrics are invalid.)
    capture = dict(h0=0.8, h1=0.2, distorted_zeroth=True)
    guarded = _metrics(MetricsConfig(fast=True, min_fundamental_overlap=0.85,
                                     recover_first_order_s=S_FIRST), **capture)
    assert not guarded.metrics['intensity'].valid          # the guard rejects the true 0th order
    free = _metrics(MetricsConfig(fast=True, min_fundamental_overlap=None, same_mode_check=False),
                    **capture)
    assert free.metrics['intensity'].valid and abs(free.metrics['intensity'].value - 0.8) < 0.05


def main():
    tests = [(k, v) for k, v in globals().items() if k.startswith('test_') and callable(v)]
    for name, fn in tests:
        fn()
        print(f'ok  {name}')
    print(f'{len(tests)} wrong-mode tests passed')


if __name__ == '__main__':
    main()
