"""Mode peaks in one scan sweep: 0th order, local FSR, first-order family, splitting.

Conventions
- Times are scope seconds. Within a sweep the time-per-FSR is not constant
  (thermal scan), so every peak gets a *local* FSR from its 0th-order
  neighbours, and offsets are expressed as fractions of that local FSR.
- A signed offset `s` is the time offset from the 0th-order peak, divided by
  the local FSR, times the sweep direction (+1 aux rising, -1 falling). It is
  therefore the same number for the same mode family in both scan directions.
  In the 2026-10-12 sessions the strongest side family sits at s ~ +0.30 in all
  four sweeps (up and down), which is the expected flip in time order.
- Heights are in scope volts. A peak whose raw samples reach the saturation
  level is flagged; its height comes from a Lorentzian fit to the unsaturated
  samples only (an extrapolation, so callers treat it as invalid by default).
"""

from dataclasses import dataclass, asdict, field

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

FIT_HALF_WINDOW_S = 0.004       # Lorentzian fit window around a peak (gamma ~1 ms)
FUNDAMENTAL_REL_HEIGHT = 0.5    # 0th-order candidates: at least this x tallest peak
MIN_SEPARATION_FSR = 0.6        # two 0th-order picks are at least this many FSRs apart
SCAN_NONLINEAR_TOL = 0.35       # local FSR off the sweep median by more -> scan_ok False
SIDE_PEAK_MIN_FRAC = 0.04       # side peaks closer than this (in FSR) to 0th are ignored
FAMILY_BIN = 0.025              # histogram bin (FSR fraction) for finding mode families
FAMILY_TOL = 0.05               # a peak belongs to a family within this FSR fraction
SPLIT_IMPROVEMENT = 0.6         # double fit must leave < this x single-fit residual
MIN_PEAK_SNR = 10.0             # weaker 0th-order peaks are not usable


def lorentzian(x, x0, gamma, height, y0):
    return height * gamma ** 2 / ((x - x0) ** 2 + gamma ** 2) + y0


def double_lorentzian(x, x1, g1, h1, x2, g2, h2, y0):
    return lorentzian(x, x1, g1, h1, 0.0) + lorentzian(x, x2, g2, h2, 0.0) + y0


@dataclass
class Peak:
    t: float                    # fitted centre [s]
    height: float               # fitted height above y0 [V]
    gamma: float                # fitted HWHM [s]
    y0: float                   # local baseline [V]
    raw_max: float              # largest raw sample in the window [V]
    saturated: bool = False
    fit_ok: bool = True
    local_fsr: float = float('nan')   # [s], from 0th-order neighbours
    scan_ok: bool = True        # local FSR close to the sweep's median
    snr: float = float('nan')   # height / trace noise sigma

    @property
    def area(self):
        return np.pi * self.height * self.gamma

    def to_dict(self):
        d = asdict(self)
        d['area'] = self.area
        return d


@dataclass
class FirstOrder:
    fundamental_t: float
    s: float                    # signed FSR-fraction offset (see module doc)
    peak: Peak
    split: bool = False
    split_s: float = 0.0        # separation as FSR fraction
    split_linewidths: float = 0.0   # separation / mean FWHM of the two components
    components: list = field(default_factory=list)   # [Peak, Peak] when split

    def to_dict(self):
        return {'fundamental_t': self.fundamental_t, 's': self.s,
                'peak': self.peak.to_dict(), 'split': self.split,
                'split_s': self.split_s, 'split_linewidths': self.split_linewidths,
                'components': [c.to_dict() for c in self.components]}


def noise_sigma(y):
    """Robust noise estimate from sample-to-sample differences."""
    d = np.diff(np.asarray(y, dtype=float))
    return float(1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2))


def estimate_fsr_time(t, y, lag_min=0.12, lag_max=1.5, rel_height=0.25, tol=0.15):
    """Time-per-FSR (seconds) of a sweep, or None.

    Periodicity test on the strong peaks (>= rel_height x the tallest): for
    each candidate lag L (a difference between two strong peaks), every peak
    scores height^2 x (height similarity of its best partner L later, within
    tol x L). Every mode family repeats at the FSR with the same height, so at
    L = FSR nearly all peaks pair up with alike partners; at a family offset
    the partners are a different family (unlike heights) or missing. The shortest lag scoring
    within 85% of the best is taken (multiples of the FSR score as well).

    Comparing each peak with its *next* period keeps this robust to the thermal
    scan's drifting time-per-FSR (up to +-25% within a sweep in the 2026-10-12
    sessions) and to a misaligned cavity whose first order is nearly as tall as
    the 0th order. Falls back to an autocorrelation with fewer than 3 peaks."""
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    idx = candidate_peaks(t, y) if t.size > 10 else np.array([], int)
    if idx.size:
        base = float(np.percentile(y, 5))
        h = y[idx] - base
        keep = h >= rel_height * h.max()
        T, H = t[idx][keep], h[keep]
        if T.size >= 3:
            diffs = (T[None, :] - T[:, None])
            cand = np.unique(np.round(diffs[(diffs >= lag_min) & (diffs <= lag_max)], 4))
            if cand.size:
                # a partner counts by how alike the two heights are (same family
                # repeats with the same height; 0th -> side family does not)
                sim = np.minimum(H[:, None], H[None, :]) / np.maximum(H[:, None], H[None, :])
                w = H ** 2
                scores = np.empty(cand.size)
                for k, L in enumerate(cand):
                    near = np.abs(diffs - L) < tol * L
                    scores[k] = float(np.sum(w * np.max(np.where(near, sim, 0.0), axis=1)))
                good = cand[scores >= 0.85 * scores.max()]
                L = float(good.min())
                for _ in range(3):      # re-centre: the shortest good lag sits low in its cluster
                    matched = diffs[np.abs(diffs - L) < tol * L]
                    L = float(np.median(matched))
                return L
    return _fsr_from_autocorrelation(t, y, lag_min=lag_min)


def _fsr_from_autocorrelation(t, y, lag_min=0.12, lag_max=1.5, smooth_s=0.015):
    """Time-per-FSR (seconds) of a sweep, or None.

    Autocorrelation of a *peak train*: every detected peak becomes a Gaussian
    blob of width `smooth_s` weighted by height^2. The blobs make it robust to
    the thermal scan's drifting time-per-FSR (narrow raw peaks decorrelate
    after a few periods), and height^2 keeps the 0th order dominant over side
    families (a 0.4-height family weighs 0.16). Multiples of the FSR correlate
    almost as well, so the shortest lag nearly as strong as the best is taken."""
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    if t.size < 10 or t[-1] - t[0] < 1.5 * lag_min:
        return None
    idx = candidate_peaks(t, y)
    if idx.size < 3:
        return None
    base = float(np.percentile(y, 5))
    w = np.clip(y[idx] - base, 0, None) ** 2
    dt = smooth_s / 5
    grid = np.arange(t[0], t[-1] + dt, dt)
    train = np.zeros(grid.size)
    np.add.at(train, np.clip(((t[idx] - t[0]) / dt).astype(int), 0, grid.size - 1), w)
    k = np.arange(-15, 16)
    train = np.convolve(train, np.exp(-0.5 * (k / 5.0) ** 2), mode='same')
    n = train.size
    f = np.fft.rfft(train, 2 * n)
    ac = np.fft.irfft(f * np.conj(f))[:n]
    ac /= np.arange(n, 0, -1)
    if ac[0] <= 0:
        return None
    ac /= ac[0]
    i0, i1 = int(lag_min / dt), min(int(lag_max / dt), int(0.75 * n))
    if i1 <= i0 + 2:
        return None
    seg = ac[i0:i1]
    pk, _ = find_peaks(seg)
    if pk.size == 0:
        return None
    strong = pk[seg[pk] >= 0.7 * seg[pk].max()]
    return float((int(strong[0]) + i0) * dt)


def width_estimate(t, y, t_guess, search=0.05):
    """Rough HWHM [s] of the peak at t_guess from its raw half-maximum crossings."""
    i = int(np.argmin(np.abs(t - t_guess)))
    dt = float(np.median(np.diff(t[max(i - 5, 0):i + 6]))) if t.size > 2 else 1e-4
    n = max(int(search / dt), 3)
    lo, hi = max(i - n, 0), min(i + n + 1, t.size)
    seg = y[lo:hi]
    base = float(np.percentile(seg, 5))
    half = base + (y[i] - base) / 2
    j = i - lo
    left = j
    while left > 0 and seg[left] > half:
        left -= 1
    right = j
    while right < seg.size - 1 and seg[right] > half:
        right += 1
    return max((right - left) * dt / 2, dt)


def fit_peak(t, y, t_guess, half_window=None, saturation_v=None):
    """Lorentzian fit around t_guess. Saturated samples are left out of the fit.

    The window defaults to +-6 estimated HWHMs (at least FIT_HALF_WINDOW_S),
    so broad low-finesse peaks are fitted as well as the narrow ones."""
    if half_window is None:
        half_window = max(6 * width_estimate(t, y, t_guess), FIT_HALF_WINDOW_S)
    m = (t >= t_guess - half_window) & (t <= t_guess + half_window)
    tt, yy = t[m], y[m]
    raw_max = float(yy.max()) if yy.size else float('nan')
    saturated = saturation_v is not None and raw_max >= saturation_v
    keep = yy < saturation_v if saturated else np.ones(yy.size, bool)
    y0_guess = float(np.percentile(yy, 5)) if yy.size else 0.0
    peak_i = int(np.argmax(yy)) if yy.size else 0
    h_guess = max(raw_max - y0_guess, 1e-9)
    above = tt[yy - y0_guess > h_guess / 2]
    g_guess = max((above[-1] - above[0]) / 2, 1e-5) if above.size > 1 else 5e-4
    p0 = [float(tt[peak_i]) if yy.size else t_guess, g_guess,
          h_guess * (3 if saturated else 1), y0_guess]
    try:
        if keep.sum() < 6:
            raise RuntimeError('too few samples')
        h_max = 5 * h_guess if saturated else np.inf   # extrapolate at most 5x the clip
        popt, _ = curve_fit(lorentzian, tt[keep], yy[keep],
                            p0=[p0[0], p0[1], min(p0[2], 0.99 * h_max), p0[3]], maxfev=4000,
                            bounds=([tt[0], 1e-6, 0, -np.inf],
                                    [tt[-1], half_window, h_max, np.inf]))
        peak = Peak(float(popt[0]), float(popt[2]), float(popt[1]), float(popt[3]),
                    raw_max, saturated, True)
        # an unsaturated peak cannot be much taller than its own samples: a fit
        # that says so (undersampled / jittered peak) is not trusted
        if not saturated and peak.height > 1.3 * (raw_max - peak.y0) + 3 * noise_sigma(yy):
            peak.fit_ok = False
        return peak
    except (RuntimeError, ValueError):
        return Peak(p0[0], h_guess, g_guess, y0_guess, raw_max, saturated, False)


def candidate_peaks(t, y, min_prominence=None):
    """Indices of all peaks clearly above noise (find_peaks on the raw trace)."""
    sig = noise_sigma(y)
    prom = min_prominence if min_prominence is not None else max(6 * sig, 0.02 * np.ptp(y))
    dt = float(np.median(np.diff(t)))
    idx, _ = find_peaks(y, prominence=prom, distance=max(1, int(0.0015 / dt)))
    return idx


def find_fundamentals(t, y, fsr_time=None, saturation_v=None,
                      rel_height=FUNDAMENTAL_REL_HEIGHT):
    """0th-order peaks in one sweep: the tallest peak of every FSR period.

    Returns (peaks, fsr_time). Each Peak has a local FSR and a scan_ok flag.
    The 0th order is assumed to be the tallest mode of the spectrum; with a
    badly misaligned cavity a higher-order mode can win, which the camera
    check (camera.py) is there to catch."""
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    if fsr_time is None:
        fsr_time = estimate_fsr_time(t, y)
    if fsr_time is None:
        return [], None
    idx = candidate_peaks(t, y)
    if idx.size == 0:
        return [], fsr_time
    heights = y[idx]
    strong = idx[heights >= rel_height * heights.max()]
    picks = []
    for i in strong[np.argsort(-y[strong])]:
        if all(abs(t[i] - t[j]) > MIN_SEPARATION_FSR * fsr_time for j in picks):
            picks.append(i)
    picks.sort()
    peaks = [fit_peak(t, y, t[i], saturation_v=saturation_v) for i in picks]
    sig = max(noise_sigma(y), 1e-12)
    for p in peaks:
        p.snr = p.height / sig
    times = np.array([p.t for p in peaks])
    if len(peaks) >= 2:
        gaps = np.diff(times)
        for k, p in enumerate(peaks):
            nb = [g for g in (gaps[k - 1] if k > 0 else None,
                              gaps[k] if k < len(gaps) else None) if g is not None]
            p.local_fsr = float(np.mean(nb))
        med = float(np.median([p.local_fsr for p in peaks]))
        for p in peaks:
            p.scan_ok = abs(p.local_fsr / med - 1) <= SCAN_NONLINEAR_TOL
    else:
        for p in peaks:
            p.local_fsr = fsr_time
    return peaks, fsr_time


def reassign_by_first_order(t, y, fundamentals, direction, first_order_s, tol=FAMILY_TOL,
                            saturation_v=None, weak_ratio=0.2):
    """Fix 0th-order picks that are really first-order peaks.

    With a badly misaligned cavity the first-order peak can be taller than the
    0th order, so "tallest per FSR" picks it. Knowing the first-order offset s
    (signed, see module doc), and that higher transverse orders are weaker than
    the first order, a pick P of height h is swapped for the peak M at -s when
    - P's own +s neighbour (P's first order if P is a 0th order, the second
      order if P is a first order) is weaker than weak_ratio x h, and
    - M exists and is taller than that +s neighbour.
    A real 0th order keeps its pick: its first order is not that weak, or there
    is no taller peak at -s (in real data a weak second family at -s is common
    and must not trigger a swap).
    Returns (new list, number reassigned); local FSRs are carried over."""
    idx = candidate_peaks(t, y)
    out, swapped = [], 0
    for f in fundamentals:
        F = f.local_fsr
        if not np.isfinite(F):
            out.append(f)
            continue

        def near(t_exp):
            m = idx[np.abs(t[idx] - t_exp) < tol * F]
            return m[np.argmax(y[m])] if m.size else None

        h = f.raw_max - f.y0
        up = near(f.t + first_order_s * direction * F)
        h_up = (y[up] - f.y0) if up is not None else 0.0
        down = near(f.t - first_order_s * direction * F)
        h_down = (y[down] - f.y0) if down is not None else 0.0
        if down is None or h_up >= weak_ratio * h or h_down <= h_up:
            out.append(f)
            continue
        g = fit_peak(t, y, t[down], saturation_v=saturation_v)
        g.local_fsr, g.scan_ok, g.snr = F, f.scan_ok, g.height / max(noise_sigma(y), 1e-12)
        out.append(g)
        swapped += 1
    return out, swapped


def side_offsets(t, y, fundamentals, direction):
    """All side peaks around each 0th-order peak as (fundamental index, s, height ratio, t)."""
    idx = candidate_peaks(t, y)
    out = []
    for k, f in enumerate(fundamentals):
        F = f.local_fsr
        if not np.isfinite(F):
            continue
        sel = idx[(t[idx] > f.t - F / 2) & (t[idx] < f.t + F / 2)]
        for i in sel:
            frac = (t[i] - f.t) / F
            if abs(frac) < SIDE_PEAK_MIN_FRAC:
                continue
            out.append((k, float(frac * direction),
                        float((y[i] - f.y0) / max(f.height, 1e-12)), float(t[i])))
    return out


def family_candidates(offsets, s_range=(-0.5, 0.5), n=3, tol=FAMILY_TOL):
    """The n strongest recurring side families as [(s, weight per 0th-order peak)]."""
    if not offsets:
        return []
    s = np.array([o[1] for o in offsets])
    w = np.array([o[2] for o in offsets])
    n_fund = len({o[0] for o in offsets})
    m = (s > s_range[0]) & (s < s_range[1])
    bins = np.arange(s_range[0], s_range[1] + FAMILY_BIN, FAMILY_BIN)
    hist, edges = np.histogram(s[m], bins=bins, weights=w[m])
    merged = hist + np.r_[hist[1:], 0]
    out = []
    for j in np.argsort(-merged):
        if merged[j] <= 0 or len(out) >= n:
            break
        centre = edges[j + 1]
        if any(abs(centre - c) < 2 * tol for c, _ in out):
            continue
        near = m & (np.abs(s - centre) < tol)
        c = float(np.average(s[near], weights=w[near])) if near.any() else float(centre)
        out.append((c, float(w[near].sum() / max(n_fund, 1))))
    return out


def find_family(offsets, s_range=(-0.5, 0.5), prior_s=None, tol=FAMILY_TOL):
    """Signed offset of the strongest recurring side family, or None.

    Weighted histogram of s (weights = height ratios) over all 0th-order peaks
    of the sweep: a real mode family shows up at the same s every FSR. With
    `prior_s` only the window prior_s +- tol is searched."""
    if not offsets:
        return None
    s = np.array([o[1] for o in offsets])
    w = np.array([o[2] for o in offsets])
    if prior_s is not None:
        m = np.abs(s - prior_s) < tol
        return float(np.average(s[m], weights=w[m])) if m.any() else None
    m = (s > s_range[0]) & (s < s_range[1])
    if not m.any():
        return None
    bins = np.arange(s_range[0], s_range[1] + FAMILY_BIN, FAMILY_BIN)
    hist, edges = np.histogram(s[m], bins=bins, weights=w[m])
    # merge neighbouring bins so a family straddling a bin edge is not split
    merged = hist + np.r_[hist[1:], 0]
    j = int(np.argmax(merged))
    centre = edges[j + 1]
    near = m & (np.abs(s - centre) < tol)
    return float(np.average(s[near], weights=w[near])) if near.any() else float(centre)


def fit_split(t, y, t_guess, gamma_ref, half_window=None):
    """Is the peak near t_guess split? Compare one vs two Lorentzians.

    Returns (single Peak, split bool, [Peak, Peak] or [])."""
    hw = half_window or max(8 * gamma_ref, FIT_HALF_WINDOW_S)
    m = (t >= t_guess - hw) & (t <= t_guess + hw)
    tt, yy = t[m], y[m]
    single = fit_peak(t, y, t_guess, half_window=hw)
    if tt.size < 12 or not single.fit_ok:
        return single, False, []
    res1 = float(np.sum((yy - lorentzian(tt, single.t, single.gamma, single.height, single.y0)) ** 2))
    g = max(single.gamma / 2, 1e-5)
    idx = candidate_peaks(tt, yy)
    if idx.size >= 2:      # two resolved components: start from them
        a, b = sorted(idx[np.argsort(-yy[idx])[:2]])
        p0 = [tt[a], g, yy[a] - single.y0, tt[b], g, yy[b] - single.y0, single.y0]
    else:                  # unresolved: start just either side of the single fit
        p0 = [single.t - single.gamma, g, single.height, single.t + single.gamma, g,
              single.height, single.y0]
    try:
        popt, _ = curve_fit(double_lorentzian, tt, yy, p0=p0, maxfev=8000,
                            bounds=([tt[0], 1e-6, 0, tt[0], 1e-6, 0, -np.inf],
                                    [tt[-1], hw, np.inf, tt[-1], hw, np.inf, np.inf]))
    except (RuntimeError, ValueError):
        return single, False, []
    res2 = float(np.sum((yy - double_lorentzian(tt, *popt)) ** 2))
    x1, g1, h1, x2, g2, h2, y0 = popt
    sep = abs(x2 - x1)
    weak = min(h1, h2) < 0.1 * max(h1, h2)
    is_split = (res2 < SPLIT_IMPROVEMENT * res1 and not weak
                and sep > 0.5 * min(g1, g2))
    if not is_split:
        return single, False, []
    comps = sorted([Peak(float(x1), float(h1), float(g1), float(y0), float('nan')),
                    Peak(float(x2), float(h2), float(g2), float(y0), float('nan'))],
                   key=lambda p: p.t)
    return single, True, comps


def first_orders(t, y, fundamentals, direction, family_s, tol=FAMILY_TOL):
    """The first-order peak next to every 0th-order peak, at signed offset family_s."""
    out = []
    for f in fundamentals:
        F = f.local_fsr
        if not np.isfinite(F) or family_s is None:
            continue
        t_exp = f.t + family_s * direction * F
        idx = candidate_peaks(t, y)
        near = idx[np.abs(t[idx] - t_exp) < tol * F]
        if near.size == 0:
            continue
        i = near[np.argmax(y[near])]
        single, split, comps = fit_split(t, y, t[i], f.gamma)
        fo = FirstOrder(f.t, float((single.t - f.t) / F * direction), single)
        if split:
            sep = comps[1].t - comps[0].t
            fo.split, fo.components = True, comps
            fo.s = float(((comps[0].t + comps[1].t) / 2 - f.t) / F * direction)
            fo.split_s = float(sep / F)
            # in units of the components' own mean FWHM (g1 + g2): independent of
            # the 0th order, whose width is unreliable when it is saturated
            g = comps[0].gamma + comps[1].gamma
            fo.split_linewidths = float(sep / g) if g > 0 else float('nan')
        out.append(fo)
    return out


def _synthetic(direction=1, s_first=0.3, split_s=0.0, fsr=0.45, gamma=1e-3,
               duration=3.0, dt=1e-4, noise=0.003, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(0, duration, dt)
    y = np.full(t.size, 0.01)
    for k in range(int(duration / fsr) + 1):
        t0 = 0.1 + k * fsr
        y += lorentzian(t, t0, gamma, 1.0, 0)
        t1 = t0 + direction * s_first * fsr
        if split_s:
            y += lorentzian(t, t1 - split_s * fsr / 2, gamma, 0.4, 0)
            y += lorentzian(t, t1 + split_s * fsr / 2, gamma, 0.4, 0)
        else:
            y += lorentzian(t, t1, gamma, 0.4, 0)
        y += lorentzian(t, t0 + 0.55 * fsr, gamma, 0.15, 0)   # an unrelated weak family
    return t, y + rng.normal(0, noise, t.size)


def _self_test():
    for direction in (1, -1):
        t, y = _synthetic(direction=direction)
        peaks, fsr = find_fundamentals(t, y)
        assert abs(fsr - 0.45) < 0.01, fsr
        assert len(peaks) >= 6 and all(abs(p.height - 1) < 0.05 for p in peaks), peaks
        fam = find_family(side_offsets(t, y, peaks, direction))
        assert abs(fam - 0.3) < 0.02, fam
        fos = first_orders(t, y, peaks, direction, fam)
        assert fos and not any(f.split for f in fos), fos
    t, y = _synthetic(split_s=0.012)
    peaks, fsr = find_fundamentals(t, y)
    fos = first_orders(t, y, peaks, 1, 0.3)
    assert fos and all(f.split for f in fos), [f.split for f in fos]
    assert all(abs(f.split_s - 0.012) < 0.002 for f in fos), [f.split_s for f in fos]
    # saturation: clip the 0th orders at 0.6 V, fit should recover ~1 V from the wings
    t, y = _synthetic()
    y = np.minimum(y, 0.6)
    peaks, _ = find_fundamentals(t, y, saturation_v=0.6)
    assert all(p.saturated for p in peaks)
    assert all(abs(p.height - 1) < 0.15 for p in peaks), [p.height for p in peaks]
    # misaligned: first order (1.3) taller than the 0th (1.0); the prior fixes the picks
    t = np.arange(0, 3.0, 1e-4)
    y = np.full(t.size, 0.01)
    for k in range(7):
        t0 = 0.1 + k * 0.45
        y += lorentzian(t, t0, 1e-3, 1.0, 0) + lorentzian(t, t0 + 0.3 * 0.45, 1e-3, 1.3, 0)
    y += np.random.default_rng(2).normal(0, 0.003, t.size)
    peaks, _ = find_fundamentals(t, y)
    assert all(abs(p.height - 1.3) < 0.05 for p in peaks)          # fooled without the prior
    fixed, n = reassign_by_first_order(t, y, peaks, 1, 0.3)
    assert n == len(peaks) and all(abs(p.height - 1.0) < 0.05 for p in fixed), [p.height for p in fixed]
    print('spectrum self-test OK')


if __name__ == '__main__':
    _self_test()
