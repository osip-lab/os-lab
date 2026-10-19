"""One capture -> alignment metrics, each with a validity flag.

    evaluate_capture(t, signal, aux, frames=None, windows=None, mask=None, cfg=MetricsConfig())
    evaluate_session(path, cfg=MetricsConfig())      # a saved synced session folder

Metrics (see kalishlot/MECHANICAL_CONTROL.md, section 2):
- intensity          median fitted 0th-order peak height [V]          (higher = better)
- gaussian_overlap   median overlap of 0th-order frames with a Gaussian (1 = perfect)
- ellipticity        median sigma_min / sigma_max of those frames       (1 = round)
- centroid_x/_y      median 0th-order beam centre on the camera [px]   (guard metric)
- sensor_x/_y        the same in uncropped, unbinned sensor pixels - the units of
                     kalishlot's camera markers (needs cfg.sensor_geometry)
- first_order_s      signed offset of the first-order family [FSR]     (diagnostic)
- split_linewidths   median first-order splitting / component FWHM     (lower = better)
- split_fsr          median first-order splitting as an FSR fraction   (lower = better)

Rules that make a metric invalid rather than wrong: saturated peaks (unless
cfg.allow_saturated), peaks in the nonlinear scan region, too few samples,
frames that mostly saw other modes (low exposure purity), dim or failed fits.
The optimizer must treat `valid=False` as "no information", never as zero.

Up- and down-sweeps are never pooled: every metric is computed per direction,
and the headline value uses cfg.direction (+1, -1, or None = the direction with
more valid samples).
"""

import json
import sys
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np

from alignment import camera, spectrum, sweeps

_REPO = Path(__file__).resolve().parents[2]
FAMILY_CAMERA_MARGIN = 0.08     # camera overrides the strongest family only by this much
SAME_MODE_MIN_RATIO = 0.8       # a 0th-order pick whose frame overlaps the best pick's Gaussian
                                # less than this x that pick's own overlap is another mode
                                # (2026-10-19 live: impostors 0.48, true 0th orders 0.96)


@dataclass
class MetricsConfig:
    guard_s: float = sweeps.DEFAULT_GUARD_S
    direction: int | None = None          # headline direction; None = best populated
    fsr_time_s: float | None = None       # None = estimate from each sweep
    saturation_v: float | None = None     # None = from the session's scope range
    allow_saturated: bool = False         # use fitted heights of clipped peaks
    first_order_s: float | None = None    # known first-order offset (signed FSR fraction). Picks the
                                          # first-order family AND corrects 0th-order picks that are
                                          # really first orders (misaligned cavity); None = strongest family
    first_order_sign: int | None = sweeps.FIRST_ORDER_AFTER_ON_RISING
                                          # sign of s the first-order family must have when first_order_s
                                          # is None. Lab rule 2026-10-19 (aux rising -> first order after
                                          # the 0th, falling -> before) = +1 at any cavity length. None =
                                          # either sign (the strongest family, camera tie-break)
    min_samples: int = 2                  # fewer valid samples -> metric invalid
    min_purity: float = 0.15              # frame must get >= this share of its light from the target peak
    use_mask: bool = True                 # replace pixels outside the session mask by background
    min_fundamental_overlap: float | None = None  # camera guard: a 0th-order pick whose frame overlaps a
                                          # Gaussian less is not used (it may be a higher order). Off by
                                          # default: low-finesse sessions (2026-10-11) show real 0th orders
                                          # at 0.72-0.85, high-finesse ones (2026-10-12) at ~0.97.
    sensor_geometry: dict | None = None   # where the frames sit on the sensor, as a session's 'camera'
                                          # block records it: offset_x/offset_y (ROI origin, in BINNED px)
                                          # and binning_x/binning_y. Enables sensor_x/_y. evaluate_session
                                          # fills it from the session; None = no sensor coordinates.


@dataclass
class Metric:
    name: str
    value: float = float('nan')
    spread: float = float('nan')          # robust std of the samples
    n: int = 0
    valid: bool = False
    reason: str = ''

    def to_dict(self):
        return asdict(self)


@dataclass
class CaptureMetrics:
    metrics: dict = field(default_factory=dict)          # name -> Metric (headline direction)
    by_direction: dict = field(default_factory=dict)     # +1/-1 -> {name -> Metric}
    direction: int | None = None
    sweeps: list = field(default_factory=list)           # per-sweep details (dicts)

    def value(self, name):
        m = self.metrics.get(name)
        return m.value if m is not None and m.valid else None

    def to_dict(self):
        return {'direction': self.direction,
                'metrics': {k: m.to_dict() for k, m in self.metrics.items()},
                'by_direction': {str(d): {k: m.to_dict() for k, m in ms.items()}
                                 for d, ms in self.by_direction.items()},
                'sweeps': self.sweeps}


def frame_to_sensor(x, y, geometry):
    """Frame pixel (x = column, y = row) -> uncropped, unbinned sensor pixel, the
    units of kalishlot's markers. A binned pixel covers b sensor pixels; its
    centre is at b * (offset + index) + (b - 1) / 2."""
    bx = int(geometry.get('binning_x', 1) or 1)
    by = int(geometry.get('binning_y', 1) or 1)
    return (bx * (geometry.get('offset_x', 0) + x) + (bx - 1) / 2,
            by * (geometry.get('offset_y', 0) + y) + (by - 1) / 2)


def _robust(values):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float('nan'), float('nan'), 0
    med = float(np.median(v))
    spread = float(1.4826 * np.median(np.abs(v - med))) if v.size > 1 else float('nan')
    return med, spread, int(v.size)


def _metric(name, values, min_n, reason_if_empty):
    med, spread, n = _robust(values)
    if n == 0:
        return Metric(name, reason=reason_if_empty)
    if n < min_n:
        return Metric(name, med, spread, n, False, f'only {n} sample(s), need {min_n}')
    return Metric(name, med, spread, n, True)


def _same_mode_impostors(rows, frames, mask, cfg):
    """0th-order picks whose camera frame is not the same spatial mode as the
    most Gaussian pick of the sweep: {t: likeness}.

    Every 0th order of one capture is the same mode, so a pick that looks
    different is a higher order taken for a 0th order. Seen live 2026-10-19:
    a strong first order (0.85x the 0th's height) made the FSR estimate come
    out at half its value, and "tallest per FSR" then alternated between the
    two modes (likeness 0.48 vs 0.96 for true 0th orders)."""
    scored = [r for r in rows if r.get('frame_usable') and r['gauss']['valid']]
    if len(scored) < 2:
        return {}
    ref_row = max(scored, key=lambda r: r['gauss']['overlap'])
    ref = camera.GaussianScore(**ref_row['gauss'])
    out = {}
    for r in scored:
        if r is ref_row:
            continue
        like = camera.fundamental_likeness(_masked(frames[r['frame']], mask, cfg), ref)
        r['same_mode_likeness'] = like
        if np.isfinite(like) and like < SAME_MODE_MIN_RATIO * ref.overlap:
            out[r['t']] = like
    return out


def _analyze_sweep(t, y, sw, frames, windows, mask, cfg):
    m = sw.mask(t)
    tt, yy = t[m], y[m]
    have_camera = frames is not None and windows is not None
    fundamentals, fsr = spectrum.find_fundamentals(tt, yy, cfg.fsr_time_s, cfg.saturation_v)
    rec = {'sweep': sw.to_dict(), 'fsr_time_s': fsr, 'fundamentals': [], 'first_orders': []}
    if have_camera and fundamentals:
        # Camera check that the picks are one mode; if some are not, pick again
        # with the FSR implied by the true ones (impostors halve the estimate).
        rows = [{'t': f.t, **_frame_score(t, y, f.t, f.gamma, frames, windows, mask, cfg)}
                for f in fundamentals]
        impostors = _same_mode_impostors(rows, frames, mask, cfg)
        if impostors:
            rec['impostors'] = {f'{k:.4f}': v for k, v in impostors.items()}
            kept = sorted(f.t for f in fundamentals if f.t not in impostors)
            if len(kept) >= 2:
                fsr = float(np.median(np.diff(kept)))
                fundamentals, _ = spectrum.find_fundamentals(tt, yy, fsr, cfg.saturation_v)
                rec['fsr_time_s'], rec['fsr_from'] = fsr, 'camera-checked 0th orders'
    if cfg.first_order_s is not None and fundamentals:
        fundamentals, rec['reassigned'] = spectrum.reassign_by_first_order(
            tt, yy, fundamentals, sw.direction, cfg.first_order_s, saturation_v=cfg.saturation_v)
    by_t = {}
    for f in fundamentals:
        strong = f.snr >= spectrum.MIN_PEAK_SNR
        usable = (f.fit_ok and f.scan_ok and strong
                  and (cfg.allow_saturated or not f.saturated))
        why = ('' if usable else 'fit failed' if not f.fit_ok else
               'weak' if not strong else
               'nonlinear scan' if not f.scan_ok else 'saturated')
        row = {**f.to_dict(), 'usable': usable, 'why': why}
        if have_camera:
            row.update(_frame_score(t, y, f.t, f.gamma, frames, windows, mask, cfg))
            g = row.get('gauss')
            if (cfg.min_fundamental_overlap is not None and usable and row.get('frame_usable')
                    and g and g['overlap'] < cfg.min_fundamental_overlap):
                row['usable'], row['why'] = False, 'frame not Gaussian (higher order?)'
        rec['fundamentals'].append(row)
        by_t[f.t] = row
    if have_camera:
        # whatever the re-pick could not fix (too few true 0th orders) is
        # at least kept out of the metrics
        for t_imp in _same_mode_impostors(rec['fundamentals'], frames, mask, cfg):
            by_t[t_imp]['usable'], by_t[t_imp]['why'] = False, 'not the same mode as the 0th order'

    offsets = spectrum.side_offsets(tt, yy, fundamentals, sw.direction)
    if cfg.first_order_s is None and cfg.first_order_sign:
        offsets = [o for o in offsets if o[1] * cfg.first_order_sign > 0]
    candidates = spectrum.family_candidates(offsets)
    rec['family_candidates'] = candidates
    if cfg.first_order_s is not None:
        family = spectrum.find_family(offsets, prior_s=cfg.first_order_s)
        rec['family_choice'] = 'prior'
    else:
        family = spectrum.find_family(offsets)
        rec['family_choice'] = 'strongest'
        if have_camera and len(candidates) >= 2:
            # the two strongest families are often +-s of comparable weight;
            # the camera decides which one looks like a first-order mode
            scores = [_family_first_order_score(t, fundamentals, by_t, sw.direction, c,
                                                frames, windows, mask, cfg)
                      for c, _ in candidates[:2]]
            rec['family_camera_scores'] = scores
            if all(np.isfinite(scores)) and abs(scores[0] - scores[1]) >= FAMILY_CAMERA_MARGIN:
                family = spectrum.find_family(offsets, prior_s=candidates[int(np.argmax(scores))][0])
                rec['family_choice'] = 'camera'
    rec['family_s'] = family

    for fo in spectrum.first_orders(tt, yy, fundamentals, sw.direction, family):
        row = fo.to_dict()
        ref_row = by_t.get(fo.fundamental_t, {})
        if have_camera and ref_row.get('gauss'):
            i = _frame_index(windows, fo.peak.t)
            ref = camera.GaussianScore(**ref_row['gauss'])
            if i is not None and ref.valid:
                frame = _masked(frames[i], mask, cfg)
                row['frame'] = i
                row['fundamental_likeness'] = camera.fundamental_likeness(frame, ref)
                row['first_order_likeness'], row['template'] = camera.first_order_likeness(frame, ref)
                row['purity'] = camera.exposure_purity(t, y, windows[i], fo.peak.t,
                                                       max(5 * fo.peak.gamma, 0.002))
                row['camera_confirmed'] = bool(
                    row['fundamental_likeness'] < camera.FIRST_ORDER_MAX_OVERLAP
                    and row['first_order_likeness'] >= camera.FIRST_ORDER_MIN_LIKENESS)
        rec['first_orders'].append(row)
    return rec


def _family_first_order_score(t, fundamentals, by_t, direction, s, frames, windows, mask, cfg):
    """Median first-order template overlap of the frames at offset s from each 0th order."""
    vals = []
    for f in fundamentals:
        g = by_t[f.t].get('gauss')
        if not g or not g['valid'] or not np.isfinite(f.local_fsr):
            continue
        i = _frame_index(windows, f.t + s * direction * f.local_fsr)
        if i is not None:
            vals.append(camera.first_order_likeness(_masked(frames[i], mask, cfg),
                                                    camera.GaussianScore(**g))[0])
    return float(np.median(vals)) if vals else float('nan')


def _frame_index(windows, t_peak):
    w = np.asarray(windows)
    inside = np.nonzero((w[:, 0] <= t_peak) & (t_peak < w[:, 1]))[0]
    return int(inside[0]) if inside.size else None


def _masked(frame, mask, cfg):
    return frame if (mask is None or not cfg.use_mask) else np.where(
        mask, frame, np.median(frame[~mask]) if (~mask).any() else 0)


def _frame_score(t, y, t_peak, gamma, frames, windows, mask, cfg):
    i = _frame_index(windows, t_peak)
    if i is None:
        return {'frame': None, 'frame_why': 'peak fell between exposures'}
    purity = camera.exposure_purity(t, y, windows[i], t_peak, max(5 * gamma, 0.002))
    score = camera.gaussian_score(frames[i], mask if cfg.use_mask else None)
    return {'frame': i, 'purity': purity, 'gauss': score.to_dict(),
            'frame_usable': bool(score.valid and purity >= cfg.min_purity),
            'frame_why': score.reason if not score.valid else
            (f'purity {purity:.2f} < {cfg.min_purity}' if purity < cfg.min_purity else '')}


def _direction_metrics(recs, cfg):
    heights, overlaps, ells, cx, cy, fam, splits, splits_fsr = [], [], [], [], [], [], [], []
    confirmed_any = False
    for r in recs:
        if r['family_s'] is not None:
            fam.append(r['family_s'])
        for f in r['fundamentals']:
            if f['usable']:
                heights.append(f['height'])
                if f.get('frame_usable'):
                    g = f['gauss']
                    overlaps.append(g['overlap'])
                    ells.append(g['ellipticity'])
                    cx.append(g['x0'])
                    cy.append(g['y0'])
        for fo in r['first_orders']:
            if fo.get('camera_confirmed') is False:
                continue
            confirmed_any |= bool(fo.get('camera_confirmed'))
            splits.append(fo['split_linewidths'] if fo['split'] else 0.0)
            splits_fsr.append(fo['split_s'] if fo['split'] else 0.0)
    n = cfg.min_samples
    out = {
        'intensity': _metric('intensity', heights, n, 'no usable 0th-order peak'),
        'gaussian_overlap': _metric('gaussian_overlap', overlaps, n, 'no usable 0th-order frame'),
        'ellipticity': _metric('ellipticity', ells, n, 'no usable 0th-order frame'),
        'centroid_x': _metric('centroid_x', cx, n, 'no usable 0th-order frame'),
        'centroid_y': _metric('centroid_y', cy, n, 'no usable 0th-order frame'),
        'first_order_s': _metric('first_order_s', fam, 1, 'no side family found'),
        'split_linewidths': _metric('split_linewidths', splits, n, 'no first-order peak'),
        'split_fsr': _metric('split_fsr', splits_fsr, n, 'no first-order peak'),
    }
    if cfg.sensor_geometry is not None:
        sx, sy = frame_to_sensor(np.asarray(cx, dtype=float), np.asarray(cy, dtype=float),
                                 cfg.sensor_geometry)
        out['sensor_x'] = _metric('sensor_x', sx, n, 'no usable 0th-order frame')
        out['sensor_y'] = _metric('sensor_y', sy, n, 'no usable 0th-order frame')
    else:
        for k in ('sensor_x', 'sensor_y'):
            out[k] = Metric(k, reason='sensor geometry unknown (MetricsConfig.sensor_geometry)')
    for k in ('split_linewidths', 'split_fsr'):
        if out[k].valid and not confirmed_any:
            out[k].reason = 'first order not confirmed on the camera'
    return out


def evaluate_capture(t, signal, aux, frames=None, windows=None, mask=None, cfg=None):
    cfg = cfg or MetricsConfig()
    t = np.asarray(t, dtype=float)
    y = np.asarray(signal, dtype=float)
    sws = sweeps.split_sweeps(t, aux, guard_s=cfg.guard_s)
    recs = [_analyze_sweep(t, y, sw, frames, windows, mask, cfg) for sw in sws]
    result = CaptureMetrics(sweeps=recs)
    for d in (1, -1):
        rd = [r for r in recs if r['sweep']['direction'] == d]
        if rd:
            result.by_direction[d] = _direction_metrics(rd, cfg)
    if not result.by_direction:
        result.metrics = _direction_metrics([], cfg)
        return result
    if cfg.direction in result.by_direction:
        d = cfg.direction
    else:
        d = max(result.by_direction, key=lambda k: result.by_direction[k]['intensity'].n)
    result.direction = d
    result.metrics = result.by_direction[d]
    return result


# ------------------------------------------------------------ saved sessions

def load_session_arrays(path, with_frames=True):
    """(session, t, signal, aux, frames, windows, mask) from a synced session folder."""
    sys.path.insert(0, str(_REPO / 'pico_scope'))
    from mode_video_sync import load_session, session_windows  # noqa: E402
    path = Path(path)
    sess_file = path if path.suffix == '.json' else sorted(path.glob('*_session.json'))[0]
    folder = sess_file.parent
    session = json.loads(sess_file.read_text(encoding='utf-8'))
    z = np.load(folder / session['scope']['file'])
    aux = z['aux'] if 'aux' in z.files else None
    frames = windows = mask = None
    if with_frames:
        _, frames = load_session(sess_file)
        windows, _ = session_windows(sess_file)
        if session.get('mask_file') and (folder / session['mask_file']).exists():
            mask = np.load(folder / session['mask_file']).astype(bool)
    return session, z['t'], z['signal'], aux, frames, windows, mask


def evaluate_session(path, cfg=None, with_frames=True):
    cfg = cfg or MetricsConfig()
    session, t, y, aux, frames, windows, mask = load_session_arrays(path, with_frames)
    if aux is None:
        raise ValueError(f'{path}: no aux (scan ramp) channel; sweeps cannot be split')
    if cfg.saturation_v is None and session.get('scope', {}).get('range_v'):
        cfg = MetricsConfig(**{**asdict(cfg), 'saturation_v': 0.995 * float(session['scope']['range_v'])})
    cam = session.get('camera') or {}
    if cfg.sensor_geometry is None and 'offset_x' in cam:
        cfg = MetricsConfig(**{**asdict(cfg), 'sensor_geometry': {
            k: cam.get(k, d) for k, d in (('offset_x', 0), ('offset_y', 0),
                                          ('binning_x', 1), ('binning_y', 1))}})
    return evaluate_capture(t, y, aux, frames, windows, mask, cfg)


def _self_test():
    # synthetic triangle-scanned trace: one turnaround, peaks every FSR
    t = np.arange(0, 6, 1e-4)
    aux = np.where(t < 2.0, -t, -4.0 + t)             # min at t=2 -> down then up
    y = np.full(t.size, 0.01)
    for k in range(14):
        t0 = 0.1 + k * 0.45
        y += spectrum.lorentzian(t, t0, 1e-3, 1.0, 0)
        y += spectrum.lorentzian(t, t0 + (0.3 * 0.45 if t0 > 2 else -0.3 * 0.45), 1e-3, 0.4, 0)
    y += np.random.default_rng(0).normal(0, 0.003, t.size)
    res = evaluate_capture(t, y, aux)
    assert set(res.by_direction) == {1, -1}, res.by_direction.keys()
    for d, ms in res.by_direction.items():
        assert ms['intensity'].valid and abs(ms['intensity'].value - 1) < 0.05, (d, ms['intensity'])
        assert abs(ms['first_order_s'].value - 0.3) < 0.02, (d, ms['first_order_s'])
    assert not res.metrics['gaussian_overlap'].valid      # no camera given
    print('metrics self-test OK')


if __name__ == '__main__':
    _self_test()
