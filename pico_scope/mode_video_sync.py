"""Align a mode video with the cavity spectrum that was recorded alongside it.

No hardware, no GUI, no plotting - this is the piece that earns a self-test:

    python pico_scope/mode_video_sync.py --self-test

## What is being aligned, and why it can be

The camera and the Channel D photodiode watch the same cavity transmission, so
the total brightness of frame k is the scope trace boxcar-integrated over that
frame's exposure window. Nothing else is shared: PicoScope 7's record is started
by a human pressing a button, so the host does not know when it began, and there
is no common clock to appeal to - coarse or fine.

That leaves exactly one unknown. The camera's chunk timestamps give the frame
times *relative to each other* to sub-microsecond precision, and they make a
dropped frame visible on their own, so the only thing missing is where frame 0
sits in the scope's timebase. `fit_time_offset` finds that one scalar by sliding
the frame grid along the trace until the predicted brightness sequence matches
the measured one. See pico_scope/SYNCHRONIZED_VIDEO_SPECTRUM.md for the design
and for the measured precision (0.4-2.4% brightness noise on this setup, giving
an offset good to well under a tenth of a frame).

The gain and the dark level of the camera are unknown and irrelevant, so they
are fitted away analytically at every candidate offset: only the *shape* of the
brightness sequence carries timing.

## Two gotchas, documented rather than "fixed"

**Where t = 0 sits does not matter.** The fitted offset is expressed in the
scope CSV's own `Time` column, and every window this module returns is in that
same coordinate, so whatever origin PicoScope chose cancels out.

**Which unit that column is in does matter**, and it varies between exports -
the same lab has `.psdata` files whose `Time` reads in seconds and others in
milliseconds. The repo's older loaders discard the units row
(`pd.read_csv(..., skiprows=[1, 2])`); `load_scope_csv` here reads it and
converts to seconds and volts, which is why it does not reuse them.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

TIME_COLUMN = 'Time'
SIGNAL_COLUMN = 'Channel D'      # cavity transmission, as in mode_map_2d.py

# The units row of a PicoScope export, e.g. "(s),(V),(mV)".
TIME_UNITS = {'s': 1.0, 'ms': 1e-3, 'us': 1e-6, 'µs': 1e-6, 'ns': 1e-9}
VOLT_UNITS = {'v': 1.0, 'mv': 1e-3, 'uv': 1e-6, 'µv': 1e-6}


class ScopeTrace:
    """A scope channel in seconds and volts, whatever the CSV said."""

    def __init__(self, t, signal, time_unit, signal_unit, path=None):
        self.t = t
        self.signal = signal
        self.time_unit = time_unit
        self.signal_unit = signal_unit
        self.path = path

    @property
    def duration(self):
        return float(self.t[-1] - self.t[0])

    def __repr__(self):
        return (f'ScopeTrace({self.t.size} samples, '
                f'{self.t[0]:.6g}..{self.t[-1]:.6g} s, '
                f'from ({self.time_unit})/({self.signal_unit}))')


class OffsetFit:
    """Result of fit_time_offset.

    `t0` is when frame 0's exposure began, in the scope's time coordinate.

    Two separate questions have to be asked about it, and conflating them is
    misleading - which the first real capture demonstrated:

    `depth` = median residual / best residual. **Did the fit find the sweep at
    all?** A featureless burst - beam blocked, or the burst missing the sweep -
    gives a depth near 1.

    Do not read depth as accuracy. It is capped by how well the camera can
    track the photodiode at all, and on this setup that ceiling is low: the
    photodiode integrates the whole transmitted beam while the camera weights
    it spatially, so the counts-per-volt differ from one transverse mode to the
    next and no single gain fits them all. Measured over 14 captures, depth
    ranged from 1.2 to 16 while the fitted offset stayed the same to within a
    few microseconds - constraining the search to a 50 ms window changed the
    answer in only 2 of them, and those two were the ones that failed outright.
    A depth of 1.5 with a consistent offset is a good fit of a mismatched
    model, not a bad fit.

    `margin` = best rival residual / best residual, where the rival must lie
    more than a few frame periods away. **Is that offset unique?** A cavity
    sweep repeats every free spectral range, so a long scope record contains
    many positions that fit almost as well and the margin collapses towards 1 -
    even when the alignment is perfectly correct. A short record has few such
    aliases and a high margin.

    So `locked and not unique` is the normal outcome for a long record: the
    alignment within the sweep is right, but which repetition it was is
    undetermined. For identifying a transverse mode that is usually harmless,
    since equivalent positions in the sweep carry equivalent mode content.
    """

    def __init__(self, t0, residual, margin, gain, offset, grid, residuals,
                 depth=None):
        self.t0 = t0
        self.residual = residual
        self.margin = margin
        self.depth = depth
        self.gain = gain
        self.offset = offset
        self.grid = grid
        self.residuals = residuals

    @property
    def locked(self):
        """The frame grid found the sweep's structure.

        The threshold is deliberately low. Depth is limited by model mismatch
        rather than by whether the offset is right (see the class docstring),
        so a stricter gate rejects fits that are demonstrably correct.
        """
        return self.depth is not None and self.depth > 1.5

    @property
    def unique(self):
        """No rival offset a free spectral range away fits nearly as well."""
        return self.margin > 1.5

    @property
    def trustworthy(self):
        """Locked onto the sweep and unambiguous about where."""
        return self.locked and self.unique

    def __repr__(self):
        return (f'OffsetFit(t0={self.t0:.6f} s, depth={self.depth:.1f}x, '
                f'margin={self.margin:.1f}x, gain={self.gain:.4g})')


# ------------------------------------------------------------------ loading
def _unit_scale(token, table, what):
    """Turn a units-row token like '(mV)' into a multiplier."""
    cleaned = token.strip().strip('()').strip()
    try:
        return table[cleaned if cleaned in table else cleaned.lower()]
    except KeyError:
        raise ValueError(f'unrecognised {what} unit {token!r}; known: '
                         f'{sorted(table)}')


def load_scope_csv(path, time_column=TIME_COLUMN, signal_column=SIGNAL_COLUMN):
    """Read a PicoScope CSV export into seconds and volts.

    Row 1 is the header, row 2 the units and row 3 blank. Unlike the older
    loaders in this repo, the units row is read rather than skipped - exports
    from the same instrument disagree about whether `Time` is seconds or
    milliseconds, and a silent factor of 1000 would put every frame in the
    wrong place.
    """
    path = Path(path)
    header = pd.read_csv(path, nrows=1)
    units = {column: str(value) for column, value in header.iloc[0].items()}
    frame = pd.read_csv(path, skiprows=[1, 2])
    missing = [c for c in (time_column, signal_column) if c not in frame.columns]
    if missing:
        raise KeyError(f'{path.name} has no column(s) {missing}; it has '
                       f'{list(frame.columns)}')
    frame = frame.loc[:, [time_column, signal_column]].dropna()
    time_scale = _unit_scale(units[time_column], TIME_UNITS, 'time')
    signal_scale = _unit_scale(units[signal_column], VOLT_UNITS, 'voltage')
    return ScopeTrace(frame[time_column].to_numpy(float) * time_scale,
                      frame[signal_column].to_numpy(float) * signal_scale,
                      units[time_column], units[signal_column], path)


def load_session(path, mmap=True):
    """Read a capture written by pico_scope/mode_video_capture.py.

    `path` may be the session JSON or the folder holding it. Returns
    (session_dict, frames); the frame stack is memory-mapped by default, since
    a capture is tens of megabytes and most callers only touch a few frames.

    On Windows a memory-mapped file stays locked until the array is released,
    so the session folder cannot be deleted or overwritten while it is open.
    Pass `mmap=False` to read the frames into memory instead, or call
    `release_frames()` when done.
    """
    path = Path(path)
    if path.is_dir():
        candidates = sorted(path.glob('*_session.json'))
        if len(candidates) != 1:
            raise FileNotFoundError(
                f'expected exactly one *_session.json in {path}, '
                f'found {len(candidates)}')
        path = candidates[0]
    session = json.loads(path.read_text(encoding='utf-8'))
    frames_path = path.parent / session['frames_file']
    frames = np.load(frames_path, mmap_mode='r' if mmap else None)
    return session, frames


def release_frames(frames):
    """Close a memory-mapped frame stack so its file can be deleted.

    A no-op for an ordinary array. Only Windows really needs this, and only
    when the session folder is about to be removed or rewritten.
    """
    handle = getattr(frames, '_mmap', None)
    if handle is not None:
        handle.close()


# --------------------------------------------------------------- brightness
def varying_pixel_mask(frames, threshold=0.15):
    """Pixels whose value changes during the burst - i.e. where the mode is.

    The mode covers about 1% of the frame, so an unmasked mean spends the rest
    of its pixels accumulating noise; masking is worth 5-7x on this setup.

    It is not free: the photodiode sums light the mask excludes, so a mode that
    strays outside it makes the two sequences disagree. Capture both series and
    let the fit's margin decide - see SYNCHRONIZED_VIDEO_SPECTRUM.md.
    """
    frames = np.asarray(frames)
    span = frames.max(axis=0).astype(np.float32) - frames.min(axis=0).astype(np.float32)
    if span.max() <= 0:
        return np.ones(frames.shape[1:], dtype=bool)
    return span > threshold * span.max()


def frame_brightness(frames, mask=None):
    """Mean pixel value per frame, over `mask` if one is given."""
    frames = np.asarray(frames)
    flat = frames.reshape(len(frames), -1).astype(np.float32)
    if mask is None:
        return flat.mean(axis=1)
    flat_mask = np.asarray(mask).ravel()
    if flat_mask.shape[0] != flat.shape[1]:
        raise ValueError(f'mask has {flat_mask.shape[0]} pixels but frames '
                         f'have {flat.shape[1]}')
    if not flat_mask.any():
        raise ValueError('mask selects no pixels')
    return flat[:, flat_mask].mean(axis=1)


# ------------------------------------------------------------- frame timing
def frame_start_times(meta):
    """Exposure start of every frame, in seconds relative to frame 0.

    Taken from the camera's own chunk timestamps, so a dropped frame leaves a
    real gap here instead of silently shifting everything after it - which is
    what lets the fit stay correct without the frames being evenly spaced.
    """
    stamps = np.array([row['camera_timestamp_ns'] for row in meta], dtype=float)
    if stamps.size == 0:
        raise ValueError('no frames in meta')
    return (stamps - stamps[0]) / 1e9


def dropped_frames(meta):
    """Gaps in the BlockID sequence: which frames the camera never delivered."""
    ids = np.array([row['block_id'] for row in meta], dtype=np.int64)
    gaps = np.diff(ids)
    return [{'after_index': int(i), 'after_block_id': int(ids[i]),
             'n_missing': int(gap - 1)}
            for i, gap in enumerate(gaps) if gap != 1]


def frame_windows(frame_starts, exposure_s, t0=0.0):
    """(n, 2) array of [exposure_start, exposure_end] in the scope's timebase."""
    starts = np.asarray(frame_starts, dtype=float) + t0
    return np.column_stack([starts, starts + exposure_s])


def frame_at_time(windows, t):
    """Index of the frame whose exposure window contains `t`, or None.

    None means `t` fell in dead time between exposures, or outside the burst.
    Use nearest_frame when a viewer needs something to show regardless.
    """
    windows = np.asarray(windows)
    inside = np.nonzero((windows[:, 0] <= t) & (t < windows[:, 1]))[0]
    return int(inside[0]) if inside.size else None


def nearest_frame(windows, t):
    """Index of the frame whose window centre is closest to `t`."""
    windows = np.asarray(windows)
    centres = windows.mean(axis=1)
    return int(np.argmin(np.abs(centres - t)))


# ---------------------------------------------------------------- the fit
def _cumulative(t, signal):
    """Trapezoidal running integral of `signal`, so non-uniform dt is fine."""
    increments = 0.5 * (signal[1:] + signal[:-1]) * np.diff(t)
    return np.concatenate([[0.0], np.cumsum(increments)])


def predicted_brightness(t0_grid, frame_starts, exposure_s, t, cumulative):
    """Boxcar-integrated trace for each candidate offset: (n_grid, n_frames)."""
    t0_grid = np.atleast_1d(np.asarray(t0_grid, dtype=float))
    starts = t0_grid[:, None] + np.asarray(frame_starts, dtype=float)[None, :]
    lower = np.interp(starts, t, cumulative)
    upper = np.interp(starts + exposure_s, t, cumulative)
    return (upper - lower) / exposure_s


def _residuals_after_linear_fit(model, observed):
    """Mean squared residual once gain and dark level are fitted out.

    Camera counts per volt and the dark level are both unknown, and neither
    carries timing information, so they are marginalised analytically at every
    candidate offset rather than searched over.
    """
    centred_model = model - model.mean(axis=1, keepdims=True)
    centred_obs = observed - observed.mean()
    numerator = centred_model @ centred_obs
    denominator = np.einsum('ij,ij->i', centred_model, centred_model)
    gain = np.divide(numerator, denominator,
                     out=np.zeros_like(numerator),
                     where=denominator > 0)
    residual = centred_obs[None, :] - gain[:, None] * centred_model
    return (residual ** 2).mean(axis=1), gain


def fit_time_offset(frame_starts, exposure_s, brightness, trace,
                    search=None, coarse_step=None, refine_steps=200,
                    alias_guard=None):
    """Find when frame 0's exposure began, in the scope's time coordinate.

    Slides the frame grid along the trace, and at each position compares the
    boxcar-integrated trace with the measured brightness after fitting out gain
    and dark level. Two passes: a coarse grid over the whole plausible range,
    then a fine one around the winner.

    `search` is (low, high) for t0 and defaults to every position that keeps the
    whole burst inside the record. `alias_guard` is how far from the best offset
    a rival minimum must be to count towards `margin`; it defaults to three
    frame periods.

    Returns an OffsetFit. Check `.margin` before believing `.t0` - a margin near
    1 means the fit could not tell the true offset from an alias, which happens
    when the brightness sequence is featureless (no resonances in the burst) or
    when the camera saturated.
    """
    frame_starts = np.asarray(frame_starts, dtype=float)
    brightness = np.asarray(brightness, dtype=float)
    if frame_starts.size != brightness.size:
        raise ValueError(f'{frame_starts.size} frame times but '
                         f'{brightness.size} brightness values')
    if frame_starts.size < 3:
        raise ValueError('need at least 3 frames to fit an offset')

    t, signal = trace.t, trace.signal
    cumulative = _cumulative(t, signal)
    burst = float(frame_starts[-1] + exposure_s)
    if search is None:
        search = (float(t[0]), float(t[-1] - burst))
    low, high = search
    if high <= low:
        raise ValueError(
            f'the burst is {burst:.3f} s but the record is only '
            f'{trace.duration:.3f} s - they cannot be made to overlap. Record '
            f'fewer frames, or a longer scope trace.')

    period = float(np.median(np.diff(frame_starts))) if frame_starts.size > 1 \
        else exposure_s
    if coarse_step is None:
        coarse_step = exposure_s / 50.0
    if alias_guard is None:
        alias_guard = 3 * period

    # A long scope record makes this grid large - 20 s at 0.2 ms is 10^5
    # candidates - and the model matrix is (candidates x frames), so it is
    # evaluated in chunks rather than all at once.
    grid = np.arange(low, high, coarse_step)
    residuals = np.empty(grid.size)
    chunk = max(1, int(4e6 // max(frame_starts.size, 1)))
    for start in range(0, grid.size, chunk):
        piece = grid[start:start + chunk]
        model = predicted_brightness(piece, frame_starts, exposure_s, t,
                                     cumulative)
        residuals[start:start + chunk] = _residuals_after_linear_fit(
            model, brightness)[0]
    best = int(np.argmin(residuals))

    fine = np.linspace(max(low, grid[best] - coarse_step),
                       min(high, grid[best] + coarse_step), refine_steps)
    fine_model = predicted_brightness(fine, frame_starts, exposure_s, t,
                                      cumulative)
    fine_residuals, fine_gains = _residuals_after_linear_fit(fine_model,
                                                            brightness)
    fine_best = int(np.argmin(fine_residuals))
    t0 = float(fine[fine_best])
    residual = float(fine_residuals[fine_best])

    far = np.abs(grid - t0) > alias_guard
    margin = float(residuals[far].min() / residual) if far.any() and residual > 0 \
        else float('inf')

    depth = float(np.median(residuals) / residual) if residual > 0 \
        else float('inf')
    gain = float(fine_gains[fine_best])
    model_best = predicted_brightness(np.array([t0]), frame_starts, exposure_s,
                                      t, cumulative)[0]
    offset = float(brightness.mean() - gain * model_best.mean())
    return OffsetFit(t0, residual, margin, gain, offset, grid, residuals,
                     depth=depth)


# ------------------------------------------------- the sync-cable path
def frame_windows_from_sync(t, sync_volts, threshold=None):
    """Exposure windows straight off an ExposureActive pulse train.

    Unused by the optical route this repo actually takes - no such cable exists
    in the lab - but this is what a hardware sync would give, and it remains
    the best independent check if one is ever made up. The threshold defaults
    to halfway between the observed low and high levels.
    """
    t = np.asarray(t, dtype=float)
    sync = np.asarray(sync_volts, dtype=float)
    if threshold is None:
        threshold = 0.5 * (np.percentile(sync, 1) + np.percentile(sync, 99))
    high = sync > threshold
    rising = np.nonzero(~high[:-1] & high[1:])[0]
    falling = np.nonzero(high[:-1] & ~high[1:])[0]
    if rising.size == 0:
        raise ValueError(
            f'no rising edge above {threshold:.3g} V in the sync channel - the '
            f'camera was not pulsing, or the wrong channel was recorded')
    falling = falling[falling > rising[0]]          # ignore a partial first pulse
    n = min(rising.size, falling.size)
    if n == 0:
        raise ValueError('found a rising edge but no falling edge after it')
    return np.column_stack([t[rising[:n]], t[falling[:n]]])


def align_frames(windows, meta, period_tolerance=0.25):
    """Match sync-pulse windows to recorded frames, or say where it went wrong.

    Asserts one window per frame. When the counts differ, the camera timestamps
    say where the loss happened, so the mismatch is reported against a named
    frame instead of silently shifting every later frame by one.
    """
    windows = np.asarray(windows)
    starts = frame_start_times(meta)
    if len(windows) == len(starts):
        return {'windows': windows, 'frame_starts': starts,
                'n_frames': len(starts), 'dropped': dropped_frames(meta)}

    drops = dropped_frames(meta)
    edge_periods = np.diff(windows[:, 0])
    frame_periods = np.diff(starts)
    expected = float(np.median(frame_periods)) if frame_periods.size else None
    suspects = [int(i) for i, gap in enumerate(edge_periods)
                if expected and abs(gap - expected) > period_tolerance * expected]
    raise ValueError(
        f'{len(windows)} sync pulses but {len(starts)} recorded frames. '
        f'BlockID gaps say frames were lost {drops or "nowhere"}; pulse periods '
        f'deviate from {expected} s after pulse index {suspects or "nowhere"}.')


# ---------------------------------------------------------------- self-test
def _synthetic_trace(duration=1.0, dt=1e-5, pair_gap=0.032, fsr=0.21,
                     fwhm=0.0015, seed=0, regular=False):
    """A stand-in cavity spectrum: 0th/1st pairs repeating at the FSR.

    With `regular=True` every period is identical, which makes the trace
    perfectly aliased - a real sweep is not, because the ramp is not perfectly
    linear and the coupling drifts across it, so successive pairs differ in
    spacing and in height. Both cases are exercised by the self-test.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, duration, dt)
    signal = np.full_like(t, 0.05)
    centre, index = 0.06, 0
    while centre < duration:
        # a slightly non-linear ramp and a drifting coupling: enough to break
        # the alias, far less than the variation in the real recordings
        stretch = 1.0 if regular else 1.0 + 0.04 * index
        envelope = 1.0 if regular else 1.0 - 0.12 * index
        for offset, height in ((0.0, 1.0), (pair_gap * stretch, 0.55)):
            gamma = fwhm / 2
            signal += (height * envelope * gamma ** 2
                       / ((t - (centre + offset)) ** 2 + gamma ** 2))
        centre += fsr * stretch
        index += 1
    signal += rng.normal(0, 0.002, t.size)
    return ScopeTrace(t, signal, 's', 'V')


def _self_test():
    print('mode_video_sync self-test')
    rng = np.random.default_rng(7)
    trace = _synthetic_trace()

    # --- the offset fit, including a dropped frame ------------------------
    period, exposure, n_frames = 0.0100406, 0.0099, 60
    true_t0 = 0.2345
    starts = np.arange(n_frames) * period
    keep = np.ones(n_frames, dtype=bool)
    keep[42] = False                       # the camera dropped frame 42
    starts = starts[keep]

    cumulative = _cumulative(trace.t, trace.signal)
    clean = predicted_brightness(np.array([true_t0]), starts, exposure,
                                 trace.t, cumulative)[0]
    span = clean.max() - clean.min()
    margins = {}
    for noise_frac, tolerance in ((0.01, 5e-4), (0.03, 1e-3), (0.10, 3e-3)):
        observed = 3.3 * clean + 12.0 + rng.normal(0, noise_frac * span,
                                                   clean.size)
        fit = fit_time_offset(starts, exposure, observed, trace)
        error = abs(fit.t0 - true_t0)
        print(f'  noise {noise_frac:>5.0%}: t0 error {error * 1e3:7.4f} ms, '
              f'depth {fit.depth:7.1f}x, margin {fit.margin:6.1f}x, '
              f'gain {fit.gain:.3f}')
        assert fit.locked, f'should have locked on at {noise_frac:.0%}'
        assert error < tolerance, f'{noise_frac:.0%}: {error * 1e3:.4f} ms'
        assert fit.trustworthy, f'margin {fit.margin} at {noise_frac:.0%}'
        assert fit.gain > 0, 'gain should recover positive'
        margins[noise_frac] = fit.margin

    # `margin` has to fire on the two ways this fit genuinely cannot know the
    # answer. Both are realistic, and both would otherwise return a confident
    # number that happens to be wrong.

    # 1. A burst too short to see more than one repeat of a regular sweep: every
    #    FSR away is an equally good offset. Record edges are what rescue the
    #    long bursts above, and a short one does not reach them.
    regular = _synthetic_trace(regular=True)
    short_starts = np.arange(12) * period
    short_clean = predicted_brightness(
        np.array([0.27]), short_starts, exposure, regular.t,
        _cumulative(regular.t, regular.signal))[0]
    short_observed = short_clean + rng.normal(
        0, 0.01 * (short_clean.max() - short_clean.min()), short_clean.size)
    short_fit = fit_time_offset(short_starts, exposure, short_observed, regular)
    long_margin = margins[0.01]   # same 1% noise as the short burst below
    print(f'  12-frame burst on a regular sweep: margin '
          f'{short_fit.margin:.2f}x, against {long_margin:.0f}x for the '
          f'{starts.size}-frame burst above')
    # Not an outright failure - the ends of the record and the Lorentzian tails
    # of neighbouring orders still distinguish the offsets a little - but the
    # confidence collapses by orders of magnitude, which is the point: margin
    # tracks how much evidence the burst actually contains.
    assert short_fit.margin < 0.05 * long_margin, \
        f'a burst spanning one FSR should be far less certain ' \
        f'({short_fit.margin} vs {long_margin})'

    # 2. A burst that saw no resonance at all - beam blocked, laser unlocked,
    #    or the sweep missed. There is no structure to align, so no offset is
    #    better than any other.
    flat = np.full(starts.size, 40.0) + rng.normal(0, 0.05, starts.size)
    flat_fit = fit_time_offset(starts, exposure, flat, trace)
    print(f'  featureless burst: depth {flat_fit.depth:.2f}x, margin '
          f'{flat_fit.margin:.2f}x, locked={flat_fit.locked}, '
          f'unique={flat_fit.unique}')
    assert not flat_fit.locked, \
        f'a featureless burst must not read as locked (depth {flat_fit.depth})'
    assert not flat_fit.trustworthy

    # 3. A long record of a repeating sweep: locked on, but not unique. This is
    #    the situation the first real capture landed in, and the two flags have
    #    to separate it from the featureless case above rather than lumping
    #    both under "not trusted" - here the alignment is right and only the
    #    choice of repetition is open.
    long_trace = _synthetic_trace(duration=12.0, regular=True)
    long_clean = predicted_brightness(
        np.array([4.0]), starts, exposure, long_trace.t,
        _cumulative(long_trace.t, long_trace.signal))[0]
    long_observed = long_clean + rng.normal(
        0, 0.02 * (long_clean.max() - long_clean.min()), long_clean.size)
    long_fit = fit_time_offset(starts, exposure, long_observed, long_trace)
    print(f'  long regular record: depth {long_fit.depth:.1f}x, margin '
          f'{long_fit.margin:.2f}x, locked={long_fit.locked}, '
          f'unique={long_fit.unique}')
    assert long_fit.locked, 'it did find the sweep'
    assert not long_fit.unique, 'but a repeating sweep leaves it aliased'

    # a burst longer than the record cannot be placed
    try:
        fit_time_offset(np.arange(300) * period, exposure,
                        np.zeros(300), trace)
    except ValueError as error:
        assert 'cannot be made to overlap' in str(error)
        print('  over-long burst rejected with a usable message')
    else:
        raise AssertionError('an over-long burst should have been rejected')

    # --- windows and lookup, at the edges ---------------------------------
    windows = frame_windows(starts, exposure, true_t0)
    assert windows.shape == (starts.size, 2)
    first = windows[0]
    assert frame_at_time(windows, first[0]) == 0, 'window start is inclusive'
    assert frame_at_time(windows, first[1] - 1e-12) == 0
    assert frame_at_time(windows, first[1] + 1e-9) != 0, 'window end is exclusive'
    assert frame_at_time(windows, windows[-1, 1] + 1.0) is None
    assert nearest_frame(windows, windows[-1, 1] + 1.0) == len(windows) - 1
    gap_time = windows[0, 1] + 0.5 * (windows[1, 0] - windows[0, 1])
    assert frame_at_time(windows, gap_time) is None, 'dead time is not a frame'
    print('  frame_at_time correct at window edges and in dead time')

    # --- brightness and masking -------------------------------------------
    frames = rng.integers(0, 4, size=(8, 6, 6)).astype(np.uint8)
    frames[:, 2:4, 2:4] += np.arange(8, dtype=np.uint8)[:, None, None] * 20
    mask = varying_pixel_mask(frames)
    assert mask[2:4, 2:4].all(), 'the varying block must be inside the mask'
    assert mask.sum() < mask.size, 'the mask must exclude something'
    masked = frame_brightness(frames, mask)
    full = frame_brightness(frames)
    assert masked.max() - masked.min() > full.max() - full.min(), \
        'masking should raise the dynamic range'
    try:
        frame_brightness(frames, np.zeros((6, 6), dtype=bool))
    except ValueError as error:
        assert 'selects no pixels' in str(error)
    print('  masking raises the dynamic range and rejects an empty mask')

    # --- camera timestamps -------------------------------------------------
    meta = [{'block_id': i, 'camera_timestamp_ns': int(i * period * 1e9),
             'host_time_s': 0.0} for i in range(10)]
    assert np.allclose(frame_start_times(meta), np.arange(10) * period)
    assert dropped_frames(meta) == []
    gappy = [row for i, row in enumerate(meta) if i != 4]
    drops = dropped_frames(gappy)
    assert len(drops) == 1 and drops[0]['n_missing'] == 1, drops
    print('  chunk timestamps and BlockID gap detection agree')

    # --- the sync-cable path ----------------------------------------------
    t = np.arange(0.0, 0.5, 1e-5)
    pulse = ((t % period) < exposure).astype(float) * 3.3
    windows_from_pulses = frame_windows_from_sync(t, pulse)
    measured = np.median(windows_from_pulses[:, 1] - windows_from_pulses[:, 0])
    assert abs(measured - exposure) < 2e-5, measured
    assert abs(np.median(np.diff(windows_from_pulses[:, 0])) - period) < 2e-5
    try:
        frame_windows_from_sync(t, np.zeros_like(t))
    except ValueError as error:
        assert 'no rising edge' in str(error)
    pulse_meta = [{'block_id': i, 'camera_timestamp_ns': int(i * period * 1e9),
                   'host_time_s': 0.0}
                  for i in range(len(windows_from_pulses))]
    aligned = align_frames(windows_from_pulses, pulse_meta)
    assert aligned['n_frames'] == len(windows_from_pulses)
    try:
        align_frames(windows_from_pulses[:-3], pulse_meta)
    except ValueError as error:
        assert 'sync pulses but' in str(error)
    print('  sync-pulse windows recover the exposure and the period')

    # --- CSV units ---------------------------------------------------------
    import tempfile
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / 'trace.csv'
        path.write_text('Time,Channel A,Channel D\n(ms),(V),(mV)\n\n'
                        '-500.0,0.1,1000.0\n0.0,0.2,2000.0\n500.0,0.3,3000.0\n',
                        encoding='utf-8')
        loaded = load_scope_csv(path)
        assert np.allclose(loaded.t, [-0.5, 0.0, 0.5]), loaded.t
        assert np.allclose(loaded.signal, [1.0, 2.0, 3.0]), loaded.signal
        path.write_text('Time,Channel D\n(s),(V)\n\n0.0,1.0\n1.0,2.0\n',
                        encoding='utf-8')
        assert np.allclose(load_scope_csv(path).t, [0.0, 1.0])
    print('  CSV units row honoured: ms -> s and mV -> V')

    print('self-test passed')


def load_session_trace(session_path):
    """The scope trace a Phase 2 capture recorded for itself, as a ScopeTrace.

    Phase 1 captures have no such file - the scope side was a .psdata exported
    by hand - and raise here, which is what tells the caller to pass --scope.
    """
    session_path = Path(session_path)
    if session_path.is_dir():
        candidates = sorted(session_path.glob('*_session.json'))
        if len(candidates) != 1:
            raise FileNotFoundError(
                f'expected one *_session.json in {session_path}, '
                f'found {len(candidates)}')
        session_path = candidates[0]
    session = json.loads(session_path.read_text(encoding='utf-8'))
    if 'scope' not in session:
        raise FileNotFoundError(
            f'{session_path.name} has no scope trace of its own - it was made '
            f'with PicoScope 7 driving the scope. Pass the .psdata with '
            f'--scope instead.')
    data = np.load(session_path.parent / session['scope']['file'])
    return ScopeTrace(data['t'], data['signal'], 's', 'V', session_path)


def refine_session(session_path, window_s=0.25, verbose=True):
    """Improve a Phase 2 capture's host-clock offset with the optical fit.

    Optional by design. Driving both instruments from one process already puts
    frame 0 within a few milliseconds, which is a fraction of a frame and good
    enough for most work; this earns the last two orders of magnitude, and -
    more usefully - reports `depth`, which says whether the camera and the
    scope actually saw the same thing at all.

    Because `t0_host` is already close, the search is a narrow window around it
    rather than the whole record, so the free-spectral-range aliasing that
    plagues a blind search over a long trace does not arise.
    """
    session_path = Path(session_path)
    trace = load_session_trace(session_path)
    session, frames = load_session(session_path)
    starts = frame_start_times(session['meta'])
    exposure = float(session['exposure_s'])
    t0_host = float(session['sync']['t0_host_s'])

    burst = float(starts[-1] + exposure)
    low = max(float(trace.t[0]), t0_host - window_s)
    high = min(float(trace.t[-1] - burst), t0_host + window_s)
    if high <= low:
        raise ValueError(
            f'the host offset {t0_host * 1e3:.1f} ms leaves no room for the '
            f'{burst:.3f} s burst inside a {trace.duration:.3f} s block')

    results = {}
    for name in ('masked', 'full'):
        brightness = np.array(session[f'brightness_{name}'], dtype=float)
        results[name] = fit_time_offset(starts, exposure, brightness, trace,
                                        search=(low, high))
    best_name = max(results, key=lambda k: results[k].depth)
    best = results[best_name]
    correction = best.t0 - t0_host

    if verbose:
        print(f'{trace}')
        print(f'  host-clock offset : {t0_host * 1e3:9.3f} ms')
        for name, fit in results.items():
            print(f'  {name:>6} fit       : {fit.t0 * 1e3:9.3f} ms  '
                  f'depth {fit.depth:7.1f}x  margin {fit.margin:6.2f}x')
        print(f'  correction        : {correction * 1e3:+9.3f} ms '
              f'({abs(correction) / exposure:.3f} of an exposure)')
        frames_off = abs(correction) / exposure
        if not best.locked:
            print('  ! the fit did not lock onto the sweep - check that the '
                  'burst caught resonances and that the camera sees the same '
                  'light as channel D')
        elif frames_off > 2.0:
            print(f'  ! the correction is {frames_off:.1f} frames, far more '
                  f'than the 0.78 frames of jitter the calibrated host clock '
                  f'should show. Either the fit found an alias, or '
                  f'HOST_T0_BIAS_S needs re-measuring.')
        else:
            print(f'  the calibrated host clock alone was within '
                  f'{frames_off:.2f} of a frame')

    session['sync'].update({
        't0_fitted_s': best.t0,
        't0_correction_s': correction,
        'fit_series': best_name,
        'fit_depth': best.depth,
        'fit_margin': best.margin,
        'fit_locked': bool(best.locked),
    })
    json_path = (session_path if session_path.suffix == '.json'
                 else sorted(session_path.glob('*_session.json'))[0])
    # numpy scalars reach here through the fit results; a default converter
    # keeps one of them from costing the whole session file
    json_path.write_text(
        json.dumps(session, indent=1,
                   default=lambda v: v.item() if hasattr(v, 'item') else str(v)),
        encoding='utf-8')
    if verbose:
        print(f'  wrote t0_fitted_s to {json_path.name}')
    return {'session': session, 'frames': frames, 'trace': trace,
            'fits': results, 'best': best, 'best_series': best_name,
            't0_host': t0_host, 'correction': correction,
            'windows': frame_windows(starts, exposure, best.t0)}


def session_windows(session_path, prefer_fitted=True):
    """Frame exposure windows in the scope's timebase, for a Phase 2 capture.

    Uses the refined offset when refine_session() has been run and the plain
    host-clock one otherwise, so a viewer can work either way.
    """
    session_path = Path(session_path)
    if session_path.is_dir():
        session_path = sorted(session_path.glob('*_session.json'))[0]
    session = json.loads(session_path.read_text(encoding='utf-8'))
    sync = session['sync']
    key = 't0_fitted_s' if (prefer_fitted and 't0_fitted_s' in sync) \
        else 't0_host_s'
    starts = frame_start_times(session['meta'])
    return frame_windows(starts, float(session['exposure_s']), float(sync[key])), key


def fit_session(session_path, scope_path, signal_column=SIGNAL_COLUMN,
                verbose=True, buffer=None):
    """Align a capture folder with a scope recording; return everything found.

    `scope_path` may be a `.psdata` - it is converted through the same helper
    the other scope scripts use, so the waveform buffer is chosen the usual way
    - or a CSV that is read directly.

    Both brightness series are fitted. The masked one is far less noisy but
    excludes light the photodiode still sees, so whichever reports the better
    margin is the one to believe, and a disagreement between them is worth
    looking at rather than averaging away.
    """
    from utilities.utils import psdata_buffer_csvs

    scope_path = Path(scope_path)
    if scope_path.suffix.lower() == '.psdata':
        csvs = [Path(c) for c in psdata_buffer_csvs(scope_path)]
    else:
        csvs = [scope_path]
    if buffer is not None:
        csvs = [csvs[buffer - 1]]

    session, frames = load_session(session_path)

    starts = frame_start_times(session['meta'])
    exposure = float(session['exposure_s'])
    drops = dropped_frames(session['meta'])
    if verbose:
        print(f'session {session["frames_shape"]} {session["frames_dtype"]}, '
              f'exposure {exposure * 1e3:.2f} ms, '
              f'burst {starts[-1] + exposure:.3f} s')
        if drops:
            print(f'  dropped frames: {drops} - the fit uses the camera '
                  f'timestamps, so this shifts nothing')

    # A .psdata can hold several waveform buffers and only one of them was
    # rolling when the burst happened. Rather than ask, fit them all: a buffer
    # that does not contain the burst has nothing to lock onto and reports a
    # margin near 1, so the margin identifies the right one by itself.
    per_buffer = {}
    burst = float(starts[-1] + exposure)
    for csv in csvs:
        trace = load_scope_csv(csv, signal_column=signal_column)
        if trace.duration < burst:
            # A .psdata usually ends with a partial buffer, cut short when the
            # recording was stopped. It cannot contain the burst; skip it
            # rather than failing the whole fit.
            if verbose:
                print(f'  {csv.name}: {trace.duration:.3f} s - shorter than the '
                      f'{burst:.3f} s burst, skipped')
            continue
        fits = {}
        for name in ('masked', 'full'):
            brightness = np.array(session[f'brightness_{name}'], dtype=float)
            fits[name] = fit_time_offset(starts, exposure, brightness, trace)
        per_buffer[csv.name] = {'trace': trace, 'fits': fits}
        if verbose:
            print(f'  {csv.name}: {trace.t.size} samples, '
                  f'{trace.duration:.2f} s')
            for name, fit in fits.items():
                flags = ('locked' if fit.locked else 'NOT LOCKED')
                flags += ', unique' if fit.unique else ', aliased'
                print(f'      {name:>6}: t0 = {fit.t0:10.6f} s, depth '
                      f'{fit.depth:7.1f}x, margin {fit.margin:5.2f}x  [{flags}]')

    # Rank buffers by depth, not margin: depth says how well the burst matches
    # this record, while margin only says whether a rival offset within the same
    # record fits too - which is a property of the sweep's periodicity, not of
    # whether this is the right buffer.
    if not per_buffer:
        raise ValueError(
            f'no waveform buffer is as long as the {burst:.3f} s burst. Record '
            f'a longer scope trace, or fewer frames (--frames).')
    winner = max(per_buffer,
                 key=lambda k: max(f.depth for f in per_buffer[k]['fits'].values()))
    trace = per_buffer[winner]['trace']
    results = per_buffer[winner]['fits']
    if verbose and len(per_buffer) > 1:
        print(f'  -> buffer {winner} contains the burst')

    best_name = max(results, key=lambda k: results[k].margin)
    best = results[best_name]
    disagreement = abs(results['masked'].t0 - results['full'].t0)
    if verbose:
        print(f'  best: {best_name}, t0 = {best.t0:.6f} s')
        print(f'  the two series disagree by {disagreement * 1e3:.4f} ms '
              f'({disagreement / exposure:.3f} of an exposure)')
        if disagreement > 0.25 * exposure:
            print('  ! they disagree by more than a quarter of a frame. Check '
                  'whether a mode is falling outside the mask, or saturating.')
        if not best.locked:
            print('  ! the fit did not lock onto the sweep at all. Did the '
                  'burst overlap the scope record, and did any resonance land '
                  'in it?')
        elif not best.unique:
            print(f'  ! locked on, but {best.margin:.2f}x from the best rival '
                  f'offset - the sweep repeats and this record is long enough '
                  f'to contain many equally good positions. The alignment '
                  f'within the sweep is still right; which repetition is not '
                  f'pinned. Use a shorter scope record to remove the ambiguity.')

    windows = frame_windows(starts, exposure, best.t0)
    return {'trace': trace, 'session': session, 'frames': frames,
            'frame_starts': starts, 'exposure_s': exposure, 'fits': results,
            'best': best, 'best_series': best_name, 'windows': windows,
            'disagreement_s': disagreement, 'dropped': drops,
            'buffer': winner, 'per_buffer': per_buffer}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--self-test', action='store_true',
                        help='run the offline checks and exit')
    parser.add_argument('--session',
                        help='capture folder or *_session.json from '
                             'mode_video_capture.py')
    parser.add_argument('--scope',
                        help='the .psdata or .csv recorded at the same time')
    parser.add_argument('--signal-column', default=SIGNAL_COLUMN,
                        help=f'scope column to align against '
                             f'(default {SIGNAL_COLUMN})')
    parser.add_argument('--refine', action='store_true',
                        help='refine a Phase 2 capture that recorded its own '
                             'scope trace; no --scope needed')
    parser.add_argument('--window', type=float, default=0.25,
                        help='half-width in seconds of the search around the '
                             'host-clock offset when refining (default 0.25)')
    parser.add_argument('--buffer', type=int, default=None,
                        help='use only this waveform buffer of a .psdata '
                             '(1-based); by default every buffer is fitted and '
                             'the one whose margin is best is the one used')
    args = parser.parse_args()
    if args.self_test:
        _self_test()
        return
    if args.refine:
        if not args.session:
            parser.error('--refine needs --session')
        refine_session(args.session, window_s=args.window)
        return
    if args.session and args.scope:
        fit_session(args.session, args.scope, args.signal_column,
                    buffer=args.buffer)
        return
    parser.print_help()


if __name__ == '__main__':
    main()
