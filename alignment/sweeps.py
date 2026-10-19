"""Split a scope capture into monotonic laser-scan sweeps.

The laser frequency is scanned by a triangle ramp on its temperature. The scope
records that ramp on the aux channel. Two facts from the recorded sessions
(2026-10-11/12) shape this module:

- The aux channel is the *drive voltage*. The laser frequency follows it with a
  thermal lag, so the real frequency turnaround happens ~0.5-0.9 s after the
  aux extremum. Around it the frequency stands still: no modes pass (signal
  "lost"), then they pass slowly (peaks look amplified). Hence a guard window
  after every aux turnaround, default 1.0 s.
- Up- and down-sweeps have different time-per-FSR (~0.42 s vs ~0.52 s), so they
  are never mixed. Each Sweep carries its direction.

Direction is +1 when the aux voltage rises, -1 when it falls. Whether +1 means
rising optical frequency is not assumed anywhere.
"""

from dataclasses import dataclass, asdict

import numpy as np

DEFAULT_GUARD_S = 1.0       # dropped after each aux turnaround (measured lag ~0.5-0.9 s)
DEFAULT_SMOOTH_S = 0.05     # moving-average window for the aux channel
DEFAULT_MIN_SWEEP_S = 0.5   # shorter usable pieces are discarded
MIN_SWING_FRACTION = 0.05   # an extremum counts only if the ramp then moves this
                            # fraction of its observed span (rejects noise wiggles)


@dataclass
class Sweep:
    t_start: float          # first usable time (after the guard)
    t_end: float            # last usable time
    direction: int          # +1 aux rising, -1 aux falling
    turnaround_before: float | None   # aux extremum time that opened it, if seen

    @property
    def duration(self):
        return self.t_end - self.t_start

    def mask(self, t):
        t = np.asarray(t)
        return (t >= self.t_start) & (t <= self.t_end)

    def to_dict(self):
        return asdict(self)


def smooth(y, n):
    """Moving average with edge-corrected normalization (same length as y)."""
    n = max(int(n), 1)
    if n == 1:
        return np.asarray(y, dtype=float)
    kernel = np.ones(n)
    num = np.convolve(y, kernel, mode='same')
    den = np.convolve(np.ones_like(y, dtype=float), kernel, mode='same')
    return num / den


def find_turnarounds(t, aux, smooth_s=DEFAULT_SMOOTH_S,
                     min_swing_fraction=MIN_SWING_FRACTION):
    """Times of the aux ramp's extrema, as a list of (time, kind) with kind
    'max' or 'min'. A zig-zag detector with hysteresis: an extremum is accepted
    only once the ramp has moved away from it by min_swing_fraction of the
    observed span, so noise on a flat stretch never produces turnarounds."""
    t = np.asarray(t, dtype=float)
    aux = np.asarray(aux, dtype=float)
    if t.size < 3:
        return []
    dt = float(np.median(np.diff(t)))
    a = smooth(aux, round(smooth_s / dt))
    span = float(a.max() - a.min())
    if span <= 0:
        return []
    swing = min_swing_fraction * span

    out = []
    trend = 0                 # +1 rising, -1 falling, 0 not decided yet
    hi_i = lo_i = 0           # running extrema since the last reversal
    for i in range(1, a.size):
        if a[i] > a[hi_i]:
            hi_i = i
        if a[i] < a[lo_i]:
            lo_i = i
        if trend >= 0 and a[i] < a[hi_i] - swing:
            if trend == 1:
                out.append((float(t[hi_i]), 'max'))
            trend, lo_i = -1, i
        elif trend <= 0 and a[i] > a[lo_i] + swing:
            if trend == -1:
                out.append((float(t[lo_i]), 'min'))
            trend, hi_i = 1, i
    return out


def split_sweeps(t, aux, guard_s=DEFAULT_GUARD_S, smooth_s=DEFAULT_SMOOTH_S,
                 min_sweep_s=DEFAULT_MIN_SWEEP_S):
    """Usable monotonic sweeps of a capture, in time order.

    Each piece between two aux turnarounds (or the capture's ends) loses
    `guard_s` after its opening turnaround. A piece at the very start of the
    capture has no observed turnaround and keeps its start: the ramp's own
    extremum may lie just before the capture, which `aux_near_extreme` can
    catch when the ramp limits are known.
    """
    t = np.asarray(t, dtype=float)
    aux = np.asarray(aux, dtype=float)
    turns = find_turnarounds(t, aux, smooth_s)
    edges = [(float(t[0]), None)] + [(tt, kind) for tt, kind in turns] + [(float(t[-1]), 'end')]
    sweeps = []
    for (t0, kind0), (t1, _) in zip(edges[:-1], edges[1:]):
        start = t0 + guard_s if kind0 in ('max', 'min') else t0
        if t1 - start < min_sweep_s:
            continue
        m = (t >= start) & (t <= t1)
        idx = np.nonzero(m)[0]
        direction = 1 if aux[idx[-1]] > aux[idx[0]] else -1
        if kind0 == 'max':
            direction = -1
        elif kind0 == 'min':
            direction = 1
        sweeps.append(Sweep(start, t1, direction, t0 if kind0 in ('max', 'min') else None))
    return sweeps


def aux_near_extreme(aux_value, ramp_min, ramp_max, slope_v_per_s, guard_s=DEFAULT_GUARD_S):
    """True when an aux value is within `guard_s` of travel from a ramp limit,
    i.e. a sweep starting there may still be inside the post-turnaround lag."""
    reach = abs(slope_v_per_s) * guard_s
    return aux_value - ramp_min < reach or ramp_max - aux_value < reach


def _self_test():
    # 0.05 Hz triangle, 3 Vpp, sampled 1 kHz for 25 s: turnarounds every 10 s
    t = np.arange(0, 25, 1e-3)
    period = 20.0
    phase = (t / period) % 1.0
    aux = 1.5 * (4 * np.abs(phase - 0.5) - 1)       # max at t=0,20; min at t=10
    aux += np.random.default_rng(0).normal(0, 0.01, t.size)
    turns = find_turnarounds(t, aux)
    kinds = [k for _, k in turns]
    assert kinds == ['min', 'max'], turns
    assert abs(turns[0][0] - 10) < 0.1 and abs(turns[1][0] - 20) < 0.1, turns
    sw = split_sweeps(t, aux, guard_s=1.0)
    assert [s.direction for s in sw] == [-1, 1, -1], sw
    assert abs(sw[1].t_start - 11) < 0.1 and abs(sw[2].t_start - 21) < 0.1, sw
    # a capture that never turns: one sweep, full length
    sw = split_sweeps(t[:5000], aux[:5000] * 0 + np.linspace(0, 1, 5000))
    assert len(sw) == 1 and sw[0].direction == 1 and sw[0].t_start == 0.0
    print('sweeps self-test OK')


if __name__ == '__main__':
    _self_test()
