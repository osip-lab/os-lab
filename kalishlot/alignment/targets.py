"""Objective: bring the 0th-order mode onto a camera marker.

In the lab the mode is often steered by hand until the centre of its Gaussian
fit sits on a kalishlot camera marker left there earlier ("the mode sat here
with the long arm at 45 cm"). MarkerTarget makes that an alignment objective:

    target = MarkerTarget.from_markers('device_state.json', 'arm 45 cm', tolerance_px=3)
    loop = AlignmentLoop(axes, measure, target, tiers=[['m1_phi', 'm1_theta']])

The axes the loop may use are the tiers given (freeze the rest). The measured
position is the median sensor_x/_y of a capture (alignment/metrics.py: the fit
centre of every usable 0th-order frame, in the markers' units, i.e. uncropped
and unbinned sensor pixels).

Noise. The fit centre lands somewhere slightly different in every blink of
the mode (pointing jitter, vibrations, fit noise), so:
- one evaluation = the median over all 0th-order blinks of a capture, not one fit;
- the loop first steers (alignment/loop.py): it probes each axis with a step
  grown until the position moves by more than 3x its noise, then moves
  straight at the marker by least squares, refining the estimate every move;
- the score is minus the distance to the marker; the loop only accepts a move
  that beats the incumbent by more than the measured noise floor of that
  distance, so it does not chase jitter;
- "reached" needs the distance to be within `tolerance_px` for the MEAN centre
  of the evaluation plus `confirm_repeats` further captures taken in place;
  a single lucky capture does not end the run. The loop then stops (no
  verification move is needed: it is already standing there, confirmed).
- a tolerance tighter than the noise can never be confirmed; the run then ends
  on the evaluation budget or a stall, with summary['target_reached'] False.

Constraints work as for Objective, e.g. constraints=(('intensity', '>=', 0.3),)
keeps the mode resonating while it is being walked.
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path


def load_markers(path, device=None):
    """Markers from a kalishlot marker file (save_markers) or from kalishlot's
    device_state.json (every camera's markers, or only `device`'s)."""
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    if isinstance(data, list):
        return data
    if data.get('format') == 'kalishlot-markers':
        return list(data.get('markers') or [])
    if data.get('format') == 'os-lab-system-state':   # device_state.json is a SystemState (utilities/system_state.py)
        data = data.get('values') or {}
    markers = []                                   # device_state.json: {device_id: snapshot}
    for dev, snap in data.items():
        if (device is None or dev == device) and isinstance(snap, dict):
            markers += [dict(m, device=dev) for m in snap.get('markers') or []]
    return markers


def find_marker(markers, name):
    """The marker whose label (or id) is `name`; ValueError if none or ambiguous."""
    hits = [m for m in markers if m.get('label') == name] or [m for m in markers if m.get('id') == name]
    if not hits:
        labels = sorted({m.get('label') or m.get('id') for m in markers})
        raise ValueError(f'no marker {name!r}; there are: {labels}')
    if len(hits) > 1:
        raise ValueError(f'{len(hits)} markers are called {name!r} (on {[m.get("device") for m in hits]}); '
                         f'pass device= or use the id')
    return hits[0]


@dataclass
class VectorTarget:
    """Bring a measured vector (several metrics) to a goal, e.g. the drift of the mode
    when the large mirror's stage moves to (0, 0):

        VectorTarget(('drift_spot_y', 'drift_spot_z'), (0.0, 0.0), tolerance=2.0, unit='um')

    The score is minus the distance to the goal. Because the target also reports the
    measured vector (position) and the goal, the loops steer at it with a Jacobian
    (AlignmentLoop, and the outer levels of alignment/nested.py) instead of only
    stepping one axis at a time."""
    metrics: tuple
    goal: tuple
    tolerance: float = 1.0
    confirm_repeats: int = 2          # extra captures in place before "reached" is believed
    constraints: tuple = ()
    label: str = ''
    unit: str = ''

    @property
    def metric(self):                 # for the loop's printouts
        unit = f' [{self.unit}]' if self.unit else ''
        return f'distance of ({", ".join(self.metrics)}) to {self.label or tuple(self.goal)}{unit}'

    def position(self, ev):
        """The measured vector of an Evaluation, or None when invalid."""
        q = tuple(ev.metrics.get(m) for m in self.metrics)
        return None if any(v is None for v in q) else q

    def distance(self, *q):
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(q, self.goal)))

    def evaluate(self, cm):
        ms = [cm.metrics.get(m) for m in self.metrics]
        bad = next((m for m in ms if m is None or not m.valid), False)
        if bad is not False:
            return None, False, f'{self.metrics} invalid: {bad.reason if bad else "missing"}'
        score = -self.distance(*(m.value for m in ms))
        for name, op, thr in self.constraints:
            c = cm.metrics.get(name)
            if c is None or not c.valid:
                return score, False, f'constraint {name} invalid'
            if not (c.value >= thr if op == '>=' else c.value <= thr):
                return score, False, f'constraint {name} {op} {thr} violated ({c.value:.4g})'
        return score, True, ''

    def value_of(self, score):
        return -score

    def mean_distance(self, evaluations):
        """Distance of the mean measured vector over evaluations taken at one point."""
        pts = [q for q in (self.position(e) for e in evaluations) if q is not None]
        if not pts:
            return math.inf
        return self.distance(*(sum(p[i] for p in pts) / len(pts) for i in range(len(self.metrics))))

    def reached(self, evaluations):
        """True when every evaluation is feasible and the mean vector is within tolerance."""
        return (bool(evaluations) and all(e.feasible for e in evaluations)
                and self.mean_distance(evaluations) <= self.tolerance)


class MarkerTarget(VectorTarget):
    """VectorTarget of the mode's fit centre (sensor_x, sensor_y) to a marker, in px."""

    def __init__(self, x, y, tolerance_px=3.0, confirm_repeats=2, constraints=(), label='',
                 metric_x='sensor_x', metric_y='sensor_y'):
        super().__init__((metric_x, metric_y), (float(x), float(y)), tolerance_px, confirm_repeats,
                         constraints, label, 'px')

    x = property(lambda self: self.goal[0])
    y = property(lambda self: self.goal[1])
    tolerance_px = property(lambda self: self.tolerance)
    metric_x = property(lambda self: self.metrics[0])
    metric_y = property(lambda self: self.metrics[1])

    @classmethod
    def from_markers(cls, path, name, device=None, **kw):
        m = find_marker(load_markers(path, device), name)
        return cls(float(m['x']), float(m['y']), label=m.get('label') or m.get('id', ''), **kw)

    @property
    def metric(self):                 # for the loop's printouts
        return f'distance to marker {self.label or (round(self.x), round(self.y))} [px]'
