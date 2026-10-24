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
class MarkerTarget:
    x: float                          # sensor px (uncropped, unbinned), as kalishlot stores markers
    y: float
    tolerance_px: float = 3.0
    confirm_repeats: int = 2          # extra captures in place before "reached" is believed
    constraints: tuple = ()
    label: str = ''
    metric_x: str = 'sensor_x'
    metric_y: str = 'sensor_y'

    @classmethod
    def from_markers(cls, path, name, device=None, **kw):
        m = find_marker(load_markers(path, device), name)
        return cls(float(m['x']), float(m['y']), label=m.get('label') or m.get('id', ''), **kw)

    @property
    def metric(self):                 # for the loop's printouts
        return f'distance to marker {self.label or (round(self.x), round(self.y))} [px]'

    @property
    def goal(self):                   # for the loop's Jacobian steering
        return self.x, self.y

    def position(self, ev):
        """Measured (x, y) of an Evaluation, or None when invalid."""
        x, y = ev.metrics.get(self.metric_x), ev.metrics.get(self.metric_y)
        return None if x is None or y is None else (x, y)

    def distance(self, x, y):
        return math.hypot(x - self.x, y - self.y)

    def evaluate(self, cm):
        mx, my = cm.metrics.get(self.metric_x), cm.metrics.get(self.metric_y)
        if mx is None or my is None or not (mx.valid and my.valid):
            bad = mx if (mx is None or not mx.valid) else my
            return None, False, f'mode position invalid: {bad.reason if bad else "missing"}'
        score = -self.distance(mx.value, my.value)
        for name, op, thr in self.constraints:
            c = cm.metrics.get(name)
            if c is None or not c.valid:
                return score, False, f'constraint {name} invalid'
            if not (c.value >= thr if op == '>=' else c.value <= thr):
                return score, False, f'constraint {name} {op} {thr} violated ({c.value:.4g})'
        return score, True, ''

    def mean_distance(self, evaluations):
        """Distance of the mean measured centre over evaluations taken at one point."""
        pts = [(e.metrics.get(self.metric_x), e.metrics.get(self.metric_y)) for e in evaluations]
        pts = [(x, y) for x, y in pts if x is not None and y is not None]
        if not pts:
            return math.inf
        return self.distance(sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))

    def reached(self, evaluations):
        """True when every evaluation is feasible and the mean centre is within tolerance."""
        return (bool(evaluations) and all(e.feasible for e in evaluations)
                and self.mean_distance(evaluations) <= self.tolerance_px)
