"""Procedure targets: a target value that needs moves, not just one capture.

A Level of alignment/nested.py takes `measure=procedure`. The procedure gets a
handle `run` and returns a CaptureMetrics, which the level's objective then
scores as usual:

    run.capture(label)    settle every axis, capture once (counted, logged)
    run.move_to({axis: position}), run.move_by({axis: delta})   through the safe axes
    run.positions(), run.axes
    run.move_back()       put every axis the procedure moved back

Whatever the procedure does, the loop puts every axis it moved back afterwards,
also when it is stopped or fails in the middle.

Example: is the optical axis parallel to the stage under the large mirror?

    levels = [Level(['arm_phi', 'arm_theta'],
                    VectorTarget(('drift_spot_y', 'drift_spot_z'), (0.0, 0.0), tolerance=2.0),
                    measure=move_and_compare('stage', -10.0)),
              Level(['mirror_phi', 'mirror_theta'], Objective('split', maximize=False))]
"""

import math

from alignment.metrics import CaptureMetrics, Metric


def move_and_compare(axis, travel, metrics=('spot_y', 'spot_z'), name='drift'):
    """Procedure: capture, move `axis` by `travel`, capture again, move back.

    Returns the first capture's metrics plus, for every m in `metrics`,
    '<name>_<m>' = (after - before), and '<name>' = the length of that vector.
    E.g. the mode's position on the large mirror before and after moving the
    mirror's stage: the drift is zero when the stage moves along the optical axis."""
    def procedure(run):
        before = run.capture('before')
        run.move_by({axis: travel})
        after = run.capture(f'{axis} moved')
        run.move_back()
        out = dict(before.metrics)
        parts = []
        for m in metrics:
            key = f'{name}_{m}'
            a, b = before.metrics.get(m), after.metrics.get(m)
            if a is None or b is None or not (a.valid and b.valid):
                bad = a if (a is None or not a.valid) else b
                out[key] = Metric(key, reason=f'{m} invalid: {bad.reason if bad else "missing"}')
                continue
            out[key] = Metric(key, b.value - a.value, math.hypot(a.spread, b.spread), min(a.n, b.n), True)
            parts.append(out[key])
        if len(parts) == len(metrics):
            out[name] = Metric(name, math.sqrt(sum(p.value ** 2 for p in parts)),
                               math.sqrt(sum(p.spread ** 2 for p in parts)), min(p.n for p in parts), True)
        else:
            out[name] = Metric(name, reason='a position was invalid')
        return CaptureMetrics(metrics=out)
    procedure.__name__ = f'move_and_compare({axis}, {travel})'
    return procedure
