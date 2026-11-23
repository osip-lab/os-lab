"""Tests of the aligner's recipe (table -> loop objects) and per-signal tolerances.

    python alignment/test_recipe.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alignment import recipe  # noqa: E402
from alignment.loop import AlignmentLoop, Evaluation, Objective  # noqa: E402
from alignment.metrics import CaptureMetrics, Metric  # noqa: E402
from alignment.targets import VectorTarget  # noqa: E402
from alignment.test_com_measure import sim_axes  # noqa: E402

CTRL = [{'axis': 'a1'}]


def num(v):
    return {'kind': 'number', 'value': v}


def refused(level, words):
    try:
        recipe.validate(level)
    except ValueError as e:
        assert words in str(e), (words, str(e))
    else:
        raise AssertionError(f'accepted: {level}')


def test_validate():
    goto = {'signal': 's1', 'goal': 'goto', 'value': num(1.0), 'tolerance': 1}
    recipe.validate({'rows': [goto], 'controls': CTRL})
    refused({'rows': [], 'controls': CTRL}, 'at least one signal')
    refused({'rows': [goto], 'controls': []}, 'control')
    refused({'rows': [goto, {'signal': 's2', 'goal': 'max'}], 'controls': CTRL}, 'not both')
    refused({'rows': [{'signal': 's1', 'goal': 'max'}, {'signal': 's2', 'goal': 'min'}],
             'controls': CTRL}, 'only one')
    refused({'rows': [{'signal': 's1', 'goal': 'ge', 'value': num(1)}], 'controls': CTRL}, 'guards alone')
    refused({'rows': [dict(goto, tolerance=0)], 'controls': CTRL}, 'tolerance')
    refused({'rows': [dict(goto, value={'kind': 'number'})], 'controls': CTRL}, 'give a number')
    refused({'rows': [goto, dict(goto)], 'controls': CTRL}, 'two go-to rows')
    refused({'rows': [{'signal': 's1', 'goal': 'ge', 'value': {'kind': 'hold'}}, goto],
             'controls': CTRL}, 'only for go-to')
    recipe.validate({'rows': [{'signal': 's1', 'goal': 'max', 'value': {}},
                              {'signal': 's2', 'goal': 'ge', 'value': num(3)}], 'controls': CTRL})


def test_build_objective():
    level = {'rows': [{'signal': 'x', 'goal': 'goto', 'value': num(10), 'tolerance': 1},
                      {'signal': 'y', 'goal': 'goto', 'value': num(20), 'tolerance': 4},
                      {'signal': 'n', 'goal': 'ge', 'value': num(100)}], 'controls': CTRL}
    t = recipe.build_objective(level, {0: 10.0, 1: 20.0, 2: 100.0})
    assert isinstance(t, VectorTarget) and t.metrics == ('x', 'y') and t.raw_goal == (10.0, 20.0)
    assert t.tolerances == (1.0, 4.0) and t.constraints == (('n', '>=', 100.0),)
    level = {'rows': [{'signal': 'p', 'goal': 'min', 'value': {}}], 'controls': CTRL}
    o = recipe.build_objective(level, {})
    assert isinstance(o, Objective) and o.metric == 'p' and not o.maximize and o.good_enough is None


def ev(**vals):
    return Evaluation(0, {}, 0.0, True, '', vals, 'trial')


def test_per_signal_tolerance():
    t = VectorTarget(('x', 'y'), (0.0, 0.0), tolerances=(1.0, 10.0))
    assert t.reached([ev(x=0.9, y=9.0)])
    assert not t.reached([ev(x=1.1, y=0.0)])        # x off by more than its own tolerance
    assert not t.reached([ev(x=0.0, y=10.5)])
    cm = CaptureMetrics(metrics={'x': Metric('x', 3.0, valid=True), 'y': Metric('y', 40.0, valid=True)})
    score, ok, _ = t.evaluate(cm)
    assert ok and abs(score + 5.0) < 1e-9             # sqrt(3^2 + 4^2) in units of tolerance
    plain = VectorTarget(('x', 'y'), (0.0, 0.0), tolerance=2.0)   # unchanged without tolerances
    assert plain.reached([ev(x=1.0, y=1.0)]) and plain.goal == (0.0, 0.0)


def test_steering_with_mixed_scales():
    """Two signals in very different units: px (tol 0.5) and a 'MHz' (tol 0.002)."""
    axes = dict(list(sim_axes().items())[:2])
    J = np.array([[0.05, 0.01], [0.00002, -0.00004]])
    goal = np.array([12.0, -0.08])

    def measure():
        p = np.array([a.position for a in axes.values()])
        q = J @ p
        return CaptureMetrics(metrics={'px': Metric('px', q[0], valid=True),
                                       'mhz': Metric('mhz', q[1], valid=True)})
    t = VectorTarget(('px', 'mhz'), tuple(goal), tolerances=(0.5, 0.002))
    s = AlignmentLoop(axes, measure, t, max_evaluations=60, print_fn=None).run()
    q = J @ np.array([a.position for a in axes.values()])
    assert s['target_reached'] and abs(q[0] - goal[0]) <= 0.5 and abs(q[1] - goal[1]) <= 0.002, (q, s)


def main():
    tests = [(k, v) for k, v in globals().items() if k.startswith('test_') and callable(v)]
    for name, fn in tests:
        fn()
        print(f'ok  {name}')
    print(f'{len(tests)} recipe tests passed')


if __name__ == '__main__':
    main()
