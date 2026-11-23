"""Open-loop tracking (alignment/loop.py) against last night's failure.

The 2026-11-23 23:58 run: four Picomotors, two cameras, tolerances 1/1/3/3 px,
start ~20 px off. Every round trip moved the spots 8-14 px (forward and
backward steps differ), so the loop's return-to-incumbent search chased a
ghost and ended on budget. Here the same, simulated: open-loop axes whose
backward steps are 30% short and whose moves vary by 30% (step_jitter), 0.1 px
camera noise, J of the lab's size. A fixed asymmetry alone does not reproduce
the failure (the old search then still lands); the random part does.

    python alignment/test_track.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alignment.loop import AlignmentLoop  # noqa: E402
from alignment.metrics import CaptureMetrics, Metric  # noqa: E402
from alignment.targets import VectorTarget  # noqa: E402
from motion import AxisLimits, build_axis  # noqa: E402

# px per step, from the run's probes (#4-#7): basler x, y; ximea x, y vs ch1..ch4
J = np.array([[0.090, 0.012, 0.012, 0.060],
              [0.006, -0.097, 0.072, 0.007],
              [0.023, -0.002, -0.141, -0.007],
              [-0.017, -0.003, -0.011, 0.117]])
GOAL = np.array([1077.1, 705.8, 1051.0, 844.6])
TOL = (1.0, 1.0, 3.0, 3.0)
NAMES = ('bx', 'by', 'xx', 'xy')


def world(backward_scale=0.7, seed=0, step_jitter=0.3):
    lim = AxisLimits(min=-10000, max=10000, max_step=200, max_slew=1e9, settle_s=0.0,
                     vibration_settle_s=0.0)
    cfg = {f'pico{c}': {'driver': 'simulated', 'address': f'sim:{c}',
                        'options': dict(units='steps', kind='steps', start=0.0, has_readback=False,
                                        backward_scale=backward_scale, step_jitter=step_jitter,
                                        seed=seed * 10 + c), 'limits': lim}
           for c in range(1, 5)}
    axes = {n: build_axis(n, cfg) for n in cfg}
    for a in axes.values():
        a.open()
    rng = np.random.default_rng(seed)
    start = GOAL + np.array([-17.0, 7.0, -8.4, 4.3])       # the run's starting miss

    def spots():
        return start + J @ np.array([a._raw.true_position for a in axes.values()])

    def measure():
        q = spots() + rng.normal(0, 0.1, 4)
        return CaptureMetrics(metrics={k: Metric(k, v, valid=True) for k, v in zip(NAMES, q)})
    return axes, measure, spots


def target():
    return VectorTarget(NAMES, tuple(GOAL), tolerances=TOL)


def within(spots, slack=1.1):
    """The TRUE spots within tolerance (the loop judges noisy means: allow a hair)."""
    return bool(np.all(np.abs(spots() - GOAL) <= slack * np.array(TOL)))


def test_open_loop_axes_are_detected_and_tracked():
    axes, measure, spots = world()
    loop = AlignmentLoop(axes, measure, target(), max_evaluations=60, print_fn=None)
    s = loop.run()
    assert s['tracked'] and s['target_reached'] and s['end_reason'] == 'reached', s
    assert within(spots), spots() - GOAL
    assert s['evaluations'] <= 25, s


def test_tracking_never_goes_back():
    axes, measure, spots = world()
    seen = []
    orig = AlignmentLoop.measure

    def spy(self, kind):
        seen.append(dict(self.positions()))
        return orig(self, kind)
    AlignmentLoop.measure = spy
    try:
        AlignmentLoop(axes, measure, target(), max_evaluations=60, print_fn=None).run()
    finally:
        AlignmentLoop.measure = orig
    points = [tuple(round(v, 6) for v in p.values()) for p in seen]
    revisits = [p for i, p in enumerate(points) if p in points[:i] and p != points[i - 1]]
    assert not revisits, revisits                   # measured in place again: fine; returned: never


def test_the_old_search_fails_where_tracking_succeeds():
    """The reproduction: forced back to the return-to-incumbent search, it misses."""
    reached = {}
    for track in (False, 'auto'):
        hits = 0
        for seed in range(8):
            axes, measure, spots = world(seed=seed)
            s = AlignmentLoop(axes, measure, target(), max_evaluations=80, print_fn=None,
                              track=track).run()
            hits += within(spots)
        reached[track] = hits
    assert reached['auto'] == 8 and reached[False] <= 6, reached


def test_repeatable_axes_keep_the_old_search():
    axes, measure, spots = world(backward_scale=1.0, step_jitter=0.0)
    import dataclasses
    for a in axes.values():                         # e.g. a piezo driver echoing its voltage
        a._raw.info = dataclasses.replace(a._raw.info, has_readback=True)
    s = AlignmentLoop(axes, measure, target(), max_evaluations=60, print_fn=None).run()
    assert not s['tracked'] and s['target_reached'], s


def test_previous_jacobian_skips_the_probes():
    axes, measure, spots = world(seed=1)
    first = AlignmentLoop(axes, measure, target(), max_evaluations=60, print_fn=None).run()
    jac = first['jacobian']
    assert first['target_reached'] and jac and not first['jacobian_reused'], first
    # in the metrics' own units (px per step), roughly the model's J: Broyden refines it only
    # along the moves made, and the steps here vary by 30%
    err = np.linalg.norm(np.array(jac['J']) - J) / np.linalg.norm(J)
    assert err < 0.6, err
    for a in axes.values():                          # someone walks the spots away again
        a.move_by(150.0, who='test')
    kinds = []
    loop = AlignmentLoop(axes, measure, target(), max_evaluations=60, print_fn=None, jacobian=jac)
    orig = loop.measure
    loop.measure = lambda kind: (kinds.append(kind), orig(kind))[1]
    second = loop.run()
    assert second['jacobian_reused'] and second['target_reached'], second
    assert 'probe' not in kinds, kinds
    assert second['evaluations'] < first['evaluations'], (second['evaluations'], first['evaluations'])
    # other tolerances: the matrix is kept in px per step, so it still applies
    t2 = VectorTarget(NAMES, tuple(GOAL), tolerances=(2.0, 2.0, 2.0, 2.0))
    third = AlignmentLoop(axes, measure, t2, max_evaluations=60, print_fn=None,
                          jacobian=second['jacobian']).run()
    assert third['jacobian_reused'], third


def test_a_matrix_for_other_signals_is_ignored():
    axes, measure, spots = world()
    jac = {'metrics': ['a', 'b', 'c', 'd'], 'axes': list(axes), 'J': J.tolist(), 'probes': {}}
    s = AlignmentLoop(axes, measure, target(), max_evaluations=60, print_fn=None, jacobian=jac).run()
    assert not s['jacobian_reused'] and s['target_reached'], s


def test_stuck_ends_with_a_reason():
    axes, measure, spots = world()
    t = VectorTarget(NAMES, tuple(GOAL), tolerances=(0.01,) * 4)   # far below the noise
    s = AlignmentLoop(axes, measure, t, max_evaluations=200, print_fn=None).run()
    assert not s['target_reached'] and s['end_reason'].startswith('stuck'), s
    assert s['evaluations'] < 200, s


def main():
    tests = [(k, v) for k, v in globals().items() if k.startswith('test_') and callable(v)]
    for name, fn in tests:
        fn()
        print(f'ok  {name}')
    print(f'{len(tests)} tracking tests passed')


if __name__ == '__main__':
    main()
