"""Tests of the nested alignment loop on the simulator (analytic physics, fake clock).

    python alignment/test_nested.py

Two scenarios (kalishlot/MECHANICAL_CONTROL.md, section 3.e):
- ridge: coupling = exp(-d M d) in (sim_mirror_phi, sim_lens_y), 1/e at 40 units
  along (1, 0.5) and at 4 across. The start sits on the crest, 30 units out,
  where every single-axis step loses coupling.
- symmetry: the splitting is signed, 4e-3 FSR per unit of phi and -4e-3 of
  lens_y, so lens_y can cancel what phi breaks. The start is symmetric. The
  symmetry metric is injected from the truth ('asymmetry' = |split| in
  linewidths + noise): the scope's split_linewidths reads 0 in the simulator
  when one first-order family dominates, so it cannot test the loop.
"""

import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alignment.loop import AlignmentLoop, Objective  # noqa: E402
from alignment.metrics import Metric  # noqa: E402
from alignment.nested import Level, NestedLoop  # noqa: E402
from alignment.run_alignment_sim import FakeClock, build, make_measure  # noqa: E402
from simulation.cavity_physics import AnalyticPhysics, ridge_curvature  # noqa: E402

QUIET = dict(print_fn=None)
PHI, THETA, LENS_Y = 'sim_mirror_phi', 'sim_mirror_theta', 'sim_lens_y'
RIDGE_DIR = (1.0, 0.5)                 # lens_y moves 0.5 per unit of phi along the ridge
SPLIT = {PHI: 4e-3, LENS_Y: -4e-3}     # FSR per unit: symmetric while lens_y follows phi 1:1
SYM = Objective('asymmetry', maximize=False, good_enough=1.0)


def ridge_case(seed=1, start=30.0):
    clock = FakeClock()
    physics = AnalyticPhysics(curvature=ridge_curvature([PHI, LENS_Y], RIDGE_DIR, 40.0, 4.0))
    axes, setup = build(clock, seed=seed, physics=physics)
    u = np.array(RIDGE_DIR) / np.linalg.norm(RIDGE_DIR)
    axes[PHI].move_by(start * u[0], who='misalign')
    axes[LENS_Y].move_by(start * u[1], who='misalign')
    return clock, axes, setup


def ridge_levels():
    return [Level([PHI], Objective('intensity')),
            Level([LENS_Y], Objective('intensity'), steps={LENS_Y: 1.0})]


def symmetry_case(seed=1):
    clock = FakeClock()
    axes, setup = build(clock, seed=seed, physics=AnalyticPhysics(signed_split=SPLIT))
    axes[PHI].move_by(6.0, who='misalign')
    axes[LENS_Y].move_by(6.0, who='misalign')          # 1:1, so still symmetric
    axes[THETA].move_by(-4.0, who='misalign')
    inner = make_measure(setup, clock)
    rng = np.random.default_rng(seed)

    def measure():
        cm = inner()
        cm.metrics['asymmetry'] = Metric('asymmetry', true_asymmetry(setup) + abs(rng.normal(0, 0.05)),
                                         0.05, 10, True)
        return cm
    return clock, axes, setup, measure


def true_asymmetry(setup):
    return abs(setup.mode_state().split) / (2 * setup.scan.linewidth_fsr)


@lru_cache(maxsize=None)
def ridge_runs():
    """(start coupling, plain result, nested result, nested loop), run once for several tests."""
    clock, axes, setup = ridge_case()
    c0 = setup.mode_state().coupling
    AlignmentLoop(axes, make_measure(setup, clock), Objective('intensity'), tiers=[[PHI, LENS_Y]],
                  steps={LENS_Y: 1.0}, max_evaluations=150, **QUIET).run()
    plain = setup.mode_state().coupling
    clock, axes, setup = ridge_case()
    loop = NestedLoop(axes, make_measure(setup, clock), ridge_levels(), max_evaluations=150, **QUIET)
    summary = loop.run()
    return c0, plain, setup.mode_state().coupling, loop, summary


def test_ridge_plain_loop_stalls():
    """The problem: from the crest every single-axis step loses, the joint loop stops early."""
    c0, plain, _, _, _ = ridge_runs()
    assert c0 < 0.6 and plain < 0.8, (c0, plain)


def test_ridge_nested_recovers():
    c0, plain, nested, _, summary = ridge_runs()
    assert nested > 0.95 and nested > plain + 0.15, (c0, plain, nested, summary)
    assert summary['evaluations'] <= 150


def test_predictor_learns_ridge_direction():
    """d(lens_y*)/d(phi) is the ridge's slope, 0.5."""
    *_, summary = ridge_runs()
    k = summary['predictor'][0]['K'][0][0]
    assert abs(k - RIDGE_DIR[1]) < 0.2 * RIDGE_DIR[1], summary['predictor']


def test_keep_symmetric():
    """Maximize coupling with the tilts while lens_y keeps the mode symmetric."""
    clock, axes, setup, measure = symmetry_case()
    c0 = setup.mode_state().coupling
    loop = NestedLoop(axes, measure, [Level([PHI, THETA], Objective('intensity')),
                                      Level([LENS_Y], SYM, steps={LENS_Y: 1.0})],
                      max_evaluations=150, **QUIET)
    summary = loop.run()
    nested = setup.mode_state().coupling
    assert c0 < 0.45 and nested > 0.95, (c0, nested, summary)
    assert true_asymmetry(setup) <= 1.0, true_asymmetry(setup)
    accepted = [p for p in loop.profile[0] if p['accepted']]
    assert accepted and all(p['metrics']['asymmetry'] <= 1.1 for p in accepted), \
        [p['metrics']['asymmetry'] for p in accepted]

    # the same request as a constraint on one joint loop: rejected moves, no restoring
    clock, axes, setup, measure = symmetry_case()
    AlignmentLoop(axes, measure, Objective('intensity', constraints=(('asymmetry', '<=', 1.0),)),
                  tiers=[[PHI, THETA, LENS_Y]], steps={LENS_Y: 1.0}, max_evaluations=150, **QUIET).run()
    assert setup.mode_state().coupling < nested - 0.02, (setup.mode_state().coupling, nested)


def test_profile_noise_flag():
    """On (default): the scatter of the inner runs is estimated and raises the margin. Off: not measured."""
    *_, loop, summary = ridge_runs()
    assert summary['profile_noise'].get(0, 0) > 0, summary['profile_noise']
    assert loop.margin(0) == loop.accept_sigmas * max(loop.noise[0], summary['profile_noise'][0])
    assert any(h['kind'] == 'profile noise' for h in loop.history)
    clock, axes, setup = ridge_case()
    off = NestedLoop(axes, make_measure(setup, clock), ridge_levels(), max_evaluations=30,
                     profile_noise=False, **QUIET)
    summary = off.run()
    assert summary['profile_noise'] == {} and off.margin(0) == off.accept_sigmas * off.noise[0]
    assert not any(h['kind'] == 'profile noise' for h in off.history)


def test_stop_returns_all_axes_to_best():
    clock, axes, setup = ridge_case()
    inner = make_measure(setup, clock)
    holder = {}

    def measure():
        result = inner()
        if len(holder['loop'].history) == 40:
            holder['loop'].stop()
        return result
    loop = NestedLoop(axes, measure, ridge_levels(), max_evaluations=150, **QUIET)
    holder['loop'] = loop
    summary = loop.run()
    assert summary['aborted'], summary
    pos = {n: a.commanded for n, a in axes.items()}
    assert all(abs(pos[n] - summary['best'][n]) < 1e-9 for n in pos), (pos, summary['best'])
    assert summary['best'][LENS_Y] != ridge_case()[1][LENS_Y].commanded   # the inner axis is part of best


def test_frozen_inner_axis_and_shared_budget():
    clock, axes, setup = ridge_case()
    axes[LENS_Y].set_frozen(True, who='test')
    moves = axes[LENS_Y]._raw.moves
    summary = NestedLoop(axes, make_measure(setup, clock), ridge_levels(), max_evaluations=40, **QUIET).run()
    assert axes[LENS_Y]._raw.moves == moves
    assert summary['evaluations'] <= 40, summary['evaluations']


def test_three_levels_smoke():
    clock, axes, setup = ridge_case()
    axes[THETA].move_by(3.0, who='misalign')
    levels = [Level([THETA], Objective('intensity'))] + ridge_levels()
    levels[1].max_evaluations = 25      # else the first middle-level solve takes the whole budget
    loop = NestedLoop(axes, make_measure(setup, clock), levels, max_evaluations=80, **QUIET)
    summary = loop.run()
    assert summary['evaluations'] <= 80
    assert {h['level'] for h in loop.history} >= {0, 1, 2}, {h['level'] for h in loop.history}
    pos = {n: a.commanded for n, a in axes.items()}
    assert all(abs(pos[n] - summary['best'][n]) < 1e-9 for n in pos)


def main():
    tests = [(k, v) for k, v in globals().items() if k.startswith('test_') and callable(v)]
    for name, fn in tests:
        fn()
        print(f'ok  {name}')
    print(f'{len(tests)} nested tests passed')


if __name__ == '__main__':
    main()
