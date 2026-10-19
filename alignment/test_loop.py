"""Fast tests of the alignment loop on the simulator (analytic physics, fake clock).

    python alignment/test_loop.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alignment.loop import AlignmentLoop, Objective, scan_axis  # noqa: E402
from alignment.run_alignment_sim import FakeClock, build, make_measure  # noqa: E402
from motion.axes_config import TIERS  # noqa: E402
from simulation.cavity_physics import AnalyticPhysics  # noqa: E402

QUIET = dict(print_fn=None)


def setup_case(misalign, seed=1):
    clock = FakeClock()
    axes, setup = build(clock, seed=seed, physics=AnalyticPhysics())
    for n, d in misalign.items():
        axes[n].move_by(d, who='misalign')
    return clock, axes, setup


def test_recovers_coupling():
    clock, axes, setup = setup_case({'sim_mirror_tilt_x': 7.0, 'sim_mirror_tilt_y': -5.0})
    before = setup.mode_state().coupling
    loop = AlignmentLoop(axes, make_measure(setup, clock), Objective('intensity'), tiers=TIERS,
                         max_evaluations=120, **QUIET)
    summary = loop.run()
    after = setup.mode_state().coupling
    assert before < 0.4 and after > 0.95, (before, after, summary)
    assert not summary['aborted']


def test_frozen_axis_never_moves():
    clock, axes, setup = setup_case({'sim_mirror_tilt_x': 6.0, 'sim_mirror_tilt_y': 4.0})
    axes['sim_mirror_tilt_y'].set_frozen(True, who='test')
    moves = axes['sim_mirror_tilt_y']._raw.moves
    AlignmentLoop(axes, make_measure(setup, clock), tiers=TIERS, max_evaluations=60, **QUIET).run()
    assert axes['sim_mirror_tilt_y']._raw.moves == moves


def test_measures_only_after_settling():
    clock, axes, setup = setup_case({'sim_mirror_tilt_x': 5.0})
    inner = make_measure(setup, clock)

    def measure():
        for a in axes.values():
            assert clock.t >= a.ready_at - 1e-9, f'{a.name} still settling'
        cap_vibrating = any(clock.t - a._raw.last_move_t < a.limits.vibration_settle_s
                            for a in axes.values())
        assert not cap_vibrating
        return inner()
    AlignmentLoop(axes, measure, tiers=TIERS, max_evaluations=30, **QUIET).run()


def test_abort_returns_to_best():
    clock, axes, setup = setup_case({'sim_mirror_tilt_x': 6.0})
    inner = make_measure(setup, clock)
    holder = {}

    def measure():
        result = inner()
        if len(holder['loop'].history) == 12:
            holder['loop'].stop()
        return result
    loop = AlignmentLoop(axes, measure, tiers=TIERS, max_evaluations=100, **QUIET)
    holder['loop'] = loop
    summary = loop.run()
    assert summary['aborted']
    pos = {n: a.commanded for n, a in axes.items()}
    assert all(abs(pos[n] - summary['best'][n]) < 1e-9 for n in pos), (pos, summary['best'])


def test_signal_lost_at_start():
    clock, axes, setup = setup_case({'sim_mirror_tilt_x': 30.0})     # beyond the stable range
    start = {n: a.commanded for n, a in axes.items()}
    loop = AlignmentLoop(axes, make_measure(setup, clock), tiers=TIERS, **QUIET)
    try:
        loop.run()
    except RuntimeError as e:
        assert 'signal lost' in str(e)
    else:
        raise AssertionError('expected signal lost')
    assert {n: a.commanded for n, a in axes.items()} == start


def test_scan_axis_finds_peak():
    clock, axes, setup = setup_case({'sim_mirror_tilt_x': 4.0})
    ax = axes['sim_mirror_tilt_x']
    rows = scan_axis(ax, [33.5 + k for k in range(9)], make_measure(setup, clock))
    best = max(rows, key=lambda r: r['score'] or -1)
    assert abs(best['target'] - 37.5) <= 1.0, [(r['target'], r['score']) for r in rows]
    assert ax.commanded == best['target']


def main():
    tests = [(k, v) for k, v in globals().items() if k.startswith('test_') and callable(v)]
    for name, fn in tests:
        fn()
        print(f'ok  {name}')
    print(f'{len(tests)} loop tests passed')


if __name__ == '__main__':
    main()
