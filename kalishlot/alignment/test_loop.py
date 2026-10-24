"""Fast tests of the alignment loop on the simulator (analytic physics, fake clock).

    python alignment/test_loop.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alignment.loop import AlignmentLoop, Objective, scan_axis  # noqa: E402
from alignment.metrics import frame_to_sensor  # noqa: E402
from alignment.run_alignment_sim import FakeClock, build, leave_marker, make_measure  # noqa: E402
from alignment.targets import MarkerTarget, find_marker, load_markers  # noqa: E402
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
    clock, axes, setup = setup_case({'sim_mirror_phi': 7.0, 'sim_mirror_theta': -5.0})
    before = setup.mode_state().coupling
    loop = AlignmentLoop(axes, make_measure(setup, clock), Objective('intensity'), tiers=TIERS,
                         max_evaluations=120, **QUIET)
    summary = loop.run()
    after = setup.mode_state().coupling
    assert before < 0.4 and after > 0.95, (before, after, summary)
    assert not summary['aborted']


def test_frozen_axis_never_moves():
    clock, axes, setup = setup_case({'sim_mirror_phi': 6.0, 'sim_mirror_theta': 4.0})
    axes['sim_mirror_theta'].set_frozen(True, who='test')
    moves = axes['sim_mirror_theta']._raw.moves
    AlignmentLoop(axes, make_measure(setup, clock), tiers=TIERS, max_evaluations=60, **QUIET).run()
    assert axes['sim_mirror_theta']._raw.moves == moves


def test_measures_only_after_settling():
    clock, axes, setup = setup_case({'sim_mirror_phi': 5.0})
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
    clock, axes, setup = setup_case({'sim_mirror_phi': 6.0})
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
    clock, axes, setup = setup_case({'sim_mirror_phi': 30.0})     # beyond the stable range
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
    clock, axes, setup = setup_case({'sim_mirror_phi': 4.0})
    ax = axes['sim_mirror_phi']
    rows = scan_axis(ax, [33.5 + k for k in range(9)], make_measure(setup, clock))
    best = max(rows, key=lambda r: r['score'] or -1)
    assert abs(best['target'] - 37.5) <= 1.0, [(r['target'], r['score']) for r in rows]
    assert ax.commanded == best['target']


def test_marker_target_reached():
    """Leave a marker where the mode sits, go back, steer the mode onto it."""
    clock, axes, setup = setup_case({})
    measure = make_measure(setup, clock, with_frames=True)
    target = leave_marker(axes, setup, measure, {'sim_mirror_phi': 4.0, 'sim_mirror_theta': -3.0},
                          tolerance_px=2.5)
    true_dist = lambda: target.distance(*frame_to_sensor(*setup.true_centre_px(), setup.sensor_geometry()))
    before = true_dist()
    loop = AlignmentLoop(axes, measure, target, tiers=[['sim_mirror_phi', 'sim_mirror_theta']],
                         max_evaluations=60, **QUIET)
    summary = loop.run()
    assert before > 8, before
    assert summary['target_reached'], summary
    assert true_dist() < 2 * target.tolerance_px, (before, true_dist(), summary)
    assert loop.history[-1].kind == 'confirm target'


def test_marker_lookup_and_sensor_units(tmp=None):
    import json
    import tempfile
    d = Path(tmp or tempfile.mkdtemp())
    state = {'ximea': {'markers': [{'id': 'a1', 'label': 'arm 45 cm', 'x': 800.5, 'y': 1100.0, 'r': 9}]},
             'basler': {'markers': [{'id': 'b1', 'label': 'other', 'x': 1, 'y': 2, 'r': 3}]}}
    (d / 'device_state.json').write_text(json.dumps(state))
    (d / 'markers.json').write_text(json.dumps({'format': 'kalishlot-markers', 'markers':
                                                state['ximea']['markers']}))
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from utilities.system_state import SystemState
    SystemState(state).save(d / 'device_state_new.json')       # the server now writes a SystemState
    assert find_marker(load_markers(d / 'device_state_new.json'), 'b1')['device'] == 'basler'
    for f in ('device_state.json', 'device_state_new.json', 'markers.json'):
        t = MarkerTarget.from_markers(d / f, 'arm 45 cm')
        assert (t.x, t.y, t.label) == (800.5, 1100.0, 'arm 45 cm')
    assert find_marker(load_markers(d / 'device_state.json'), 'b1')['label'] == 'other'
    try:
        find_marker(load_markers(d / 'device_state.json'), 'missing')
        raise AssertionError('expected ValueError')
    except ValueError:
        pass
    # binned ROI pixel -> unbinned sensor pixel (the markers' units)
    assert frame_to_sensor(10, 20, {'offset_x': 366, 'offset_y': 544, 'binning_x': 2,
                                    'binning_y': 2}) == (752.5, 1128.5)


def main():
    tests = [(k, v) for k, v in globals().items() if k.startswith('test_') and callable(v)]
    for name, fn in tests:
        fn()
        print(f'ok  {name}')
    print(f'{len(tests)} loop tests passed')


if __name__ == '__main__':
    main()
