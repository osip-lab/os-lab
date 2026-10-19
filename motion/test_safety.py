"""Hardware-free tests of every SafeAxis / VirtualAxis safety rule.

    python -m motion.test_safety        (from the repo root)

Uses a fake clock (no real sleeping) and a recording driver, so each test can
check exactly what reached the "hardware".
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from motion import build_axis  # noqa: E402
from motion.axes_config import AXES  # noqa: E402
from motion.safety import AxisLimits, SafeAxis, SafetyError  # noqa: E402
from motion.simulated import SimulatedAxis  # noqa: E402
from motion.virtual import VirtualAxis  # noqa: E402


class FakeClock:
    def __init__(self):
        self.t = 0.0
        self.slept = 0.0

    def now(self):
        return self.t

    def sleep(self, dt):
        self.t += dt
        self.slept += dt


class Recording(SimulatedAxis):
    """SimulatedAxis that records every move_to it receives."""

    def __init__(self, *a, on_move=None, **kw):
        super().__init__(*a, **kw)
        self.sent = []
        self.on_move = on_move

    def move_to(self, target):
        self.sent.append(target)
        super().move_to(target)
        if self.on_move:
            self.on_move(len(self.sent))


LIM = AxisLimits(min=0.0, max=75.0, max_step=1.0, max_slew=10.0, settle_s=0.1,
                 vibration_settle_s=0.5)


def make(limits=LIM, start=10.0, **raw_kw):
    clock = FakeClock()
    raw = Recording('ax', start=start, now=clock.now, **raw_kw)
    ax = SafeAxis(raw, limits, sleep=clock.sleep, now=clock.now)
    ax.open()
    return ax, raw, clock


def refused(fn, *args, **kw):
    try:
        fn(*args, **kw)
    except SafetyError as e:
        return str(e)
    raise AssertionError(f'{getattr(fn, "__name__", fn)} was not refused')


def test_open_and_close_never_move():
    ax, raw, _ = make()
    assert raw.sent == [] and ax.commanded == 10.0
    ax.close()
    assert raw.sent == []


def test_out_of_range_rejected_not_clamped():
    ax, raw, _ = make()
    for bad in (-0.1, 75.01, 1e9):
        refused(ax.move_to, bad, who='t')
    for bad in (math.nan, math.inf, 'abc', None):
        refused(ax.move_to, bad, who='t')
    assert raw.sent == [] and ax.commanded == 10.0


def test_large_move_is_ramped():
    ax, raw, clock = make()
    r = ax.move_to(15.5, who='t')
    steps = [b - a for a, b in zip([10.0] + raw.sent, raw.sent)]
    assert raw.sent[-1] == 15.5 and r.commanded == 15.5
    assert all(abs(s) <= LIM.max_step + 1e-12 for s in steps), steps
    assert len(raw.sent) == 6
    assert clock.slept >= 5.5 / LIM.max_slew + LIM.settle_s - 1e-9
    assert r.travel == 5.5


def test_frozen():
    ax, raw, _ = make(limits=AxisLimits(**{**LIM.__dict__, 'frozen': True}))
    refused(ax.move_to, 11, who='t')
    refused(ax.set_frozen, False, who='t')          # config freeze cannot be undone at run time
    ax2, raw2, _ = make()
    ax2.set_frozen(True, who='ui')
    refused(ax2.move_by, 1, who='t')
    ax2.set_frozen(False, who='ui')
    ax2.move_by(1, who='t')
    assert raw.sent == [] and raw2.sent == [11.0]


def test_unknown_position_refused_until_declared():
    clock = FakeClock()
    raw = Recording('stick', has_readback=False, known_on_open=False, now=clock.now)
    ax = SafeAxis(raw, LIM, sleep=clock.sleep, now=clock.now)
    ax.open()
    assert not ax.position_known
    refused(ax.move_to, 5, who='t')
    refused(ax.set_known_position, 80, who='person')
    ax.set_known_position(4.0, who='person')
    ax.move_to(5, who='t')
    assert raw.sent == [5.0]


def test_not_open_refused():
    clock = FakeClock()
    ax = SafeAxis(Recording('x', start=1.0), LIM, sleep=clock.sleep, now=clock.now)
    refused(ax.move_to, 2, who='t')


def test_estop_stops_mid_ramp_and_latches():
    holder = {}
    ax, raw, _ = make(on_move=lambda n: n == 3 and holder['ax'].estop('test'))
    holder['ax'] = ax
    refused(ax.move_to, 20, who='t')
    assert len(raw.sent) == 3 and ax.commanded == raw.sent[-1] == 13.0   # held where it was
    refused(ax.move_to, 14, who='t')
    ax.reset_estop(who='person')
    raw.on_move = None
    ax.move_to(14, who='t')
    assert ax.commanded == 14.0


def test_anti_backlash_approach():
    lim = AxisLimits(**{**LIM.__dict__, 'approach': 1, 'approach_overshoot': 2.0})
    ax, raw, _ = make(limits=lim)
    ax.move_to(12, who='t')                         # moving up: direct
    assert raw.sent == [11.0, 12.0]
    raw.sent.clear()
    ax.move_to(11, who='t')                         # moving down: go to 9, come back up
    assert raw.sent == [11.0, 10.0, 9.0, 10.0, 11.0], raw.sent
    raw.sent.clear()
    ax.move_to(0.5, who='t')                        # overshoot clamped at min (0)
    assert min(raw.sent) == 0.0 and raw.sent[-1] == 0.5, raw.sent
    raw.sent.clear()
    ax.move_to(0.0, who='t')                        # target at min: no approach possible
    assert raw.sent == [0.0]


def test_travel_budget():
    ax, raw, _ = make(limits=AxisLimits(**{**LIM.__dict__, 'travel_budget': 5.0}))
    ax.move_to(13, who='t')
    refused(ax.move_to, 16, who='t')                # 3 + 3 > 5
    assert ax.commanded == 13.0
    ax.move_to(15, who='t')                         # 3 + 2 = 5: allowed


def test_vibration_settle_before_measurement():
    ax, raw, clock = make()
    ax.move_to(11, who='t')
    assert ax.ready_at >= clock.t + LIM.vibration_settle_s - 1e-9
    before = clock.t
    ax.wait_until_settled()
    assert clock.t - before >= LIM.vibration_settle_s - 1e-9


def test_move_log():
    ax, raw, _ = make()
    ax.move_to(11, who='optimizer')
    moves = [e for e in ax.log.entries if e['event'] == 'move']
    assert moves and moves[-1]['who'] == 'optimizer' and moves[-1]['commanded'] == 11.0
    assert moves[-1]['start'] == 10.0


def test_virtual_axis_checks_all_before_moving():
    a, ra, _ = make(start=10.0)
    b, rb, _ = make(start=74.5)
    walk = VirtualAxis('walk', [(a, 1.0), (b, 1.0)])
    refused(walk.move_by, 1.0, who='t')             # b would leave its range
    assert ra.sent == [] and rb.sent == []           # ... so a did not move either
    walk.move_by(-1.0, who='t')
    assert a.commanded == 9.0 and b.commanded == 73.5 and walk.position == -1.0
    b.set_frozen(True, who='ui')
    refused(walk.move_by, -1.0, who='t')


def test_simulator_repeatability_and_backlash():
    clock = FakeClock()
    raw = SimulatedAxis('s', start=10, backlash=0.6, repeatability_sigma=0.05, seed=1,
                        now=clock.now)
    ax = SafeAxis(raw, LIM, sleep=clock.sleep, now=clock.now)
    ax.open()
    up, down = [], []
    for _ in range(40):
        ax.move_to(12, who='t')
        ax.move_to(15, who='t')
        ax.move_to(14, who='t')
        down.append(raw.true_position)              # arrived moving down
        ax.move_to(12, who='t')
        ax.move_to(14, who='t')
        up.append(raw.true_position)                # arrived moving up
    import numpy as np
    gap = np.mean(down) - np.mean(up)
    assert abs(gap - 0.6) < 0.05, gap               # backlash shows as a direction-dependent offset
    assert 0.03 < np.std(up) < 0.08                  # random part remains even from one side


def test_config_axes_build():
    for name in AXES:
        ax = build_axis(name)
        ax.open()
        assert ax.position_known, name


def main():
    tests = [(k, v) for k, v in globals().items() if k.startswith('test_') and callable(v)]
    for name, fn in tests:
        fn()
        print(f'ok  {name}')
    print(f'{len(tests)} safety tests passed')


if __name__ == '__main__':
    main()
