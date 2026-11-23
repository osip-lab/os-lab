"""Hardware-free tests of the aligner box (adapters/aligner.py).

Two fake cameras (centre of mass + markers) whose spots move linearly with the
four simulated axes of a real MotionAxesAdapter, all in one fake registry.

    python test_aligner.py
"""

import random
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from adapters.aligner import AlignerAdapter  # noqa: E402
from adapters.motion_axes import MotionAxesAdapter  # noqa: E402
from motion import AxisLimits  # noqa: E402

J = np.array([[0.050, 0.004, 0.030, -0.002],
              [0.003, -0.045, 0.001, 0.028],
              [0.020, 0.001, 0.060, 0.003],
              [-0.002, 0.025, 0.004, 0.055]])
START = np.array([1000.0, 760.0, 1120.0, 790.0])
GOAL = {'basler_camera:1': (1077.1, 705.8), 'ximea_camera:2': (1051.0, 844.6)}


class FakeCamera:
    """What the aligner uses of a camera adapter: com_samples, com_describe,
    command('com_on'), describe()['markers']."""

    def __init__(self, device_id, rows, motion, slow_s=0.0):
        self.device_id, self.rows, self.motion = device_id, rows, motion
        self.com = False
        self.slow_s = slow_s
        self.rng = random.Random(hash(device_id) % 1000)

    def describe(self):
        x, y = GOAL[self.device_id]
        return {'markers': [{'id': 'm1', 'label': 'target', 'x': x, 'y': y}]}

    def com_describe(self):
        return {'com': self.com, 'com_cut': 0.2}

    def com_command(self, name, args):
        return None

    def command(self, name, args):
        assert name == 'com_on'
        self.com = True
        return {'ok': True}

    def com_samples(self, n=1, since=None, timeout=5.0):
        if not self.com:
            raise RuntimeError('the centre of mass is off on this camera')
        time.sleep(self.slow_s)
        p = np.array([self.motion.axes[f'p{c}'].position for c in range(1, 5)])
        s = START + J @ p
        x, y = s[self.rows[0]], s[self.rows[1]]
        return [{'valid': True, 'clipped': False, 'counts': 1e5, 'frame_time_s': time.time(),
                 'sensor_x': x + self.rng.gauss(0, 0.1), 'sensor_y': y + self.rng.gauss(0, 0.1)}
                for _ in range(n)]


def make_world(slow_s=0.0):
    lim = AxisLimits(min=-10000, max=10000, max_step=200, max_slew=1e9, settle_s=0.0,
                     vibration_settle_s=0.0)
    cfg = {f'p{c}': {'driver': 'simulated', 'address': f'sim:{c}',
                     'options': dict(units='steps', kind='position', start=0.0), 'limits': lim}
           for c in range(1, 5)}
    motion = MotionAxesAdapter('simulated', config=cfg)
    motion.open()
    devices = {'motion_axes:simulated': motion,
               'basler_camera:1': FakeCamera('basler_camera:1', (0, 1), motion, slow_s),
               'ximea_camera:2': FakeCamera('ximea_camera:2', (2, 3), motion, slow_s)}
    aligner = AlignerAdapter('main')
    aligner.registry = lambda: (devices, {})
    events = aligner.add_listener()
    return aligner, motion, devices, events


def setup_rows(tol=0.5):
    rows = []
    for dev in GOAL:
        for q, c in (('com_x', 'x'), ('com_y', 'y')):
            rows.append({'signal': f'{dev}/{q}', 'goal': 'goto', 'tolerance': tol,
                         'value': {'kind': 'marker', 'device_id': dev, 'marker': 'target', 'coord': c}})
    return {'levels': [{'rows': rows, 'controls': [{'axis': f'p{c}', 'step': None} for c in range(1, 5)]}],
            'fields': {'delay_s': 0.0, 'samples': 5, 'max_evaluations': 60}}


def wait_idle(aligner, timeout=30):
    deadline = time.time() + timeout
    while aligner.state != 'idle' or (aligner._worker and aligner._worker.is_alive()):
        assert time.time() < deadline, 'run did not end'
        time.sleep(0.02)


def spots(motion):
    p = np.array([motion.axes[f'p{c}'].position for c in range(1, 5)])
    return START + J @ p


def test_catalog_lists_signals_controls_markers():
    aligner, motion, devices, _ = make_world()
    cat = aligner.describe()['catalog']
    ids = {s['id'] for s in cat['signals']}
    assert {'basler_camera:1/com_x', 'ximea_camera:2/com_y'} <= ids
    assert [c['axis'] for c in cat['controls']] == ['p1', 'p2', 'p3', 'p4']
    assert cat['markers']['basler_camera:1'][0]['label'] == 'target'


def test_dry_run_moves_nothing():
    aligner, motion, devices, _ = make_world()
    aligner.command('set_setup', {'setup': setup_rows()})
    rows = aligner.command('dry_run', {})['rows']
    assert [round(r['miss'], 0) for r in rows] == [-77.0, 54.0, 69.0, -55.0], rows
    assert all(r['ok'] is False for r in rows)
    assert all(motion.axes[f'p{c}'].position == 0 for c in range(1, 5))
    assert devices['basler_camera:1'].com          # the centre of mass was switched on


def test_run_reaches_every_marker_within_its_tolerance():
    aligner, motion, devices, events = make_world()
    aligner.command('set_setup', {'setup': setup_rows(tol=0.5)})
    aligner.command('start', {})
    assert motion.owner == 'aligner'
    try:
        motion.command('move_by', {'axis': 'p1', 'delta': 10, 'who': 'jog box'})
    except ValueError as e:
        assert 'in use by aligner' in str(e)
    else:
        raise AssertionError('the jog box moved an axis during a run')
    wait_idle(aligner)
    assert aligner.result['summary']['target_reached'], aligner.result
    miss = spots(motion) - np.array([*GOAL['basler_camera:1'], *GOAL['ximea_camera:2']])
    assert np.all(np.abs(miss) <= 0.5), miss
    assert motion.owner is None
    kinds = set()
    while not events.empty():
        kinds.add(events.get_nowait()['type'])
    assert {'status', 'evaluation', 'result'} <= kinds, kinds


def test_pause_resume_and_stop_returns_to_best():
    aligner, motion, devices, _ = make_world(slow_s=0.05)
    aligner.command('set_setup', {'setup': setup_rows(tol=0.01)})   # tighter than the noise
    aligner.command('start', {})
    while len(aligner.history) < 6:
        time.sleep(0.01)
    aligner.command('pause', {})
    time.sleep(0.3)
    n = len(aligner.history)
    time.sleep(0.3)
    assert len(aligner.history) <= n + 1 and aligner.state == 'paused'
    aligner.command('resume', {})
    while len(aligner.history) < n + 3:
        time.sleep(0.01)
    aligner.command('stop', {'on_abort': 'best'})
    wait_idle(aligner)
    s = aligner.result['summary']
    assert s['aborted'] and not s['target_reached'], s
    best = s['best']
    assert all(abs(motion.axes[a].commanded - best[a]) < 1e-9 for a in best), (best,)
    assert motion.owner is None


def test_estop_ends_the_run():
    aligner, motion, devices, _ = make_world(slow_s=0.05)
    aligner.command('set_setup', {'setup': setup_rows(tol=0.01)})
    aligner.command('start', {})
    while len(aligner.history) < 6:
        time.sleep(0.01)
    motion.command('estop', {'who': 'jog box'})       # always allowed
    wait_idle(aligner)
    assert aligner.result['error'] or aligner.result['summary'].get('aborted'), aligner.result
    assert motion.owner is None
    try:
        aligner.command('start', {})
    except ValueError as e:
        assert 'E-stopped' in str(e)
    else:
        raise AssertionError('started with E-stopped axes')
    assert motion.owner is None


def test_refusals():
    aligner, motion, devices, _ = make_world()
    s = setup_rows()
    s['levels'].append({'rows': [], 'controls': []})
    aligner.command('set_setup', {'setup': s})
    try:
        aligner.command('start', {})
    except ValueError as e:
        assert 'nested' in str(e)
    else:
        raise AssertionError('started a nested run')
    s = setup_rows()
    s['levels'][0]['controls'] = [{'axis': 'nope'}]
    aligner.command('set_setup', {'setup': s})
    try:
        aligner.command('start', {})
    except ValueError as e:
        assert 'not open' in str(e)
    else:
        raise AssertionError('started with a missing control')
    assert motion.owner is None


def test_catalog_follows_boxes_opened_later():
    aligner, motion, devices, events = make_world()
    del devices['motion_axes:simulated']               # the aligner opened first
    aligner.CATALOG_POLL_S = 0.05
    aligner.open()
    try:
        assert aligner.describe()['catalog']['controls'] == []
        devices['motion_axes:simulated'] = motion      # then the motion box
        deadline = time.time() + 2
        cat = None
        while time.time() < deadline and cat is None:
            try:
                e = events.get(timeout=0.1)
            except Exception:
                continue
            if e['type'] == 'catalog':
                cat = e['catalog']
        assert cat and [c['axis'] for c in cat['controls']] == ['p1', 'p2', 'p3', 'p4'], cat
        motion.axes['p1'].move_by(50, who='test')      # a position alone: no new catalog
        time.sleep(0.3)
        assert not any(events.get_nowait()['type'] == 'catalog' for _ in range(events.qsize()))
    finally:
        aligner.close()


def test_settings_persist():
    aligner, *_ = make_world()
    aligner.command('set_setup', {'setup': setup_rows()})
    snap = aligner.settings_snapshot()
    other = AlignerAdapter('main')
    other.restore_settings(snap)
    assert other.setup['levels'] == setup_rows()['levels']
    assert other.setup['fields']['on_abort'] == 'best'


def main():
    tests = [(k, v) for k, v in globals().items() if k.startswith('test_') and callable(v)]
    for name, fn in tests:
        fn()
        print(f'ok  {name}')
    print(f'{len(tests)} aligner tests passed')


if __name__ == '__main__':
    main()
