"""Hardware-free tests of the centre-of-mass measurement and its marker loop.

A fake kalishlot server holds two cameras whose spots move linearly (with
noise) with four simulated axes, like two mirrors walking a beam onto two
cameras.

    python alignment/test_com_measure.py
"""

import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alignment.com_measure import ComMeasure  # noqa: E402
from alignment.loop import AlignmentLoop  # noqa: E402
from alignment.run_com_marker_alignment import parse_targets  # noqa: E402
from motion import AxisLimits, build_axis  # noqa: E402

CAMS = {'basler': 'basler_camera:1', 'ximea': 'ximea_camera:2'}
MARKERS = {'basler_camera:1': [{'x': 1077.1, 'y': 705.8, 'label': 'Mirror center', 'id': 'a'}],
           'ximea_camera:2': [{'x': 1051.0, 'y': 844.6, 'label': 'Aperture', 'id': 'b'}]}
# px per step: rows basler x, y, ximea x, y; columns the four axes (two mirrors, x/y each)
J_TRUE = np.array([[0.050, 0.004, 0.030, -0.002],
                   [0.003, -0.045, 0.001, 0.028],
                   [0.020, 0.001, 0.060, 0.003],
                   [-0.002, 0.025, 0.004, 0.055]])
START = np.array([1000.0, 760.0, 1120.0, 790.0])     # spots at all-axes zero


class FakeKalishlot:
    def __init__(self, axes=None, noise_px=0.1, seed=0, invalid=None):
        self.axes = axes or {}
        self.noise_px = noise_px
        self.rng = random.Random(seed)
        self.invalid = invalid or {}           # device id -> reason of every result
        self.com = {d: False for d in MARKERS}
        self.calls = []

    def spots(self):
        p = np.array([self.axes[n].position for n in sorted(self.axes)]) if self.axes else np.zeros(4)
        return START + J_TRUE @ p

    def __call__(self, url, payload=None):
        self.calls.append((url, payload))
        if url.endswith('/api/devices'):
            return [{'device_id': d, 'com': self.com[d], 'markers': MARKERS[d]} for d in MARKERS]
        device_id = next(d for d in MARKERS if d.replace(':', '%3A') in url)
        if payload['name'] == 'com_on':
            self.com[device_id] = True
            return {'ok': True}
        assert payload['name'] == 'com_samples'
        s = self.spots()
        x0, y0 = (s[0], s[1]) if device_id.startswith('basler') else (s[2], s[3])
        out = []
        for _ in range(payload['args']['n']):
            if device_id in self.invalid:
                out.append({'valid': False, 'reason': self.invalid[device_id], 'sensor_x': None})
            else:
                out.append({'valid': True, 'clipped': False, 'counts': 1e5,
                            'sensor_x': x0 + self.rng.gauss(0, self.noise_px),
                            'sensor_y': y0 + self.rng.gauss(0, self.noise_px)})
        return {'samples': out}


def sim_axes():
    lim = AxisLimits(min=-10000, max=10000, max_step=200, max_slew=1e9, settle_s=0.0,
                     vibration_settle_s=0.0)
    cfg = {f'pico{c}': {'driver': 'simulated', 'address': f'sim:{c}',
                        'options': dict(units='steps', kind='position', start=0.0),
                        'limits': lim} for c in range(1, 5)}
    axes = {n: build_axis(n, cfg) for n in cfg}
    for a in axes.values():
        a.open()
    return axes


def test_measure_reduces_to_median_per_camera():
    fake = FakeKalishlot()
    m = ComMeasure(CAMS, samples=9, delay_s=0.0, request=fake)
    m.ensure_on()
    assert all(fake.com.values())
    cm = m()
    assert abs(cm.metrics['basler_com_x'].value - 1000.0) < 0.2
    assert abs(cm.metrics['ximea_com_y'].value - 790.0) < 0.2
    assert cm.metrics['basler_com_x'].n == 9 and cm.metrics['basler_com_x'].valid


def test_invalid_camera_makes_its_metrics_invalid():
    fake = FakeKalishlot(invalid={'ximea_camera:2': 'no spot'})
    cm = ComMeasure(CAMS, delay_s=0.0, request=fake)()
    assert cm.metrics['basler_com_x'].valid
    assert not cm.metrics['ximea_com_x'].valid and 'no spot' in cm.metrics['ximea_com_x'].reason


def test_marker_target_from_live_markers():
    m = ComMeasure(CAMS, request=FakeKalishlot())
    t = m.marker_target({'basler': 'Mirror center', 'ximea': 'Aperture'}, tolerance_px=2)
    assert t.metrics == ('basler_com_x', 'basler_com_y', 'ximea_com_x', 'ximea_com_y')
    assert t.goal == (1077.1, 705.8, 1051.0, 844.6) and t.tolerance == 2


def test_parse_targets():
    cams, marks = parse_targets(['basler_camera:24916690=Mirror center',
                                 'ximea_camera:QXNAA2506000=Aperture'])
    assert cams == {'basler': 'basler_camera:24916690', 'ximea': 'ximea_camera:QXNAA2506000'}
    assert marks == {'basler': 'Mirror center', 'ximea': 'Aperture'}


def test_loop_puts_both_spots_on_their_markers():
    axes = sim_axes()
    fake = FakeKalishlot(axes)
    m = ComMeasure(CAMS, samples=10, delay_s=0.0, request=fake)
    target = m.marker_target({'basler': 'Mirror center', 'ximea': 'Aperture'}, tolerance_px=1.5)
    start_miss = np.linalg.norm(fake.spots() - np.array(target.goal))
    summary = AlignmentLoop(axes, m, target, max_evaluations=60, print_fn=None).run()
    miss = np.linalg.norm(fake.spots() - np.array(target.goal))
    assert start_miss > 100 and summary['target_reached'] and miss < 1.5, (start_miss, miss, summary)
    assert summary['evaluations'] < 40, summary


def main():
    tests = [(k, v) for k, v in globals().items() if k.startswith('test_') and callable(v)]
    for name, fn in tests:
        fn()
        print(f'ok  {name}')
    print(f'{len(tests)} com measure tests passed')


if __name__ == '__main__':
    main()
