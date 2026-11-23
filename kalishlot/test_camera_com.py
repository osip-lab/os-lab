"""Standalone check of the camera centre of mass (adapters/camera_com.py).

Runs without the server, hardware or websockets (smoke_test.py calls run()
as check_camera_com on machines that have its dependencies):

    python kalishlot/test_camera_com.py
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from adapters.camera_com import center_of_mass  # noqa: E402
from adapters.dummy_camera import DummyCameraAdapter  # noqa: E402

rng = np.random.default_rng(1)


def beam(shape, x0, y0, sigma_x, sigma_y=None, amplitude=2000.0, background=100.0,
         noise=20.0, levels_max=4095):
    sigma_y = sigma_x if sigma_y is None else sigma_y
    yy, xx = np.indices(shape, dtype=np.float64)
    img = background + amplitude * np.exp(-(xx - x0) ** 2 / (2 * sigma_x ** 2)
                                          - (yy - y0) ** 2 / (2 * sigma_y ** 2))
    img += rng.normal(0, noise, shape)
    return np.clip(img, 0, levels_max).astype(np.uint16)


def check_accuracy():
    errors = []
    for _ in range(20):
        x0, y0 = rng.uniform(150, 850, 2)
        r = center_of_mass(beam((1000, 1000), x0, y0, 25, 40), levels_max=4095)
        assert r['valid'] and not r['reason'], r
        errors.append(max(abs(r['x'] - x0), abs(r['y'] - y0)))
    assert max(errors) < 0.2, errors
    # the thresholded moments of a Gaussian are narrower than sigma, by a fixed factor
    r = center_of_mass(beam((1000, 1000), 500, 500, 25, 40))
    assert 0.5 < r['sigma_x'] / 25 < 1.0 and abs(r['sigma_x'] / 25 - r['sigma_y'] / 40) < 0.02, r
    # small spot, weak beam: still sub-pixel
    r = center_of_mass(beam((600, 800), 300.3, 200.7, 4, amplitude=300, noise=15))
    assert r['valid'] and abs(r['x'] - 300.3) < 0.3 and abs(r['y'] - 200.7) < 0.3, r
    # cut 0: the noise floor and the window still keep the background out
    r0 = center_of_mass(beam((1000, 1000), 200, 200, 10, noise=5), cut_fraction=0.0)
    assert abs(r0['x'] - 200) < 0.1 and abs(r0['y'] - 200) < 0.1, r0
    # a hot pixel far from the spot is outside the window
    img = beam((1000, 1000), 300, 300, 10)
    img[900, 900] = 4095
    r = center_of_mass(img)
    assert abs(r['x'] - 300) < 0.1 and abs(r['y'] - 300) < 0.1, r
    print(f'accuracy ok (worst error {max(errors):.3f} px on 20 random beams; '
          f'cut 0 and a far hot pixel do not bias it)')


def check_flags():
    r = center_of_mass(beam((400, 400), 200, 200, 10, amplitude=0))
    assert not r['valid'] and r['reason'].startswith('no spot'), r
    r = center_of_mass(beam((400, 400), 200, 200, 10, amplitude=8000), levels_max=4095)
    assert r['valid'] and r['saturated'] and r['reason'] == 'saturated', r
    r = center_of_mass(beam((400, 400), 3, 200, 10))
    assert r['valid'] and r['clipped'], r
    hot = beam((400, 400), 0, 0, 1, amplitude=0)
    hot[50, 60] = 4000                                   # a hot pixel is not a spot
    r = center_of_mass(hot)
    assert not r['valid'], r
    print('flags ok (no spot, saturated, clipped, hot pixel)')


def check_speed():
    frame = beam((2048, 2048), 900, 1100, 30)
    center_of_mass(frame)
    tic = time.perf_counter()
    for _ in range(5):
        center_of_mass(frame)
    ms = (time.perf_counter() - tic) / 5 * 1e3
    small = beam((546, 720), 300, 250, 20)
    tic = time.perf_counter()
    for _ in range(20):
        center_of_mass(small)
    ms_small = (time.perf_counter() - tic) / 20 * 1e3
    assert ms < 200, ms
    print(f'speed: {ms:.1f} ms on 2048x2048, {ms_small:.1f} ms on 720x546')


def check_adapter():
    cam = DummyCameraAdapter('synthetic-0')
    cam.FRAME_RATE = 30.0
    cam.ORBIT_PERIOD_S = 1e12       # a still beam, at (768, 512)
    events = cam.add_listener()
    cam.open()
    try:
        assert cam.describe()['com'] is False and 'com_on' in cam.describe()['commands']
        try:
            cam.com_samples(1, timeout=0.2)
        except RuntimeError:
            pass
        else:
            raise AssertionError('samples while off')
        cam.command('com_on', {})
        samples = cam.com_samples(3, timeout=5)
        assert all(s['valid'] for s in samples), samples
        for s in samples:
            phase = 2 * np.pi * s['frame_time_s'] / cam.ORBIT_PERIOD_S
            ex, ey = 512 + 256 * np.cos(phase), 512 + 256 * np.sin(phase)
            assert np.hypot(s['sensor_x'] - ex, s['sensor_y'] - ey) < 0.5, (s, ex, ey)
        since = samples[-1]['frame_time_s']
        nxt = cam.com_samples(2, since=since, timeout=5)
        assert all(s['frame_time_s'] > since for s in nxt)

        # an ROI: frame px shift, sensor px stay put
        cam.command('set_roi', {'x': 100, 'y': 200, 'width': 800, 'height': 700})
        s = cam.com_samples(1, timeout=5)[0]
        assert abs(s['sensor_x'] - s['x'] - 100) < 1e-9 and abs(s['sensor_y'] - s['y'] - 200) < 1e-9, s
        cam.command('clear_roi', {})

        cam.command('set_com_cut', {'value': 0.5})
        assert cam.describe()['com_cut'] == 0.5
        assert cam.settings_snapshot()['com_cut'] == 0.5
        try:
            cam.command('set_com_cut', {'value': 1.5})
        except ValueError:
            pass
        else:
            raise AssertionError('accepted cut 1.5')
        time.sleep(0.3)
        kinds = set()
        while not events.empty():
            kinds.add(events.get_nowait().get('type'))
        assert {'com', 'com_status'} <= kinds, kinds
        cam.command('com_off', {})
        assert cam.describe()['com'] is False
    finally:
        cam.close()
    # the cut comes back with the other settings
    cam2 = DummyCameraAdapter('synthetic-1')
    cam2.restore_settings({'com_cut': 0.35})
    assert cam2.describe()['com_cut'] == 0.35
    print('camera adapter ok (com_on, fresh samples, ROI -> sensor px, cut, events, off)')


def run():
    check_accuracy()
    check_flags()
    check_speed()
    check_adapter()


if __name__ == '__main__':
    run()
