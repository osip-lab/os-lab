"""Hardware-free tests of the camera recorder and of LiveCapture.

    python alignment/test_live_capture.py        (from kalishlot/)

The recorder runs on the synthetic camera adapter. LiveCapture runs on a fake
scope and a fake camera that see the same synthetic cavity transmission, with
a known error in the scope's host clock, which the optical fit must find.
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adapters.camera_base import Recording  # noqa: E402
from adapters.dummy_camera import DummyCameraAdapter  # noqa: E402
from alignment.live_capture import LiveCapture, SyncError, shared_sync  # noqa: E402

RATE = 20_000.0
EXPOSURE_S = 0.008
PERIOD_S = 0.0088
HOST0 = 1.7e9               # host time.time() at scope time 0
CAMERA_CLOCK = 123.456      # camera timestamp (s) minus scope time
LATENCY_S = 0.0005          # frame reaches the host this long after its exposure ends


def test_recorder_on_dummy_camera():
    cam = DummyCameraAdapter('synthetic-test')
    cam.open()
    try:
        time.sleep(0.2)
        assert not cam.recording
        cam.start_recording(max_s=5)
        try:
            cam.start_recording(max_s=5)
            raise AssertionError('a second recording must refuse, not steal the first')
        except RuntimeError:
            pass
        time.sleep(0.5)
        rec = cam.stop_recording()
        assert len(rec.frames) >= 3 and len(rec.meta) == len(rec.frames)
        assert not rec.truncated and rec.nbytes == sum(f.nbytes for f in rec.frames)
        assert all('host_time_s' in m for m in rec.meta)
        assert cam.stop_recording() is None             # nothing recorded while disarmed

        cam.start_recording(max_s=5, max_bytes=2.5 * rec.frames[0].nbytes)
        time.sleep(0.5)
        capped = cam.stop_recording()
        assert len(capped.frames) == 2 and 'MB cap' in capped.truncated, capped.truncated

        cam.start_recording(max_s=0.1)
        time.sleep(0.5)
        timed = cam.stop_recording()
        assert 's cap' in timed.truncated, timed.truncated
    finally:
        cam.close()


def _transmission(n):
    """Lorentzian resonances at irregular times (no periodicity to alias on)."""
    rng = np.random.default_rng(1)
    t = np.arange(n) / RATE
    y = np.full(n, 0.01)
    for centre in np.cumsum(rng.uniform(0.05, 0.3, 40)):
        y += 0.6 / (1 + ((t - centre) / 0.001) ** 2)
    aux = 0.3 + 0.1 * np.sin(2 * np.pi * 0.05 * t)
    return t, y + rng.normal(0, 0.002, n), aux


class FakeScope:
    def __init__(self, clock_error_s):
        self.t, self.y, self.aux = _transmission(int(6 * RATE))
        self.sample_rate_hz = RATE
        self.stream_generation = 1
        self.channels = {'D': {'range_v': 5.0}, 'B': {'range_v': 2.0}}
        self.clock_error_s = clock_error_s

    def sample_at_host_time(self, host):
        return (host - HOST0 - self.clock_error_s) * RATE

    def samples_written(self, name):
        return self.t.size

    def read_span(self, name, start, stop):
        if start < 0 or stop > self.t.size:
            return None
        return (self.y if name == 'D' else self.aux)[start:stop].copy()

    def to_volts(self, name, data):
        return np.asarray(data, dtype=float)


class FakeCamera:
    """Frames whose brightness is the transmission integrated over each exposure."""

    ROI = {'x': 0, 'y': 0, 'width': 16, 'height': 16}

    def __init__(self, scope, first_frame_s, n=200):
        self.scope, self.first, self.n = scope, first_frame_s, n
        self.exposure_us = EXPOSURE_S * 1e6

    def start_recording(self, max_s, max_bytes):
        pass

    def recording_geometry(self):
        return {'roi': self.ROI, 'exposure_us': self.exposure_us}

    def stop_recording(self):
        s = self.scope
        yy, xx = np.mgrid[:16, :16]
        spot = np.exp(-((xx - 8) ** 2 + (yy - 8) ** 2) / 8.0)
        rec = Recording(roi=dict(self.ROI))
        for i in range(self.n):
            start = self.first + i * PERIOD_S
            sel = (s.t >= start) & (s.t < start + EXPOSURE_S)
            rec.frames.append((100 * s.y[sel].mean() * spot + 2.0).astype(np.uint16))
            rec.meta.append({'block_id': i + 1,
                             'camera_timestamp_ns': int((start + CAMERA_CLOCK) * 1e9),
                             'host_time_s': HOST0 + start + EXPOSURE_S + LATENCY_S,
                             'exposure_us': self.exposure_us})
        return rec


def test_live_capture_finds_the_clock_error():
    scope = FakeScope(clock_error_s=0.004)
    live = LiveCapture(scope, FakeCamera(scope, first_frame_s=2.0))
    first = live.capture(0.01)
    assert first.sync['mode'] == 'fit', first.sync
    # the frames must land where they really were, whatever the host clocks said
    assert abs(first.windows[0, 0] - 2.0) < 2e-4, first.windows[0]
    expected = scope.clock_error_s - LATENCY_S
    assert abs(first.sync['correction_s'] - expected) < 2e-4, first.sync
    assert first.frames.shape == (200, 16, 16) and first.t[0] < 2.0 < first.t[-1]

    second = live.capture(0.01)                          # tracked: narrow search
    assert second.sync['searched_s'] < first.sync['searched_s']
    assert abs(second.windows[0, 0] - 2.0) < 2e-4

    scope.stream_generation += 1                         # restart: correction dropped
    third = live.capture(0.01)
    assert third.sync['searched_s'] == first.sync['searched_s']


def test_establish_then_kept_without_blinks():
    scope = FakeScope(clock_error_s=0.004)
    live = LiveCapture(scope, FakeCamera(scope, first_frame_s=2.0))
    try:
        live.capture(0.01, placement='kept')
        raise AssertionError('a kept placement with no sync must refuse')
    except SyncError:
        pass
    live.capture(0.01, placement='establish')
    assert live.sync.valid and live.sync.method == 'sync'
    expected = scope.clock_error_s - LATENCY_S
    assert abs(live.sync.correction_s - expected) < 2e-4, live.sync.to_dict()

    blinks = scope.y
    scope.y = np.full_like(blinks, 0.01)                 # laser off: nothing to fit
    dark = live.capture(0.01, placement='kept')          # nothing to fit: placed by the kept sync
    assert dark.sync['mode'] == 'kept correction' and not dark.sync['check']['locked']
    assert abs(dark.windows[0, 0] - 2.0) < 2e-4, dark.windows[0]
    try:
        live.capture(0.01, placement='establish')
        raise AssertionError('a sync with no blinks must refuse')
    except SyncError:
        pass
    assert live.sync.valid                              # a failed sync keeps the old one

    scope.y = blinks
    lit = live.capture(0.01, placement='kept')
    assert lit.sync['check']['locked'] and abs(lit.sync['check']['residual_s']) < 2e-4
    assert not lit.sync['check']['resync_advised']

    live.camera.exposure_us = 5000.0                     # exposure change: sync dropped, and why
    try:
        live.capture(0.01, placement='kept')
        raise AssertionError('a changed exposure must clear the sync')
    except SyncError as error:
        assert 'exposure' in str(error), error


def test_shared_sync_is_dropped_on_reopen():
    scope, camera = FakeScope(0.0), object()
    a = shared_sync('pico', scope, 'cam', camera)
    assert shared_sync('pico', scope, 'cam', camera) is a
    a.set(0.0015, {'scope_stream': 1}, 30.0, 'sync')
    b = shared_sync('pico', scope, 'cam', object())      # the camera box closed and reopened
    assert b is not a and not b.valid and 'camera box was reopened' in b.cleared


class FakeScopeAdapter:
    type_name = 'picoscope'

    def __init__(self, scope):
        self.scope = scope
        scope.is_streaming, scope.serial, scope.variant = True, 'TEST01', '4424A'
        for config in scope.channels.values():
            config.update(enabled=True, coupling='DC')


class FakeXimeaAdapter(FakeCamera):
    """What the synced pipeline box asks of a XIMEA camera adapter."""
    type_name = 'ximea_camera'

    def describe(self):
        return {'settings': [{'name': 'framerate', 'value': 1 / PERIOD_S}]}

    def _frame_shape(self):
        return [16, 16]

    def hardware_info(self):
        return {'make': 'ximea', 'serial_number': 'TEST', 'pixel_size_mm': 0.0055,
                'pixel_format': 'Mono10', 'bit_depth': 10, 'width': 16, 'height': 16,
                'offset_x': 0, 'offset_y': 0, 'binning_x': 1, 'binning_y': 1,
                'exposure_us': self.exposure_us, 'gain_db': 0.0,
                'frame_rate_hz': 1 / PERIOD_S, 'resulting_frame_rate_hz': 1 / PERIOD_S}


def test_quick_capture_through_the_box():
    """Sync, then a quick capture saved as a session the usual readers take."""
    import tempfile
    from adapters import synced_pipeline as sp
    from pico_scope.mode_video_sync import load_session, release_frames, session_windows

    scope = FakeScope(clock_error_s=0.004)
    camera = FakeXimeaAdapter(scope, first_frame_s=2.0)
    devices = {'picoscope:TEST01': FakeScopeAdapter(scope), 'ximea_camera:TEST': camera}
    saved_registry, saved_sync_s = sp.SyncedPipelineAdapter.registry, sp.SYNC_DURATION_S
    sp.SyncedPipelineAdapter.registry = staticmethod(lambda: (devices, {}))
    sp.SYNC_DURATION_S = 0.01
    try:
        with tempfile.TemporaryDirectory() as tmp:
            box = sp.SyncedPipelineAdapter('main')
            box.edited.update({sp.FOLDER_PARAM: tmp, 'capture.CAPTURE_DURATION_S': 0.01,
                               'capture.FRAMES_FORMAT': 'lossless',
                               'capture.SCOPE_CHANNEL': 'D', 'capture.SCOPE_AUX_CHANNEL': 'B'})
            status = box.describe()['live']
            assert status['ready'] and not status['sync']['valid'], status
            try:
                box.command('live_capture', {})
                raise AssertionError('start capture must wait for a sync')
            except ValueError as error:
                assert 'sync' in str(error)

            def wait():
                deadline = time.time() + 30
                while box.live['busy'] and time.time() < deadline:
                    time.sleep(0.02)
                assert not box.live['busy'] and not box.live['error'], box.live

            box.command('live_sync', {})
            wait()
            assert box.describe()['live']['sync']['valid']
            box.command('live_capture', {'folder_label': 'quick', 'long_arm_cm': '12.5'})
            wait()
            session_path = Path(box.live['session'])
            assert session_path.parent.name.endswith('_quick'), session_path

            session, frames = load_session(session_path)
            assert frames.shape == (200, 16, 16) and session['long_arm_m'] == 0.125
            assert session['sync']['method'] == 'live_stream'
            windows, key = session_windows(session_path)
            origin = session['scope']['stream_t_origin_s']
            assert key == 't0_fitted_s' and abs(windows[0, 0] + origin - 2.0) < 2e-4, windows[0]
            with np.load(session_path.parent / session['scope']['file']) as trace:
                assert trace['t'][0] == 0 and 'aux' in trace.files
            release_frames(frames)

            box.command('live_sync_reset', {})
            assert not box.describe()['live']['sync']['valid']
    finally:
        sp.SyncedPipelineAdapter.registry = saved_registry
        sp.SYNC_DURATION_S = saved_sync_s


def test_scope_only():
    scope = FakeScope(clock_error_s=0.0)
    scope.sample_at_host_time = lambda host: 1.0 * RATE  # "now" is scope time 1 s
    d = LiveCapture(scope).capture(0.5, with_frames=False)
    assert d.frames is None and abs(d.t[0] - 1.0) < 1e-9 and abs(d.t[-1] - 1.5) < 1e-3


def main():
    tests = [(k, v) for k, v in globals().items() if k.startswith('test_') and callable(v)]
    for name, fn in tests:
        fn()
        print(f'ok  {name}')
    print(f'{len(tests)} live-capture tests passed')


if __name__ == '__main__':
    main()
