"""End-to-end smoke test of the web GUI server using the dummy camera.

Needs no hardware and no browser. Start the server is NOT required — this
script launches its own instance on a test port, runs the checks, and shuts
it down.

    python smoke_test.py
"""

import asyncio
import os
from pathlib import Path
import json
import threading
import time
import urllib.error
import urllib.request

import websockets

HOST = '127.0.0.1'
PORT = 8765
BASE = f'http://{HOST}:{PORT}'


def api(path, method='GET', body=None):
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(f'{BASE}{path}', data=data, method=method,
                                     headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request) as response:
        return json.loads(response.read())


def start_server():
    import tempfile
    import uvicorn
    import server
    # a private layout: the real one must neither be restored (it would open
    # the lab's cameras) nor overwritten by this run
    server.LAYOUT_PATH = Path(tempfile.mkdtemp()) / 'layout.json'
    server.layout.update({'devices': [], 'boxes': {}})
    from server import app
    config = uvicorn.Config(app, host=HOST, port=PORT, log_level='warning')
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(100):
        if server.started:
            return server
        time.sleep(0.1)
    raise RuntimeError('server did not start')


async def wait_event(socket, event_type, timeout=30):
    while True:
        message = await asyncio.wait_for(socket.recv(), timeout=timeout)
        if isinstance(message, str):
            event = json.loads(message)
            if event.get('type') == event_type:
                return event


async def check_stream(device_id):
    uri = f'ws://{HOST}:{PORT}/ws/devices/{device_id}'
    async with websockets.connect(uri) as socket:
        # collect a few frames; verify they are JPEG
        frames = 0
        while frames < 3:
            message = await asyncio.wait_for(socket.recv(), timeout=5)
            if isinstance(message, bytes):
                assert message[:2] == b'\xff\xd8', 'not a JPEG frame'
                frames += 1
        print(f'received {frames} JPEG frames ok')

        # pause via REST, expect a status event on the socket
        api(f'/api/devices/{device_id}/command', 'POST',
            {'name': 'pause', 'args': {}})
        while True:
            message = await asyncio.wait_for(socket.recv(), timeout=5)
            if isinstance(message, str):
                event = json.loads(message)
                if event.get('type') == 'status':
                    assert event['playing'] is False
                    print('pause status event ok')
                    break

        # change a setting, expect setting_applied event
        api(f'/api/devices/{device_id}/command', 'POST',
            {'name': 'set_setting', 'args': {'name': 'exposure', 'value': 5000}})
        while True:
            message = await asyncio.wait_for(socket.recv(), timeout=5)
            if isinstance(message, str):
                event = json.loads(message)
                if event.get('type') == 'setting_applied':
                    assert event['name'] == 'exposure' and event['value'] == 5000
                    print('setting_applied event ok')
                    break

        # snap while paused should deliver exactly one new frame
        api(f'/api/devices/{device_id}/command', 'POST',
            {'name': 'snap', 'args': {}})
        while True:
            message = await asyncio.wait_for(socket.recv(), timeout=5)
            if isinstance(message, bytes):
                print('single frame while paused ok')
                break

        # enable the Gaussian fit (still paused: fit_on refits the newest
        # frame right away); the dummy beam is a real Gaussian, sigma 60 px
        api(f'/api/devices/{device_id}/command', 'POST',
            {'name': 'fit_on', 'args': {}})
        event = await wait_event(socket, 'fit_status')
        assert event['enabled'] is True
        event = await wait_event(socket, 'fit')
        assert event['success'], event
        assert abs(event['params']['s_x'] - 60) < 6, event['params']
        assert len(event['cross']['row']) > 100
        print(f"fit event ok: s_x = {event['params']['s_x']:.1f} px "
              f"(expected 60)")

        # guess circle: broadcast to all viewers, then a refit with it
        api(f'/api/devices/{device_id}/command', 'POST',
            {'name': 'set_guess',
             'args': {'x_0': 500, 'y_0': 500, 'sigma': 80}})
        event = await wait_event(socket, 'guess')
        assert event['guess']['sigma'] == 80
        event = await wait_event(socket, 'fit')
        assert event['success'], event
        print('guess + refit ok')

        # blink trigger: with an absurdly high threshold no frame qualifies —
        # brightness readouts keep flowing but fit results stop
        api(f'/api/devices/{device_id}/command', 'POST',
            {'name': 'set_fit_threshold', 'args': {'value': 1e6}})
        event = await wait_event(socket, 'fit_threshold')
        assert event['value'] == 1e6
        api(f'/api/devices/{device_id}/command', 'POST',
            {'name': 'play', 'args': {}})  # resume frames (we paused earlier)
        await wait_event(socket, 'brightness')
        deadline = time.time() + 2
        while time.time() < deadline:
            message = await asyncio.wait_for(socket.recv(), timeout=5)
            if isinstance(message, str):
                event = json.loads(message)
                assert event.get('type') != 'fit', \
                    'fit ran on a frame below the trigger threshold'
        print('trigger blocks dim frames ok (brightness events still flowing)')

        api(f'/api/devices/{device_id}/command', 'POST',
            {'name': 'set_fit_threshold', 'args': {'value': 0}})
        event = await wait_event(socket, 'fit_threshold')
        assert event['value'] == 0
        event = await wait_event(socket, 'fit')
        assert event['success'], event
        print('trigger off -> fit resumes ok')

        api(f'/api/devices/{device_id}/command', 'POST',
            {'name': 'fit_off', 'args': {}})
        event = await wait_event(socket, 'fit_status')
        assert event['enabled'] is False
        print('fit_off ok')


async def wait_levels(socket, accept, what, timeout=20):
    """Read 'levels' events until one satisfies `accept(points)`.

    The socket may still hold events built before whatever just changed, so
    a test that looks at only the next one reads the past.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        event = await wait_event(socket, 'levels')
        if event['points'] and accept(event['points']):
            return event
    raise AssertionError(f'timed out waiting for {what}')


async def check_levels(device_id):
    """The intensity monitor: median / p99 / max over a rolling window."""
    uri = f'ws://{HOST}:{PORT}/ws/devices/{device_id}'
    async with websockets.connect(uri) as socket:
        def command(name, args=None):
            api(f'/api/devices/{device_id}/command', 'POST',
                {'name': name, 'args': args or {}})

        command('levels_on')
        event = await wait_event(socket, 'levels_status')
        assert event['enabled'] is True

        event = await wait_event(socket, 'levels')
        assert event['window_s'] == 30.0, event
        assert event['points'], 'no points in the first levels event'
        age, median, p99, maximum = event['points'][-1]
        # the synthetic beam sits on a background of 100 with sigma-30 noise,
        # and peaks at 100 + 1500 * exposure/3000 (exposure is 5000 by now)
        assert -1.0 <= age <= 0.0, f'newest point is {age} s old'
        assert 50 < median < 150, f'median {median} is not the background'
        assert p99 > median, f'p99 {p99} must be above the median {median}'
        assert maximum >= p99, f'max {maximum} must be at least p99 {p99}'
        assert maximum > 1000, f'max {maximum} misses the beam'
        print(f'levels ok: median={median}, p99={p99}, max={maximum}')

        # the window really accumulates. Events already in flight still carry
        # the shorter windows they were built with, so read on until one is
        # more than a second deep rather than trusting the next one.
        event = await wait_levels(socket, lambda points: points[0][0] <= -1.0,
                                  'the window to fill past 1 s')
        assert all(-30.0 <= p[0] <= 0.0 for p in event['points']), \
            'a point outside the 30 s window'
        assert len(event['points']) >= 8, \
            f'only {len(event["points"])} points in a second, expected ~10'
        print(f'window ok: {len(event["points"])} points, oldest '
              f'{event["points"][0][0]:.1f} s')

        # describe() carries the window, so a re-attaching viewer draws at once
        described = next(d for d in api('/api/devices')
                         if d['device_id'] == device_id)
        assert described['levels'] is True, described['levels']
        assert len(described['levels_points']) > 1, described['levels_points']

        # cropping changes which pixels are measured: the trace starts over
        command('set_roi', {'x': 0, 'y': 0, 'width': 256, 'height': 256})
        await wait_event(socket, 'roi')
        await wait_levels(socket, lambda points: points[0][0] > -1.0,
                          'the ROI change to clear the history')
        print('roi clears the history ok')
        command('clear_roi')
        await wait_event(socket, 'roi')

        command('levels_off')
        event = await wait_event(socket, 'levels_status')
        assert event['enabled'] is False
        described = next(d for d in api('/api/devices')
                         if d['device_id'] == device_id)
        assert 'levels_points' not in described, 'history kept after levels_off'
        print('levels_off ok (history dropped)')


def check_levels_window():
    """A point older than the window is never reported.

    The ring is trimmed when a frame arrives, so on a paused camera nothing
    trims it — and its points must still age off the chart, whose x axis
    means "seconds ago".
    """
    from adapters.camera_levels import CameraLevelsMixin

    class BareMonitor(CameraLevelsMixin):
        def emit(self, event):
            pass

    monitor = BareMonitor()
    monitor._init_levels()
    now = time.monotonic()
    monitor._levels = [(now - 45.0, 1.0, 2.0, 3), (now - 10.0, 4.0, 5.0, 6)]
    points = monitor.levels_event()['points']
    assert len(points) == 1, f'expected the 45 s-old point dropped, got {points}'
    assert points[0][1] == 4.0, points
    print('levels window ok (nothing older than 30 s is ever reported)')


def check_streamer_roi():
    """The hardware-ROI path (StreamerROIMixin), on a fake camera.

    Needs no server and no hardware, but covers what only runs with a real
    Basler/XIMEA attached: the ROI must be applied with acquisition STOPPED
    (both SDKs lock the geometry nodes while grabbing), the frames that
    follow must have the new size, and _after_roi_applied must get a chance
    to re-read the limits that moved with it.
    """
    import numpy as np

    from adapters.camera_base import CameraAdapterBase, StreamerROIMixin
    from camera_core import CameraStreamer

    class FakeCamera:
        """The camera contract CameraStreamer and StreamerROIMixin expect."""
        serial_number = 'fake-0'
        GRAB_TIMEOUT_MS = 1000
        SENSOR = (512, 640)   # height, width — deliberately not square

        def __init__(self):
            self.is_open = True
            self.streaming = False
            self.shape = self.SENSOR
            self.roi_while_streaming = None   # must stay None

        def start_streaming(self):
            self.streaming = True

        def stop_streaming(self):
            self.streaming = False

        def get_frame(self):
            time.sleep(0.01)
            return np.zeros(self.shape, dtype=np.uint16)

        def set_roi(self, width, height, offset_x=None, offset_y=None):
            if self.streaming:
                self.roi_while_streaming = (width, height)
            # snap like real hardware does: width to 4, height to 2
            width, height = width - width % 4, height - height % 2
            self.shape = (height, width)
            return {'width': width, 'height': height,
                    'offset_x': offset_x, 'offset_y': offset_y}

        def set_roi_full(self):
            height, width = self.SENSOR
            return self.set_roi(width, height, 0, 0)

    class FakeAdapter(StreamerROIMixin, CameraAdapterBase):
        type_name, display_name = 'fake_camera', 'fake camera'
        DISPLAY_DOWNSAMPLE = 2

        def __init__(self):
            super().__init__('fake-0')
            self.camera = FakeCamera()
            self.frames = []
            self.after_roi_calls = 0
            self.streamer = CameraStreamer(self.camera, on_frame=self._on_frame)

        def _on_frame(self, frame):
            self.frames.append(frame.shape)
            self._store_camera_frame(frame, frame[::2, ::2].astype(np.uint8))

        def _open(self):
            self.streamer.start()

        def _close(self):
            self.streamer.stop()

        def _sensor_shape(self):
            return list(FakeCamera.SENSOR)

        def _settings_schema(self):
            return []

        def _after_roi_applied(self, camera):
            self.after_roi_calls += 1

    def wait_for(predicate, what, timeout=5):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if predicate():
                return
            time.sleep(0.02)
        raise AssertionError(f'timed out waiting for {what}')

    adapter = FakeAdapter()
    adapter.open()
    try:
        wait_for(lambda: adapter.frames, 'the first frame')
        assert adapter.frames[-1] == (512, 640), adapter.frames[-1]

        adapter.command('set_roi', {'x': 100, 'y': 50, 'width': 202, 'height': 99})
        wait_for(lambda: adapter._roi is not None, 'the ROI to be applied')
        # snapped by the camera, not by us, and reported as snapped
        assert adapter._roi == {'x': 100, 'y': 50, 'width': 200, 'height': 98}, \
            adapter._roi
        assert adapter.camera.roi_while_streaming is None, \
            'the ROI was set while the camera was grabbing'
        assert adapter.after_roi_calls == 1
        assert adapter.camera.streaming, 'acquisition was not restarted'
        adapter.frames.clear()
        wait_for(lambda: adapter.frames, 'a frame after the ROI')
        assert adapter.frames[-1] == (98, 200), adapter.frames[-1]
        assert adapter.describe()['sensor_shape'] == [98, 200]

        adapter.command('clear_roi', {})
        wait_for(lambda: adapter._roi is None, 'the ROI to be cleared')
        adapter.frames.clear()
        wait_for(lambda: adapter.frames, 'a frame after clearing')
        assert adapter.frames[-1] == (512, 640), adapter.frames[-1]
        assert adapter.camera.roi_while_streaming is None
        print('hardware-ROI path ok (applied with acquisition stopped, '
              'snapped size reported, frames resized)')
    finally:
        adapter.close()


def jpeg_size(data):
    """(width, height) from a JPEG's frame header — what the viewer sees."""
    index = 2
    while index < len(data):
        assert data[index] == 0xFF, 'not a JPEG segment'
        marker = data[index + 1]
        length = int.from_bytes(data[index + 2:index + 4], 'big')
        if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
            height = int.from_bytes(data[index + 5:index + 7], 'big')
            width = int.from_bytes(data[index + 7:index + 9], 'big')
            return width, height
        index += 2 + length
    raise AssertionError('no JPEG frame header')


async def wait_frame_size(socket, expected, timeout=10):
    """Wait for a streamed frame of this size (frames already in flight when
    the ROI changed still carry the old one)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        message = await asyncio.wait_for(socket.recv(), timeout=timeout)
        if isinstance(message, bytes) and jpeg_size(message) == expected:
            return
    raise AssertionError(f'no {expected[0]}x{expected[1]} frame arrived')


async def check_roi(device_id):
    """ROI: crop, crop again relative to the crop, and uncrop.

    The dummy camera has no ROI feature, so this also exercises the base
    class's software crop — the path every camera without one takes.
    """
    uri = f'ws://{HOST}:{PORT}/ws/devices/{device_id}'
    async with websockets.connect(uri) as socket:
        def command(name, args=None):
            api(f'/api/devices/{device_id}/command', 'POST',
                {'name': name, 'args': args or {}})

        command('set_guess', {'x_0': 600, 'y_0': 600, 'sigma': 40})
        await wait_event(socket, 'guess')

        command('set_roi', {'x': 256, 'y': 256, 'width': 512, 'height': 512})
        event = await wait_event(socket, 'guess')
        assert event['guess']['x_0'] == 344 and event['guess']['y_0'] == 344, \
            f'guess must move with the crop, got {event["guess"]}'
        event = await wait_event(socket, 'roi')
        assert event['roi'] == {'x': 256, 'y': 256, 'width': 512, 'height': 512}
        assert event['sensor_shape'] == [512, 512], event
        assert event['sensor_full'] == [1024, 1024], event
        await wait_frame_size(socket, (512, 512))
        print('roi crop ok (guess carried along, 512x512 frames streaming)')

        described = next(d for d in api('/api/devices')
                         if d['device_id'] == device_id)
        assert described['sensor_shape'] == [512, 512], described
        assert described['roi']['x'] == 256, described
        print('describe() reports the ROI for re-attaching viewers ok')

        # a second ROI is relative to the current frame, so it zooms further
        # in; the guess is now outside the visible area and must be dropped
        command('set_roi', {'x': 0, 'y': 0, 'width': 128, 'height': 128})
        event = await wait_event(socket, 'guess')
        assert event['guess'] is None, event
        event = await wait_event(socket, 'roi')
        assert event['roi'] == {'x': 256, 'y': 256, 'width': 128, 'height': 128}
        await wait_frame_size(socket, (128, 128))
        print('second roi is relative to the first ok (guess dropped)')

        command('clear_roi')
        event = await wait_event(socket, 'roi')
        assert event['roi'] is None and event['sensor_shape'] == [1024, 1024]
        await wait_frame_size(socket, (1024, 1024))
        print('clear_roi restores the full sensor ok')

        # leave one behind for the persistence check after the re-open
        command('set_roi', {'x': 100, 'y': 200, 'width': 300, 'height': 400})
        await wait_event(socket, 'roi')


async def check_close_notification(device_id):
    uri = f'ws://{HOST}:{PORT}/ws/devices/{device_id}'
    async with websockets.connect(uri) as socket:
        api(f'/api/devices/{device_id}', 'DELETE')
        try:
            while True:
                await asyncio.wait_for(socket.recv(), timeout=5)
        except websockets.exceptions.ConnectionClosed as closed:
            code = closed.rcvd.code if closed.rcvd else None
            assert code == 4004, f'expected close code 4004, got {code}'
    # connecting to a device that does not exist must also yield a clean
    # 4004 (accept-then-close), not a bare handshake rejection
    async with websockets.connect(uri) as socket:
        try:
            await asyncio.wait_for(socket.recv(), timeout=5)
            raise AssertionError('expected immediate close for missing device')
        except websockets.exceptions.ConnectionClosed as closed:
            code = closed.rcvd.code if closed.rcvd else None
            assert code == 4004, f'expected close code 4004, got {code}'


def check_exposure_rate():
    """Exposure and frame rate following each other (XIMEA box), against a
    simulated camera: the rate it can keep is capped by readout, and by the
    exposure plus a fixed overhead - which the helpers must find by asking,
    not know."""
    from adapters.exposure_rate import exposure_for_rate, rate_for_exposure

    class Camera:
        OVERHEAD_US, READOUT_CAP_HZ = 50.0, 500.0

        def __init__(self):
            self._exposure, self._rate = 50000.0, 10.0

        @property
        def exposure_us(self):
            return self._exposure

        @exposure_us.setter
        def exposure_us(self, value):
            self._exposure = min(max(value, 10.0), 1e6)

        @property
        def frame_rate_limits_hz(self):
            return 1.0, min(self.READOUT_CAP_HZ,
                            1e6 / (self._exposure + self.OVERHEAD_US))

        @property
        def frame_rate_hz(self):
            return min(self._rate, self.frame_rate_limits_hz[1])

        @frame_rate_hz.setter
        def frame_rate_hz(self, value):
            self._rate = min(max(value, 1.0), self.frame_rate_limits_hz[1])

    camera = Camera()
    exposure_for_rate(camera, 100)          # from a 50 ms exposure
    assert abs(camera.frame_rate_hz - 100) < 1e-6, camera.frame_rate_hz
    assert 9940 <= camera.exposure_us <= 9950, camera.exposure_us
    rate_for_exposure(camera, 20000)
    assert camera.exposure_us == 20000
    assert abs(camera.frame_rate_hz - 1e6 / 20050) < 1e-6, camera.frame_rate_hz
    exposure_for_rate(camera, 2000)         # beyond the readout cap
    assert abs(camera.frame_rate_hz - 500) < 1e-6, camera.frame_rate_hz
    assert 1940 <= camera.exposure_us <= 1950, camera.exposure_us
    print('exposure and frame rate follow each other ok '
          '(100 Hz -> 9.95 ms, 20 ms -> 49.9 Hz, 2 kHz -> capped at 500)')


def check_picoscope_rigid_envelope():
    """A rolling window decimated at two different moments gives the same
    min/max for the samples both show, so the trace scrolls rigidly. The old
    newest-aligned buckets were re-cut on every refresh and jittered. No
    hardware: the decimation is a pure function."""
    import numpy as np
    from adapters.picoscope import MAX_POINTS, envelope, rigid_envelope

    rng = np.random.default_rng(1)
    n_window = 10_000
    signal = rng.integers(-2000, 2000, 30_000).astype(np.int16)  # noisy

    def view(end):
        return rigid_envelope(signal[end - n_window:end], end, MAX_POINTS,
                              n_window)

    def by_position(end):
        out, first, step = view(end)
        return {round(first + i * step, 3): v for i, v in enumerate(out)}

    a, b = by_position(20_000), by_position(20_000 + 37)   # 37 samples later
    shared = a.keys() & b.keys()
    assert len(shared) > 0.9 * len(a), (len(shared), len(a))
    assert all(a[k] == b[k] for k in shared), 'shared points moved'
    # whereas the newest-aligned envelope changes under the same shift
    old_a = envelope(signal[20_000 - n_window:20_000], MAX_POINTS)[0]
    old_b = envelope(signal[20_037 - n_window:20_037], MAX_POINTS)[0]
    assert not np.array_equal(old_a[:-74], old_b[74:][:len(old_a) - 74]),         'expected the old envelope to jitter'
    # a bucket still holds the window's extremes, and nothing is invented
    out, first, step = view(20_000)
    assert len(out) <= MAX_POINTS and step * 2 == -(-n_window // 500)
    assert out.min() >= signal.min() and out.max() <= signal.max()
    # a short window shows every sample; a part-filled one keeps its bucket
    short = signal[:500]
    assert rigid_envelope(short, 500, MAX_POINTS, 500)[0] is short
    part = rigid_envelope(signal[:3000], 3000, MAX_POINTS, n_window)
    assert part[2] == step
    print('picoscope envelope scrolls rigidly ok '
          f'({len(shared)} of {len(a)} points unchanged after a shift)')


def check_synced_pipeline():
    """The synced-pipeline box's adapter: every offered parameter is a real
    one, values are validated and persist, presets leave the folder alone, the
    gate counts lent devices, and a run's parameters say what they must. No
    hardware and no server: a registry of stand-in devices is supplied."""
    import tempfile
    from adapters import synced_pipeline as sp
    from pico_scope import run_config

    for param in sp.PARAMS:
        section, name = param['key'].split('.')
        assert name in run_config.SECTIONS[section][1], param['key']

    class Stub:
        def __init__(self, type_name):
            self.type_name = type_name

    registry = {'devices': {}, 'loans': {}}
    sp.SyncedPipelineAdapter.registry = staticmethod(
        lambda: (registry['devices'], registry['loans']))
    adapter = sp.SyncedPipelineAdapter('main')
    assert adapter.describe()['ready'] is False
    registry['devices']['ximea_camera:SN1'] = Stub('ximea_camera')
    assert adapter.describe()['ready'] is False, 'needs the scope too'
    registry['devices']['picoscope:S'] = Stub('picoscope')
    assert adapter.describe()['ready'] is True
    # a lent device is closed but still counts, or the box would grey out
    # in the middle of its own run
    registry['loans']['picoscope:S'] = {'type': 'picoscope'}
    del registry['devices']['picoscope:S']
    assert adapter.describe()['ready'] is True
    assert adapter.describe()['dependencies']['scopes'][0]['state'] == 'lent'

    with tempfile.TemporaryDirectory() as tmp:
        adapter.command('set_param', {'key': 'capture.OUTPUT_ROOT', 'value': tmp})
        adapter.command('set_param', {'key': 'capture.CAPTURE_DURATION_S',
                                      'value': '2.5'})
        for bad in ({'key': 'capture.CAPTURE_DURATION_S', 'value': 'x'},
                    {'key': 'capture.BINNING', 'value': 3},
                    {'key': 'capture.NOPE', 'value': 1},
                    {'key': 'capture.MASK_THRESHOLD', 'value': 5}):
            try:
                adapter.command('set_param', bad)
            except ValueError:
                pass
            else:
                raise AssertionError(f'{bad} should have been refused')
        adapter.command('set_adopt', {'key': 'exposure', 'value': False})
        adapter.command('set_adopt', {'key': 'roi', 'value': False})

        params = adapter.run_params()
        capture = params['sections']['capture']
        assert capture['CAPTURE_DURATION_S'] == 2.5
        assert capture['OUTPUT_ROOT'] == str(Path(tmp)), capture['OUTPUT_ROOT']
        assert capture['PROMPT_FOR_OUTPUT_ROOT'] is False
        assert capture['CAMERA'] == 'ximea' and capture['SERIAL_NUMBER'] == 'SN1'
        assert capture['EXPOSURE_US'] is None and capture['MANUAL_ROI'] is None
        assert 'SCOPE_RANGE_V' not in capture, 'a ticked row is taken from the box'
        assert params['adopt']['exposure'] is False and params['adopt']['gain']
        assert 'sync' not in params['sections'], 'only what was edited is sent'

        # persistence, with a stale entry that must be dropped
        snapshot = adapter.settings_snapshot()
        snapshot['edited']['capture.REMOVED_LONG_AGO'] = 1
        restored = sp.SyncedPipelineAdapter('main')
        restored.restore_settings(snapshot)
        assert restored.edited == adapter.edited and restored.adopt == adapter.adopt
        assert restored.long_arm_cm is None and adapter.describe()['long_arm_cm'] is None

        # the long arm: optional, saved with the settings, never sent to a run
        adapter.command('set_long_arm', {'value': '34.4'})
        assert adapter.describe()['long_arm_cm'] == 34.4
        restored.restore_settings(adapter.settings_snapshot())
        assert restored.long_arm_cm == 34.4
        assert 'long_arm_cm' not in json.dumps(adapter.run_params())
        for bad in ('abc', '-1'):
            try:
                adapter.command('set_long_arm', {'value': bad})
                raise AssertionError('a bad long arm was accepted')
            except ValueError:
                pass
        adapter.command('set_long_arm', {'value': ''})
        assert adapter.long_arm_cm is None

        # presets keep the recipe, never the place
        adapter.command('preset_save', {'name': 'slow'})
        adapter.command('set_param', {'key': 'capture.CAPTURE_DURATION_S',
                                      'value': 9})
        adapter.command('set_param', {'key': 'capture.OUTPUT_ROOT',
                                      'value': tmp + '/other'})
        adapter.command('preset_load', {'name': 'slow'})
        assert adapter.edited['capture.CAPTURE_DURATION_S'] == 2.5
        assert adapter.edited['capture.OUTPUT_ROOT'] == tmp + '/other'
        adapter.command('reset_param', {'key': 'capture.CAPTURE_DURATION_S'})
        assert 'capture.CAPTURE_DURATION_S' not in adapter.edited

        # the folder is judged before anything is recorded
        assert sp.check_folder(tmp)[0]
        assert sp.check_folder(tmp + '/new/deeper')[0]       # will be created
        assert not sp.check_folder('relative/path')[0]
        assert not sp.check_folder('')[0]
        adapter.command('set_param', {'key': 'capture.OUTPUT_ROOT', 'value': ''})
        try:
            adapter.run_params()
        except ValueError as error:
            assert 'folder' in str(error)
        else:
            raise AssertionError('a run needs a save folder')

        # browse fills the textbox through the dialog; cancelling changes nothing
        sp.SyncedPipelineAdapter.pick_folder_fn = staticmethod(lambda start: tmp)
        assert adapter.command('pick_folder', {})['path'] == tmp
        assert adapter.edited['capture.OUTPUT_ROOT'] == tmp
        sp.SyncedPipelineAdapter.pick_folder_fn = staticmethod(lambda start: None)
        assert adapter.command('pick_folder', {})['path'] is None
        assert adapter.edited['capture.OUTPUT_ROOT'] == tmp
    del registry['loans']['picoscope:S']
    try:
        adapter.run_params()
    except ValueError as error:
        assert 'PicoScope' in str(error)
    else:
        raise AssertionError('a run needs the scope')
    from server import is_virtual
    assert is_virtual(adapter) and not is_virtual(Stub('picoscope'))
    print('synced-pipeline adapter ok (gate counts lent devices, values '
          'validated and persisted, presets spare the folder)')


def check_synced_pipeline_run():
    """Starting, streaming and stopping a run, against a stand-in script: the
    output reaches the box line by line, the step and session are picked out,
    the run's parameters arrive in the environment, and a stop kills the whole
    tree and hands the borrowed devices back. No hardware."""
    import json as _json
    import os
    import sys
    import tempfile
    from adapters import synced_pipeline as sp
    from pico_scope import run_config

    class Stub:
        def __init__(self, type_name):
            self.type_name = type_name

    sp.SyncedPipelineAdapter.registry = staticmethod(lambda: (
        {'ximea_camera:SN1': Stub('ximea_camera'), 'picoscope:S': Stub('picoscope')},
        {}))
    returned = []
    sp.SyncedPipelineAdapter.return_loans = staticmethod(returned.append)

    def wait(condition, what, timeout=10.0):
        end = time.time() + timeout
        while time.time() < end:
            if condition():
                return
            time.sleep(0.05)
        raise AssertionError(f'timed out waiting for {what}')

    quick = """
import json, os, sys
p = json.load(open(os.environ['MODE_VIDEO_PARAMS']))
print('=== mode_video_capture.py ===')
print('duration', p['sections']['capture']['CAPTURE_DURATION_S'])
print('SESSION_PATH=C:/somewhere/session')
print('=== mode_video_sync.py ===')
sys.exit(int(os.environ.get('STUB_EXIT', '0')))
"""
    slow = """
import time
print('working', flush=True)
time.sleep(60)
"""

    with tempfile.TemporaryDirectory() as tmp:
        adapter = sp.SyncedPipelineAdapter('main')
        adapter.command('set_param', {'key': 'capture.OUTPUT_ROOT', 'value': tmp})
        adapter.command('set_param', {'key': 'capture.CAPTURE_DURATION_S',
                                      'value': 3})
        events = adapter.add_listener()

        sp.SyncedPipelineAdapter.pipeline_command = staticmethod(
            lambda: [sys.executable, '-u', '-c', quick])
        adapter.command('start', {})
        wait(lambda: not adapter.run['running'], 'the quick run to end')
        run = adapter.run
        assert run['returncode'] == 0 and run['error'] is None, run
        assert run['session'] == 'C:/somewhere/session', run
        assert run['step'] == 'mode_video_sync', run
        assert 'duration 3.0' in adapter.describe()['log'], adapter.describe()['log']
        seen = []
        while not events.empty():
            seen.append(events.get_nowait())
        assert any(e['type'] == 'log' and e['line'] == 'duration 3.0' for e in seen)
        assert not os.path.exists(os.environ.get(run_config.PARAMS_ENV_VAR, 'x'))

        # a failing run says so
        os.environ['STUB_EXIT'] = '3'
        adapter.command('start', {})
        wait(lambda: not adapter.run['running'], 'the failing run to end')
        assert adapter.run['returncode'] == 3 and 'code 3' in adapter.run['error']
        os.environ.pop('STUB_EXIT')

        # a run that cannot start does not start, and two cannot overlap
        sp.SyncedPipelineAdapter.pipeline_command = staticmethod(
            lambda: [sys.executable, '-u', '-c', slow])
        adapter.command('start', {})
        wait(lambda: 'working' in adapter.describe()['log'], 'the slow run')
        try:
            adapter.command('start', {})
        except ValueError as error:
            assert 'already' in str(error)
        else:
            raise AssertionError('a second run should be refused')

        adapter.command('stop', {})
        assert adapter.run['running'] is False and adapter.run['stopped'], adapter.run
        assert adapter._process.poll() is not None, 'the run must be dead'
        assert returned == [sp.LOAN_BORROWER], returned
        try:
            adapter.command('stop', {})
        except ValueError:
            pass
        else:
            raise AssertionError('nothing to stop')
    print('synced-pipeline run ok (streams, parses steps, stops the tree, '
          'returns the loans)')


def check_layout():
    """The dashboard layout is kept: devices open at shutdown come back, a
    device the user closed does not, positions are merged and never forgotten,
    and an unavailable device stays in the layout. Runs against a temporary
    layout file and the dummy camera, restoring the server's own afterwards."""
    import tempfile
    import server

    saved_path, saved_layout = server.LAYOUT_PATH, dict(server.layout)
    saved_devices = dict(server.devices)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            server.LAYOUT_PATH = Path(tmp) / 'layout.json'
            server.layout.update({'devices': [], 'boxes': {}})
            server.devices.clear()

            device = api('/api/devices', 'POST',
                         {'type': 'dummy_camera', 'address': 'layout-test'})
            device_id = device['device_id']
            api('/api/devices', 'POST', {'type': 'dummy_camera', 'address': 'second'})
            assert [d['device_id'] for d in api('/api/layout')['devices']] == [
                device_id, 'dummy_camera:second']

            # positions merge: a page with only some boxes cannot forget the rest
            api('/api/layout/boxes', 'PUT', {'boxes': {
                device_id: {'x': 1, 'y': 2, 'w': 5, 'h': 6}}})
            api('/api/layout/boxes', 'PUT', {'boxes': {
                'dummy_camera:second': {'x': 6, 'y': 0, 'w': 4, 'h': 3}}})
            boxes = api('/api/layout')['boxes']
            assert boxes[device_id] == {'x': 1, 'y': 2, 'w': 5, 'h': 6}, boxes
            assert boxes['dummy_camera:second']['w'] == 4
            try:
                api('/api/layout/boxes', 'PUT', {'boxes': {device_id: {'x': 1}}})
            except urllib.error.HTTPError as error:
                assert error.code == 400
            else:
                raise AssertionError('an incomplete box should be refused')

            # the idle shutdown closes the devices but keeps the layout
            server.close_all_devices()
            assert not server.devices
            assert len(api('/api/layout')['devices']) == 2

            # ... so a restart brings them back (plus one that cannot open)
            with server.layout_lock:
                server.layout['devices'].append(
                    {'device_id': 'nosuch:x', 'type': 'nosuch', 'address': 'x'})
            server.restore_layout()
            assert device_id in server.devices
            assert 'dummy_camera:second' in server.devices
            assert any(d['device_id'] == 'nosuch:x'
                       for d in api('/api/layout')['devices']), 'kept for later'

            # closing a box is the user saying "not next time"; its place stays
            api(f'/api/devices/{device_id}', 'DELETE')
            assert device_id not in [d['device_id'] for d in api('/api/layout')['devices']]
            assert api('/api/layout')['boxes'][device_id]['x'] == 1
            assert json.loads(server.LAYOUT_PATH.read_text())['boxes'][device_id]
            for other in ('dummy_camera:second',):
                api(f'/api/devices/{other}', 'DELETE')
    finally:
        server.LAYOUT_PATH = saved_path
        server.layout.clear()
        server.layout.update(saved_layout)
        server.devices.clear()
        server.devices.update(saved_devices)
    print('layout ok (devices and positions kept, user closes forgotten, '
          'unavailable devices retained)')


def check_camera_restore_order():
    """A camera's saved ROI is restored before its saved exposure and rate: the
    rate it can sustain depends on the geometry, and restoring the rate on the
    whole sensor clipped it for good. No hardware: the calls are recorded."""
    from adapters.dummy_camera import DummyCameraAdapter

    adapter = DummyCameraAdapter('order-test')
    calls = []
    adapter._set_roi = lambda *args: calls.append('roi')
    adapter._apply_setting = lambda name, value: calls.append(name)
    adapter.restore_settings({
        'settings': {'exposure': 5914.0, 'gain': 5.0, 'framerate': 167.3},
        'roi': {'x': 832, 'y': 890, 'width': 476, 'height': 468}})
    assert calls[0] == 'roi', calls
    assert set(calls[1:]) == {'exposure', 'gain', 'framerate'}, calls
    print('camera restore order ok (ROI before exposure and rate)')


def check_synced_pipeline_show():
    """The box's "show a capture" button: the folder is chosen (or refused),
    the viewer is started as its own process on that folder with the box's plot
    parameters, a failing viewer says why, and it needs no camera or scope.
    No hardware: a stand-in script plays the viewer."""
    import json as _json
    import sys
    import tempfile
    from adapters import synced_pipeline as sp

    sp.SyncedPipelineAdapter.registry = staticmethod(lambda: ({}, {}))
    adapter = sp.SyncedPipelineAdapter('main')
    assert adapter.describe()['ready'] is False        # and still works below

    def wait(condition, what, timeout=10.0):
        end = time.time() + timeout
        while time.time() < end:
            if condition():
                return
            time.sleep(0.05)
        raise AssertionError(f'timed out waiting for {what}')

    viewer = """
import json, os, sys
params = json.load(open(os.environ['MODE_VIDEO_PARAMS']))
assert sys.argv[1] == '--session', sys.argv
print('shade', params.get('sections', {}).get('show', {}).get('SHADE_ALPHA'))
if os.environ.get('STUB_FAIL'):
    print('no alignment in this capture')
    sys.exit(2)
"""
    sp.SyncedPipelineAdapter.show_command = staticmethod(
        lambda folder: [sys.executable, '-u', '-c', viewer, '--session', folder])

    with tempfile.TemporaryDirectory() as tmp:
        capture = Path(tmp) / '2026-10-05_112839'
        capture.mkdir()
        (capture / '2026-10-05_112839_session.json').write_text('{}')
        (Path(tmp) / 'empty').mkdir()

        # not a capture folder: refused before anything starts
        for bad in (str(Path(tmp) / 'empty'), str(Path(tmp) / 'missing')):
            try:
                adapter.command('show_session', {'path': bad})
            except ValueError as error:
                assert 'not a capture folder' in str(error)
            else:
                raise AssertionError(f'{bad} should have been refused')

        # no path: the folder window decides; cancelling starts nothing
        sp.SyncedPipelineAdapter.pick_folder_fn = staticmethod(lambda start: None)
        assert adapter.command('show_session', {})['path'] is None
        assert adapter.show['folder'] is None

        sp.SyncedPipelineAdapter.pick_folder_fn = staticmethod(
            lambda start: str(capture))
        adapter.command('set_param', {'key': 'show.SHADE_ALPHA', 'value': 0.4})
        events = adapter.add_listener()
        result = adapter.command('show_session', {})
        assert result['path'] == str(capture), result
        wait(lambda: not adapter.show['running'], 'the viewer to end')
        assert adapter.show['folder'] == str(capture) and adapter.show['error'] is None
        seen = []
        while not events.empty():
            seen.append(events.get_nowait())
        assert any(e['type'] == 'show' for e in seen), seen

        # a viewer that fails says why
        os.environ['STUB_FAIL'] = '1'
        adapter.command('show_session', {'path': str(capture)})
        wait(lambda: adapter.show['error'], 'the failure to be reported')
        assert 'no alignment' in adapter.show['error'], adapter.show
        os.environ.pop('STUB_FAIL')
    print('synced-pipeline show ok (folder checked, viewer launched with the '
          'plot parameters, failures reported)')


def check_picoscope_restore():
    """The saved channels come back even when channel A - the only one on
    when the scope opens - is saved disabled. Disabling A first would leave
    no channel on, which the scope refuses, and the whole restore used to be
    abandoned over it. No hardware: the scope is configured without opening."""
    from adapters.picoscope import PicoScopeAdapter

    adapter = PicoScopeAdapter('TEST')
    off = {'enabled': False, 'coupling': 'DC', 'range_v': 5.0}
    adapter.restore_settings({'sample_rate_hz': 1000.0, 'window_s': 10.0,
                              'channels': {
        'A': off, 'B': off, 'C': off,
        'D': {'enabled': True, 'coupling': 'AC', 'range_v': 0.05}}})
    channels = adapter.scope.channels
    assert [n for n, c in channels.items() if c['enabled']] == ['D'], channels
    assert channels['D'] == {'enabled': True, 'coupling': 'AC',
                             'range_v': 0.05}, channels['D']
    print('picoscope restores its saved channels (D only, off A) ok')


def check_camera_markers():
    """set_markers stores a validated list, broadcasts it, survives the
    settings snapshot (device_state.json), and refuses a malformed marker
    without losing the list it had. No hardware: the dummy is not opened."""
    from adapters.dummy_camera import DummyCameraAdapter

    adapter = DummyCameraAdapter('synthetic-0')
    events = []
    adapter.emit = events.append
    markers = [{'id': 'a', 'label': '45 cm', 'x': 512.3, 'y': 300, 'r': 20,
                'visible': True, 'color': '#00dcdc'},
               {'id': 'b', 'label': '46 cm', 'x': 530, 'y': 310, 'r': 20,
                'visible': False, 'color': '#e8a63e'}]
    adapter.command('set_markers', {'markers': markers})
    assert events[-1] == {'type': 'markers',
                          'markers': adapter.describe()['markers']}, events
    assert [m['label'] for m in adapter.describe()['markers']] == ['45 cm',
                                                                   '46 cm']
    snapshot = adapter.settings_snapshot()

    for bad in ([{'id': 'c', 'x': 'nan', 'y': 0, 'r': 1}],
                [{'id': 'd', 'x': 0, 'y': 0, 'r': 1}] * 2,     # duplicate id
                [{'id': 'e', 'label': 'x' * 41, 'x': 0, 'y': 0, 'r': 1}]):
        try:
            adapter.command('set_markers', {'markers': bad})
        except ValueError:
            pass
        else:
            raise AssertionError(f'accepted a malformed marker list: {bad}')
    assert len(adapter.describe()['markers']) == 2, 'a refusal lost the list'

    fresh = DummyCameraAdapter('synthetic-0')
    fresh.restore_settings({'markers': snapshot['markers']})
    assert fresh.describe()['markers'] == snapshot['markers']
    fresh.restore_settings({'markers': [{'x': 1}]})   # malformed: dropped
    assert fresh.describe()['markers'] == []
    print('camera markers validate, broadcast and persist ok')


async def close_code(socket, timeout=5):
    """Read until the server closes the socket; return its close code."""
    try:
        while True:
            await asyncio.wait_for(socket.recv(), timeout=timeout)
    except websockets.exceptions.ConnectionClosed as closed:
        return closed.rcvd.code if closed.rcvd else None


async def check_loans(address):
    """Lending a device to a script: it closes (viewers told 4005, not the
    final 4004), stays listed as a loan, cannot be re-opened meanwhile, and
    comes back with its settings when returned. Then the same through
    loan_client, the way mode_video_capture.py uses it."""
    import loan_client

    device = api('/api/devices', 'POST',
                 {'type': 'dummy_camera', 'address': address})
    device_id = device['device_id']
    api(f'/api/devices/{device_id}/command', 'POST',
        {'name': 'set_setting', 'args': {'name': 'exposure', 'value': 7000}})
    uri = f'ws://{HOST}:{PORT}/ws/devices/{device_id}'

    async with websockets.connect(uri) as socket:
        api(f'/api/devices/{device_id}/lend', 'POST', {'borrower': 'smoke'})
        code = await close_code(socket)
        assert code == 4005, f'attached viewer: expected 4005, got {code}'
    async with websockets.connect(uri) as socket:
        code = await close_code(socket)
        assert code == 4005, f'new viewer: expected 4005, got {code}'
    assert api('/api/devices') == []
    loans = api('/api/loans')
    assert [loan['device_id'] for loan in loans] == [device_id], loans
    assert loans[0]['borrower'] == 'smoke', loans
    assert loans[0]['describe']['type'] == 'dummy_camera', loans
    try:
        api('/api/devices', 'POST', {'type': 'dummy_camera', 'address': address})
        raise AssertionError('a device on loan must not re-open')
    except urllib.error.HTTPError as error:
        assert error.code == 409, error.code
    again = api(f'/api/devices/{device_id}/lend', 'POST', {'borrower': 'x'})
    assert again['borrower'] == 'smoke', 'a second lend must be a no-op'
    print('lend ok (viewers get 4005, listed in /api/loans, re-open refused)')

    returned = api(f'/api/devices/{device_id}/return', 'POST')
    exposure = next(s['value'] for s in returned['settings']
                    if s['name'] == 'exposure')
    assert exposure == 7000, f'expected exposure 7000 after return, got {exposure}'
    assert api('/api/loans') == []
    async with websockets.connect(uri) as socket:
        message = await asyncio.wait_for(socket.recv(), timeout=5)
        assert isinstance(message, bytes), 'expected a frame after the return'
    print('return ok (re-opened with its settings, viewers stream again)')

    # closing the box while on loan cancels the loan: the return then
    # finds nothing to re-open and the device stays closed
    api(f'/api/devices/{device_id}/lend', 'POST', {'borrower': 'smoke'})
    api(f'/api/devices/{device_id}', 'DELETE')
    assert api('/api/loans') == [] and api('/api/devices') == []
    assert loan_client.give_back(device_id, BASE) is False
    print('closing a box on loan cancels the loan ok')

    # the client, as the capture script uses it
    api('/api/devices', 'POST', {'type': 'dummy_camera', 'address': address})
    notes = []
    with loan_client.borrow_from_kalishlot(
            lambda d: d['type'] == 'dummy_camera', 'smoke client',
            url=BASE, log=notes.append) as lent:
        assert list(lent) == [device_id], lent
        assert lent[device_id]['type'] == 'dummy_camera', lent
        assert api('/api/devices') == []
    assert [d['device_id'] for d in api('/api/devices')] == [device_id]
    try:
        with loan_client.borrow_from_kalishlot(
                lambda d: True, 'smoke client', url=BASE, log=notes.append):
            raise KeyboardInterrupt
    except KeyboardInterrupt:
        pass
    assert [d['device_id'] for d in api('/api/devices')] == [device_id], \
        'an interrupted borrower must still give the device back'
    with loan_client.borrow_from_kalishlot(
            lambda d: False, 'smoke client', url=BASE, log=notes.append) as lent:
        assert not lent
    with loan_client.borrow_from_kalishlot(
            lambda d: True, 'x', url='http://127.0.0.1:1', log=notes.append) as lent:
        assert not lent, 'no kalishlot running must mean nothing to borrow'
    api(f'/api/devices/{device_id}', 'DELETE')
    print('loan_client ok (returns on exit and on Ctrl+C; no server is a no-op)')


async def check_idle_watchdog(address):
    """Idle timeout: warning -> dismissal restarts the countdown -> a second
    warning left alone closes every device. Runs with the timeouts shrunk to
    fractions of a second (the watchdog reads both globals every tick)."""
    import server

    async def next_event(socket, expected):
        message = await asyncio.wait_for(socket.recv(), timeout=10)
        event = json.loads(message)
        assert event['type'] == expected, f'expected {expected}, got {event}'
        return event

    server.IDLE_TIMEOUT_S, server.IDLE_GRACE_S = 1.0, 3.0
    try:
        uri = f'ws://{HOST}:{PORT}/ws/idle'
        async with websockets.connect(uri) as socket:
            device = api('/api/devices', 'POST',
                         {'type': 'dummy_camera', 'address': address})
            event = await next_event(socket, 'idle_warning')
            assert event['grace_s'] == 3.0, event
            state = api('/api/idle')
            assert state['warning_active'] and state['grace_left_s'] <= 3.0, state
            print('idle warning ok (also reported by GET /api/idle)')

            api('/api/idle/dismiss', 'POST')
            await next_event(socket, 'idle_clear')
            assert len(api('/api/devices')) == 1, 'dismissal must keep devices open'
            print('dismissal restarts the countdown ok')

            # ignore this one: the devices must go down
            await next_event(socket, 'idle_warning')
            event = await next_event(socket, 'idle_disconnected')
            assert event['devices'] == [device['device_id']], event
            assert len(api('/api/devices')) == 0
            print('ignored warning disconnects all devices ok')
    finally:
        server.IDLE_TIMEOUT_S, server.IDLE_GRACE_S = 3600.0, 10.0


def main():
    server = start_server()
    try:
        types = api('/api/device-types')
        assert any(t['type'] == 'dummy_camera' for t in types)
        print('device types ok:', [t['type'] for t in types])

        available = api('/api/device-types/dummy_camera/available')
        assert len(available) >= 1
        print('available ok:', [d['address'] for d in available])

        device = api('/api/devices', 'POST',
                     {'type': 'dummy_camera', 'address': available[0]['address']})
        device_id = device['device_id']
        assert device['existing'] is False
        print('opened', device_id)

        again = api('/api/devices', 'POST',
                    {'type': 'dummy_camera', 'address': available[0]['address']})
        assert again['existing'] is True
        print('reopen attaches to existing device ok')

        # Start from the whole sensor. The ROI is persisted like any other
        # setting, so a run that died before its clean-up would otherwise
        # hand the next run a cropped camera — and the fit checks below
        # measure a beam that is only in the uncropped frame.
        api(f'/api/devices/{device_id}/command', 'POST',
            {'name': 'clear_roi', 'args': {}})

        asyncio.run(check_stream(device_id))
        # before check_roi: that one deliberately leaves an ROI behind for
        # the persistence check, and the beam orbits outside it
        asyncio.run(check_levels(device_id))
        check_levels_window()
        asyncio.run(check_roi(device_id))
        check_streamer_roi()

        assert len(api('/api/devices')) == 1
        # a viewer still attached when the device is closed must be told
        # (close code 4004), not left listening to a dead adapter
        asyncio.run(check_close_notification(device_id))
        assert len(api('/api/devices')) == 0
        print('close ok (attached viewer notified with 4004)')

        # settings persistence: the exposure set earlier (5000) must come
        # back when the device is re-opened after having been closed
        device = api('/api/devices', 'POST',
                     {'type': 'dummy_camera', 'address': available[0]['address']})
        exposure = next(s['value'] for s in device['settings']
                        if s['name'] == 'exposure')
        assert exposure == 5000, f'expected persisted exposure 5000, got {exposure}'
        assert device['roi'] == {'x': 100, 'y': 200, 'width': 300,
                                 'height': 400}, device['roi']
        assert device['sensor_shape'] == [400, 300], device['sensor_shape']
        api(f'/api/devices/{device_id}/command', 'POST',
            {'name': 'clear_roi', 'args': {}})   # leave the device uncropped
        api(f'/api/devices/{device_id}', 'DELETE')
        print('re-open restores persisted settings ok (exposure 5000, ROI 300x400)')

        check_picoscope_restore()
        check_picoscope_rigid_envelope()
        check_synced_pipeline()
        check_synced_pipeline_run()
        check_layout()
        check_synced_pipeline_show()
        check_camera_restore_order()
        check_camera_markers()
        check_exposure_rate()
        asyncio.run(check_loans(available[0]['address']))
        asyncio.run(check_idle_watchdog(available[0]['address']))

        page = urllib.request.urlopen(f'{BASE}/').read().decode()
        assert 'OS Lab Dashboard' in page
        print('static page ok')

        print('\nsmoke test passed')
    finally:
        server.should_exit = True


if __name__ == '__main__':
    main()
