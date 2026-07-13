"""End-to-end smoke test of the web GUI server using the dummy camera.

Needs no hardware and no browser. Start the server is NOT required — this
script launches its own instance on a test port, runs the checks, and shuts
it down.

    python smoke_test.py
"""

import asyncio
import json
import threading
import time
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
    import uvicorn
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

        asyncio.run(check_stream(device_id))

        assert len(api('/api/devices')) == 1
        # a viewer still attached when the device is closed must be told
        # (close code 4004), not left listening to a dead adapter
        asyncio.run(check_close_notification(device_id))
        assert len(api('/api/devices')) == 0
        print('close ok (attached viewer notified with 4004)')

        page = urllib.request.urlopen(f'{BASE}/').read().decode()
        assert 'OS Lab Dashboard' in page
        print('static page ok')

        print('\nsmoke test passed')
    finally:
        server.should_exit = True


if __name__ == '__main__':
    main()
