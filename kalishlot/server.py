"""Unified lab web GUI server.

Run:
    python server.py              (from the kalishlot folder)
    python kalishlot/server.py    (from the repo root)
    python server.py -t 7200      (idle timeout in seconds, 0 disables it)

Then open http://localhost:8090 — or http://<this-pc>:8090 from any computer
on the lab network (allow Python through the Windows Firewall when prompted).
(Port 8090: on this PC, 8000 is reserved by Windows and 8080 is in use.)

The server owns the devices: they stay connected and running when no browser
is viewing. Boxes in the browser re-attach to already-open devices on reload.

Because the devices keep running unattended, an idle watchdog closes them all
after IDLE_TIMEOUT_S without user activity — so a camera forgotten at the end
of the day is not left exposing all night. It warns in the browser first (with
a sound) and the warning can be dismissed to restart the countdown.
"""

import argparse
import asyncio
import json
import os
import subprocess
import sys
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

# analysis code (e.g. the cavity-design NA simulation) may import matplotlib
# and even call plt.show(); the server must never open GUI windows, and doing
# so from a worker thread crashes on some backends. The launched pipelines get
# the backend the user had instead - their viewer is a window (PIPELINES).
_USER_MPLBACKEND = os.environ.get('MPLBACKEND')
os.environ.setdefault('MPLBACKEND', 'Agg')

import cv2
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from adapters.basler import BaslerCameraAdapter
from adapters.dummy_camera import DummyCameraAdapter
from adapters.picoscope import PicoScopeAdapter
from adapters.rigol_dg import RigolDGAdapter
from adapters.synced_pipeline import SyncedPipelineAdapter

_ADAPTERS = [DummyCameraAdapter, BaslerCameraAdapter,
             RigolDGAdapter, PicoScopeAdapter, SyncedPipelineAdapter]

# XIMEA is optional where the others are not: its Python package is not on
# PyPI and has to be copied out of the XIMEA Software Package by hand (see
# requirements.txt), so on a machine without it this import fails. A missing
# camera should cost that camera's box, not the whole server.
try:
    from adapters.ximea import XimeaCameraAdapter
    _ADAPTERS.append(XimeaCameraAdapter)
except ImportError as error:
    print(f'XIMEA support unavailable: {error}')

DEVICE_TYPES = {cls.type_name: cls for cls in _ADAPTERS}

JPEG_QUALITY = 80
FRAME_POLL_S = 1 / 30  # how often each websocket checks for a newer frame
# bulky periodic data events: a slow viewer gets only the newest one, so a
# stalled browser tab can never build a backlog (same rule as video frames)
COALESCE_EVENT_TYPES = {'scope_data', 'brightness', 'levels'}

# ------------------------------------------------------------ idle watchdog
# Seconds of inactivity before the browser gets a shutdown warning; the
# command line (-t) and the __main__ block below both write this global.
# 0 (or less) disables the watchdog entirely.
IDLE_TIMEOUT_S = 3600.0
IDLE_GRACE_S = 10.0    # how long the warning waits for a dismissal
IDLE_TICK_S = 0.25


@asynccontextmanager
async def lifespan(app: FastAPI):
    watchdog = asyncio.create_task(idle_watchdog())
    yield
    watchdog.cancel()
    # a device left open at process exit (Ctrl+C, terminal closed) never gets
    # its close() called otherwise — for hardware like the Basler camera that
    # leaves the driver's exclusive-open lock stuck until the device is
    # physically unplugged/replugged, even for other programs (e.g. pylon
    # Viewer). Closing here on a clean shutdown avoids that.
    close_all_devices()


app = FastAPI(title='OS Lab dashboard', lifespan=lifespan)

devices = {}  # device_id -> adapter instance
devices_lock = threading.Lock()


def is_virtual(adapter):
    """A box with no hardware behind it (the synced-pipeline box): it takes
    no part in the idle shutdown, which exists to protect hardware."""
    return getattr(adapter, 'VIRTUAL', False)

# ------------------------------------------------- settings persistence
# Last-used settings per device, kept on disk so a re-opened device (even
# after a server restart) comes back configured the way it was left.
STATE_PATH = Path(__file__).parent / 'device_state.json'
settings_lock = threading.Lock()
try:
    saved_settings = json.loads(STATE_PATH.read_text())
except Exception:
    saved_settings = {}


def record_settings(device_id, adapter):
    """Snapshot the adapter's settings and persist them when they changed."""
    try:
        snapshot = adapter.settings_snapshot()
    except Exception:
        return
    if snapshot is None:  # device keeps its own state (e.g. Rigol)
        return
    with settings_lock:
        if saved_settings.get(device_id) == snapshot:
            return
        saved_settings[device_id] = snapshot
        try:
            STATE_PATH.write_text(json.dumps(saved_settings, indent=1))
        except Exception:
            pass  # persistence must never break device control


def record_settings_later(device_id, adapter):
    """The delayed pass, for settings a device applies on its own thread.

    Skipped once this adapter is no longer the open device: a closed adapter
    still holds the state it had, and a timer of its own firing after the
    device was closed and re-opened would write that stale state over what
    the live one has recorded since.
    """
    with devices_lock:
        if devices.get(device_id) is not adapter:
            return
    record_settings(device_id, adapter)


def device_or_404(device_id):
    with devices_lock:
        adapter = devices.get(device_id)
    if adapter is None:
        raise HTTPException(status_code=404, detail=f'no open device {device_id!r}')
    return adapter


def close_all_devices():
    """Close and forget every open device. Returns the ids that were closed.

    Devices on loan are forgotten too: the shutdown is meant to leave the lab
    dark, so a script handing one back afterwards must not switch it on again.
    """
    with devices_lock:
        # boxes with no hardware stay: nothing to switch off, and their saved
        # parameters are not worth a restart of the page
        open_devices = [(device_id, adapter)
                        for device_id, adapter in devices.items()
                        if not is_virtual(adapter)]
        for device_id, _ in open_devices:
            del devices[device_id]
        loaned = list(loans)
        loans.clear()
    for device_id, adapter in open_devices:
        record_settings(device_id, adapter)
        try:
            adapter.close()
        except Exception:
            pass
    return [device_id for device_id, _ in open_devices] + loaned


# ------------------------------------------------------------ idle watchdog
# Devices run unattended (the server owns them, no browser needed), so a
# forgotten dashboard would leave e.g. a camera streaming all night. After
# IDLE_TIMEOUT_S without user activity every viewer gets a warning; unless
# somebody dismisses it within IDLE_GRACE_S, all devices are closed.
#
# "Activity" is any deliberate user action reaching the server — opening a
# device, sending it a command, writing to the log, or dismissing the warning
# — but NOT the video/data traffic a running device produces by itself.
idle_lock = threading.Lock()
idle_listeners = set()   # asyncio.Queue, one per connected /ws/idle viewer
_last_activity = time.monotonic()
_warned_at = None        # monotonic time the pending warning was raised


def note_activity():
    global _last_activity
    with idle_lock:
        _last_activity = time.monotonic()


def broadcast_idle(message):
    """Fan a watchdog message out to every viewer. Called from the event loop
    only (asyncio.Queue is not thread-safe), which is why the HTTP endpoints
    just call note_activity() and let the watchdog notice."""
    with idle_lock:
        listeners = list(idle_listeners)
    for listener in listeners:
        try:
            listener.put_nowait(message)
        except asyncio.QueueFull:
            pass


async def idle_watchdog():
    global _warned_at
    while True:
        await asyncio.sleep(IDLE_TICK_S)
        if IDLE_TIMEOUT_S <= 0:  # watchdog disabled
            continue
        now = time.monotonic()
        with idle_lock:
            last_activity, warned_at = _last_activity, _warned_at
        with devices_lock:
            any_open = any(not is_virtual(a) for a in devices.values())

        if warned_at is None:
            # nothing to protect while no device is open, and the countdown
            # then starts fresh from the moment one is opened
            if not any_open:
                note_activity()
            elif now - last_activity >= IDLE_TIMEOUT_S:
                with idle_lock:
                    _warned_at = now
                broadcast_idle({'type': 'idle_warning', 'grace_s': IDLE_GRACE_S,
                                'timeout_s': IDLE_TIMEOUT_S})
        # >= , not > : time.monotonic() on Windows steps in 15.6 ms lumps, so
        # activity in the same lump as the warning reads as exactly equal —
        # and a dismissal must never be the one thing that gets ignored.
        elif last_activity >= warned_at or not any_open:
            with idle_lock:  # dismissed (or the devices went away meanwhile)
                _warned_at = None
            note_activity()
            broadcast_idle({'type': 'idle_clear'})
        elif now - warned_at >= IDLE_GRACE_S:
            with idle_lock:
                _warned_at = None
            # closing can block for a second or two per device (hardware),
            # and the event loop also serves every video stream
            closed = await asyncio.get_running_loop().run_in_executor(
                None, close_all_devices)
            note_activity()
            broadcast_idle({'type': 'idle_disconnected', 'devices': closed})


@app.get('/api/idle')
def get_idle_state():
    """Config plus any warning already in flight, so a viewer that just
    (re)connected joins the countdown instead of missing it."""
    with idle_lock:
        warned_at = _warned_at
    return {
        'timeout_s': IDLE_TIMEOUT_S,
        'grace_s': IDLE_GRACE_S,
        'warning_active': warned_at is not None,
        'grace_left_s': (None if warned_at is None
                         else max(0.0, IDLE_GRACE_S - (time.monotonic() - warned_at))),
    }


@app.post('/api/idle/dismiss')
def dismiss_idle_warning():
    """Dismiss the warning / restart the countdown."""
    note_activity()
    return {'ok': True, 'timeout_s': IDLE_TIMEOUT_S}


@app.websocket('/ws/idle')
async def idle_stream(websocket: WebSocket):
    await websocket.accept()
    listener = asyncio.Queue(maxsize=20)
    with idle_lock:
        idle_listeners.add(listener)
    try:
        while True:
            await websocket.send_text(json.dumps(await listener.get()))
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        with idle_lock:
            idle_listeners.discard(listener)


# --------------------------------------------------------------- shared log
# One text log shared by every box (e.g. camera "record fit values"), newest
# entry first, persisted so it survives a reload/restart and is the same for
# every viewer. Not a device, so it lives here rather than in adapters/.
LOG_STATE_PATH = Path(__file__).parent / 'log_state.json'
MAX_LOG_ENTRIES = 500
log_lock = threading.Lock()
log_listeners = set()  # asyncio.Queue, one per connected /ws/log viewer
try:
    log_entries = json.loads(LOG_STATE_PATH.read_text())
except Exception:
    log_entries = []
_next_log_id = (max((e['id'] for e in log_entries), default=0) + 1)


class LogRequest(BaseModel):
    text: str


@app.get('/api/log')
def get_log():
    with log_lock:
        return list(log_entries)


@app.post('/api/log')
def post_log(request: LogRequest):
    global _next_log_id
    note_activity()
    entry = {'id': _next_log_id, 'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
             'text': request.text}
    _next_log_id += 1
    with log_lock:
        log_entries.insert(0, entry)
        del log_entries[MAX_LOG_ENTRIES:]
        try:
            LOG_STATE_PATH.write_text(json.dumps(log_entries, indent=1))
        except Exception:
            pass  # persistence must never break logging
        listeners = list(log_listeners)
    for listener in listeners:
        try:
            listener.put_nowait(entry)
        except asyncio.QueueFull:
            pass
    return entry


@app.websocket('/ws/log')
async def log_stream(websocket: WebSocket):
    await websocket.accept()
    listener = asyncio.Queue(maxsize=100)
    with log_lock:
        log_listeners.add(listener)
    try:
        while True:
            entry = await listener.get()
            await websocket.send_text(json.dumps({'type': 'entry', 'entry': entry}))
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        with log_lock:
            log_listeners.discard(listener)


# ------------------------------------------------------------------ REST API
@app.get('/api/device-types')
def get_device_types():
    return [{'type': cls.type_name, 'display_name': cls.display_name}
            for cls in DEVICE_TYPES.values()]


@app.get('/api/device-types/{type_name}/available')
def get_available(type_name: str):
    cls = DEVICE_TYPES.get(type_name)
    if cls is None:
        raise HTTPException(status_code=404, detail=f'unknown device type {type_name!r}')
    try:
        return cls.list_available()
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


class OpenRequest(BaseModel):
    type: str
    address: str


def open_adapter(type_name, address):
    """Connect a new adapter and restore its saved settings; the caller holds
    devices_lock and registers it. HTTP 409 when the hardware refuses."""
    adapter = DEVICE_TYPES[type_name](address)
    try:
        adapter.open()
    except Exception as error:
        adapter.close()
        raise HTTPException(
            status_code=409, detail=f'could not connect to {address}: {error}')
    snapshot = saved_settings.get(f'{type_name}:{address}')
    if snapshot is not None:
        try:
            adapter.restore_settings(snapshot)
        except Exception:
            pass  # a stale snapshot must never block opening the device
    return adapter


@app.post('/api/devices')
def open_device(request: OpenRequest):
    note_activity()
    if request.type not in DEVICE_TYPES:
        raise HTTPException(status_code=404, detail=f'unknown device type {request.type!r}')
    device_id = f'{request.type}:{request.address}'
    with devices_lock:
        existing = devices.get(device_id)
        if existing is not None:
            # already open (e.g. another viewer's box): attach, don't reopen
            return {'device_id': device_id, 'existing': True, **existing.describe()}
        if device_id in loans:
            raise HTTPException(
                status_code=409, detail=f'{device_id} is on loan to '
                f'{loans[device_id]["borrower"]}; return it first')
        adapter = open_adapter(request.type, request.address)
        devices[device_id] = adapter
    return {'device_id': device_id, 'existing': False, **adapter.describe()}


@app.get('/api/devices')
def list_open_devices():
    with devices_lock:
        return [{'device_id': device_id, **adapter.describe()}
                for device_id, adapter in devices.items()]


@app.delete('/api/devices/{device_id:path}')
def close_device(device_id: str):
    with devices_lock:
        adapter = devices.pop(device_id, None)
        # closed while on loan: the borrower's return then finds nothing
        # to re-open, and the device stays closed as asked
        cancelled = loans.pop(device_id, None) is not None
    if adapter is None:
        if cancelled:
            return {'ok': True}
        raise HTTPException(status_code=404, detail=f'no open device {device_id!r}')
    record_settings(device_id, adapter)
    adapter.close()
    return {'ok': True}


# ------------------------------------------------------------------- loans
# A standalone script (e.g. pico_scope/mode_video_capture.py) that needs a
# device kalishlot holds borrows it: lending closes the device here and frees
# the hardware but remembers it, and the viewers' boxes wait instead of giving
# up (close code 4005). Returning re-opens it with its saved settings and the
# boxes re-attach by themselves. The client side is loan_client.py.
# A script that dies without returning leaves the device on loan - closed,
# which is the safe state - until a box's "reconnect" button returns it.
# device_id -> {'type', 'address', 'borrower', 'since', 'describe'}, under
# devices_lock. 'describe' is the device's last describe(), so a page loaded
# mid-loan can still build its box (app.js), which then waits like the rest.
loans = {}

# the synced-pipeline box gates itself on which cameras and scope exist, lent
# ones included; it reads these without the lock (describe() runs under it)
SyncedPipelineAdapter.registry = staticmethod(lambda: (devices, loans))


class LendRequest(BaseModel):
    borrower: str = 'a script'


@app.get('/api/loans')
def list_loans():
    with devices_lock:
        return [{'device_id': device_id, **loan}
                for device_id, loan in loans.items()]


@app.post('/api/devices/{device_id:path}/lend')
def lend_device(device_id: str, request: LendRequest):
    note_activity()
    with devices_lock:
        if device_id in loans:  # lending twice is harmless (a retried call)
            return {'ok': True, 'device_id': device_id,
                    'borrower': loans[device_id]['borrower']}
        adapter = devices.pop(device_id, None)
        if adapter is None:
            raise HTTPException(status_code=404, detail=f'no open device {device_id!r}')
        type_name, address = device_id.split(':', 1)
        try:
            describe = adapter.describe()
        except Exception:
            describe = {'type': type_name, 'label': device_id}
        loan = loans[device_id] = {
            'type': type_name, 'address': address, 'borrower': request.borrower,
            'since': datetime.now().isoformat(timespec='seconds'),
            'describe': describe}
    record_settings(device_id, adapter)
    adapter.close()  # returns once the hardware is released
    return {'ok': True, 'device_id': device_id, 'borrower': loan['borrower']}


@app.post('/api/devices/{device_id:path}/return')
def return_device(device_id: str):
    note_activity()
    with devices_lock:
        loan = loans.get(device_id)
        if loan is None:
            raise HTTPException(status_code=404, detail=f'{device_id!r} is not on loan')
        # stays on loan while the hardware is still busy (the borrower has
        # not let go yet): the 409 says so, and the return can be retried
        adapter = open_adapter(loan['type'], loan['address'])
        del loans[device_id]
        devices[device_id] = adapter
    return {'ok': True, 'device_id': device_id, **adapter.describe()}


class CommandRequest(BaseModel):
    name: str
    args: dict = {}


@app.post('/api/devices/{device_id:path}/command')
def device_command(device_id: str, request: CommandRequest):
    note_activity()
    adapter = device_or_404(device_id)
    try:
        result = adapter.command(request.name, request.args)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))
    # persist the settings this command may have changed; the delayed pass
    # catches values that devices apply asynchronously on their own thread
    record_settings(device_id, adapter)
    threading.Timer(1.5, record_settings_later,
                    args=(device_id, adapter)).start()
    return result


# ----------------------------------------------------------------- pipelines
# Standalone scripts a box can launch, e.g. the camera box's "mode video"
# button. Each runs in a console window of its own on THIS PC — the scripts
# print as they go and may ask for input (the capture takes its output folder
# from the clipboard), so they need one — and borrows from kalishlot whatever
# devices it needs, exactly as when started by hand (loan_client.py). The
# console closes by itself on success and waits for a key on failure, so the
# error stays readable.
REPO_ROOT = Path(__file__).resolve().parent.parent
# name -> (script, its arguments)
PIPELINES = {
    # every setting the boxes have comes from them, the rest from the config
    'mode_video': (REPO_ROOT / 'pico_scope' / 'run_mode_video_pipeline.py',
                   ['--from-kalishlot']),
}
pipeline_lock = threading.Lock()
pipeline_runs = {}  # name -> {'process', 'started'}, the latest run of each


def pipeline_state(name):
    run = pipeline_runs.get(name)
    if run is None:
        return {'name': name, 'running': False, 'returncode': None, 'started': None}
    returncode = run['process'].poll()
    return {'name': name, 'running': returncode is None,
            'returncode': returncode, 'started': run['started']}


def pipeline_or_404(name):
    if name not in PIPELINES:
        raise HTTPException(status_code=404, detail=f'no pipeline {name!r}')
    return PIPELINES[name]


@app.get('/api/pipelines/{name}')
def get_pipeline(name: str):
    pipeline_or_404(name)
    with pipeline_lock:
        return pipeline_state(name)


@app.post('/api/pipelines/{name}')
def start_pipeline(name: str):
    script, arguments = pipeline_or_404(name)
    note_activity()
    with pipeline_lock:
        if pipeline_state(name)['running']:
            raise HTTPException(status_code=409,
                                detail=f'{script.name} is already running')
        environment = dict(os.environ)
        if _USER_MPLBACKEND is None:
            environment.pop('MPLBACKEND', None)
        python_command = [sys.executable, str(script), *arguments]
        python = subprocess.list2cmdline(python_command)
        if os.name == 'nt':
            # /s strips exactly the outer quotes, so paths with spaces survive
            command = (f'cmd /s /c "title {script.name} & {python} '
                       f'|| (pause & exit /b 1)"')
            process = subprocess.Popen(
                command, cwd=REPO_ROOT, env=environment,
                creationflags=subprocess.CREATE_NEW_CONSOLE)
        else:
            process = subprocess.Popen(python_command, cwd=REPO_ROOT,
                                       env=environment)
        pipeline_runs[name] = {
            'process': process,
            'started': datetime.now().isoformat(timespec='seconds')}
        return pipeline_state(name)


# ----------------------------------------------------------------- streaming
def encode_jpeg(image):
    ok, encoded = cv2.imencode('.jpg', image,
                               [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise RuntimeError('JPEG encoding failed')
    return encoded.tobytes()


@app.websocket('/ws/devices/{device_id:path}')
async def device_stream(websocket: WebSocket, device_id: str):
    """Per-viewer stream: binary messages are JPEG frames (newest only),
    text messages are JSON events (settings applied, status, fit results).
    ':path' converters: device addresses may contain '/' (e.g. PicoScope
    serial numbers like 10036/0060)."""
    with devices_lock:
        adapter = devices.get(device_id)
        on_loan = device_id in loans
    if adapter is None:
        # accept first, then close: a pre-accept close surfaces as a bare
        # 403 handshake rejection and the client never sees the 4004 code
        await websocket.accept()
        if on_loan:  # lent to a script: the viewer waits for its return
            await websocket.close(code=4005, reason='device on loan')
        else:
            await websocket.close(code=4004, reason='no such device')
        return
    await websocket.accept()
    listener = adapter.add_listener()
    loop = asyncio.get_running_loop()
    last_frame_id = 0
    try:
        while True:
            # forward queued events; of the bulky periodic ones only the
            # newest is sent (state events always all go through)
            events = []
            while True:
                try:
                    events.append(listener.get_nowait())
                except Exception:
                    break
            newest_data = None
            for event in events:
                if event.get('type') in COALESCE_EVENT_TYPES:
                    newest_data = event
                    continue
                await websocket.send_text(json.dumps(event))
            if newest_data is not None:
                await websocket.send_text(json.dumps(newest_data))
            # if the device was closed (by any viewer), tell this one and
            # end the stream instead of lingering on a dead adapter
            with devices_lock:
                gone = devices.get(device_id) is not adapter
                on_loan = device_id in loans
            if gone:
                await websocket.close(code=4005 if on_loan else 4004,
                                      reason='device on loan' if on_loan
                                      else 'device closed')
                break
            # send the newest frame if it changed
            frame_id, frame = adapter.latest_display_frame()
            if frame is not None and frame_id != last_frame_id:
                last_frame_id = frame_id
                payload = await loop.run_in_executor(None, encode_jpeg, frame)
                await websocket.send_bytes(payload)
            await asyncio.sleep(FRAME_POLL_S)
    except WebSocketDisconnect:
        pass
    except Exception:
        pass  # client vanished mid-send; nothing to clean up beyond the listener
    finally:
        adapter.remove_listener(listener)


# ------------------------------------------------------------- static files
@app.middleware('http')
async def no_cache_static(request, call_next):
    """Make browsers revalidate JS/HTML on every load — otherwise a plain
    refresh can keep running stale cached modules after a code update."""
    response = await call_next(request)
    if not request.url.path.startswith('/api'):
        response.headers['Cache-Control'] = 'no-cache'
    return response


app.mount('/', StaticFiles(directory=Path(__file__).parent / 'static', html=True))


if __name__ == '__main__':
    import webbrowser

    import uvicorn

    # ---------------------------------------------------------------------
    # Running from an IDE (PyCharm's green arrow passes no arguments)? Edit
    # this value — the command-line -t only overrides it when given.
    IDLE_TIMEOUT_S = 3600      # seconds of inactivity before the warning
    # ---------------------------------------------------------------------

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        '-t', '--timeout', type=float, default=IDLE_TIMEOUT_S, metavar='SECONDS',
        help=f'close all devices after this many seconds without user activity '
             f'(a dismissible warning appears {IDLE_GRACE_S:g} s earlier); '
             f'0 disables it. Default: %(default)g')
    args = parser.parse_args()
    IDLE_TIMEOUT_S = args.timeout
    print(f'idle watchdog: {IDLE_TIMEOUT_S:g} s' if IDLE_TIMEOUT_S > 0
          else 'idle watchdog: disabled')

    # pop the dashboard in the local browser once the server is up (other
    # computers browse to this PC's address themselves). Timer, not a startup
    # hook: importing the app (smoke test, scripts) must never open a browser.
    # NOT 0.0.0.0 (the address uvicorn logs): that is the bind-to-all-
    # interfaces address, browsers cannot open it.
    threading.Timer(1.0, webbrowser.open, ['http://127.0.0.1:8090']).start()
    uvicorn.run(app, host='0.0.0.0', port=8090)

