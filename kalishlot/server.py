"""Unified lab web GUI server.

Run:
    python server.py              (from the kalishlot folder)
    python kalishlot/server.py    (from the repo root)

Then open http://localhost:8090 — or http://<this-pc>:8090 from any computer
on the lab network (allow Python through the Windows Firewall when prompted).
(Port 8090: on this PC, 8000 is reserved by Windows and 8080 is in use.)

The server owns the devices: they stay connected and running when no browser
is viewing. Boxes in the browser re-attach to already-open devices on reload.
"""

import asyncio
import json
import os
import threading
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

# analysis code (e.g. the cavity-design NA simulation) may import matplotlib
# and even call plt.show(); the server must never open GUI windows, and doing
# so from a worker thread crashes on some backends
os.environ.setdefault('MPLBACKEND', 'Agg')

import cv2
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from adapters.basler import BaslerCameraAdapter
from adapters.dummy_camera import DummyCameraAdapter
from adapters.picoscope import PicoScopeAdapter
from adapters.rigol_dg import RigolDGAdapter

DEVICE_TYPES = {cls.type_name: cls
                for cls in (DummyCameraAdapter, BaslerCameraAdapter,
                            RigolDGAdapter, PicoScopeAdapter)}

JPEG_QUALITY = 80
FRAME_POLL_S = 1 / 30  # how often each websocket checks for a newer frame
# bulky periodic data events: a slow viewer gets only the newest one, so a
# stalled browser tab can never build a backlog (same rule as video frames)
COALESCE_EVENT_TYPES = {'scope_data', 'brightness'}

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    # a device left open at process exit (Ctrl+C, terminal closed) never gets
    # its close() called otherwise — for hardware like the Basler camera that
    # leaves the driver's exclusive-open lock stuck until the device is
    # physically unplugged/replugged, even for other programs (e.g. pylon
    # Viewer). Closing here on a clean shutdown avoids that.
    with devices_lock:
        open_devices = list(devices.items())
    for device_id, adapter in open_devices:
        record_settings(device_id, adapter)
        try:
            adapter.close()
        except Exception:
            pass


app = FastAPI(title='OS Lab dashboard', lifespan=lifespan)

devices = {}  # device_id -> adapter instance
devices_lock = threading.Lock()

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


def device_or_404(device_id):
    with devices_lock:
        adapter = devices.get(device_id)
    if adapter is None:
        raise HTTPException(status_code=404, detail=f'no open device {device_id!r}')
    return adapter


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


@app.post('/api/devices')
def open_device(request: OpenRequest):
    cls = DEVICE_TYPES.get(request.type)
    if cls is None:
        raise HTTPException(status_code=404, detail=f'unknown device type {request.type!r}')
    device_id = f'{request.type}:{request.address}'
    with devices_lock:
        existing = devices.get(device_id)
        if existing is not None:
            # already open (e.g. another viewer's box): attach, don't reopen
            return {'device_id': device_id, 'existing': True, **existing.describe()}
        adapter = cls(request.address)
        try:
            adapter.open()
        except Exception as error:
            adapter.close()
            raise HTTPException(
                status_code=409,
                detail=f'could not connect to {request.address}: {error}')
        snapshot = saved_settings.get(device_id)
        if snapshot is not None:
            try:
                adapter.restore_settings(snapshot)
            except Exception:
                pass  # a stale snapshot must never block opening the device
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
    if adapter is None:
        raise HTTPException(status_code=404, detail=f'no open device {device_id!r}')
    record_settings(device_id, adapter)
    adapter.close()
    return {'ok': True}


class CommandRequest(BaseModel):
    name: str
    args: dict = {}


@app.post('/api/devices/{device_id:path}/command')
def device_command(device_id: str, request: CommandRequest):
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
    threading.Timer(1.5, record_settings, args=(device_id, adapter)).start()
    return result


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
    if adapter is None:
        # accept first, then close: a pre-accept close surfaces as a bare
        # 403 handshake rejection and the client never sees the 4004 code
        await websocket.accept()
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
                if devices.get(device_id) is not adapter:
                    await websocket.close(code=4004, reason='device closed')
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
    # pop the dashboard in the local browser once the server is up (other
    # computers browse to this PC's address themselves). Timer, not a startup
    # hook: importing the app (smoke test, scripts) must never open a browser.
    # NOT 0.0.0.0 (the address uvicorn logs): that is the bind-to-all-
    # interfaces address, browsers cannot open it.
    threading.Timer(1.0, webbrowser.open, ['http://127.0.0.1:8090']).start()
    uvicorn.run(app, host='0.0.0.0', port=8090)

