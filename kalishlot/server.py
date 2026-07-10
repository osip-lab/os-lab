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
import threading
from pathlib import Path

import cv2
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from adapters.basler import BaslerCameraAdapter
from adapters.dummy_camera import DummyCameraAdapter
from adapters.rigol_dg import RigolDGAdapter

DEVICE_TYPES = {cls.type_name: cls
                for cls in (DummyCameraAdapter, BaslerCameraAdapter,
                            RigolDGAdapter)}

JPEG_QUALITY = 80
FRAME_POLL_S = 1 / 30  # how often each websocket checks for a newer frame

app = FastAPI(title='OS Lab dashboard')

devices = {}  # device_id -> adapter instance
devices_lock = threading.Lock()


def device_or_404(device_id):
    with devices_lock:
        adapter = devices.get(device_id)
    if adapter is None:
        raise HTTPException(status_code=404, detail=f'no open device {device_id!r}')
    return adapter


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
        devices[device_id] = adapter
    return {'device_id': device_id, 'existing': False, **adapter.describe()}


@app.get('/api/devices')
def list_open_devices():
    with devices_lock:
        return [{'device_id': device_id, **adapter.describe()}
                for device_id, adapter in devices.items()]


@app.delete('/api/devices/{device_id}')
def close_device(device_id: str):
    with devices_lock:
        adapter = devices.pop(device_id, None)
    if adapter is None:
        raise HTTPException(status_code=404, detail=f'no open device {device_id!r}')
    adapter.close()
    return {'ok': True}


class CommandRequest(BaseModel):
    name: str
    args: dict = {}


@app.post('/api/devices/{device_id}/command')
def device_command(device_id: str, request: CommandRequest):
    adapter = device_or_404(device_id)
    try:
        return adapter.command(request.name, request.args)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


# ----------------------------------------------------------------- streaming
def encode_jpeg(image):
    ok, encoded = cv2.imencode('.jpg', image,
                               [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise RuntimeError('JPEG encoding failed')
    return encoded.tobytes()


@app.websocket('/ws/devices/{device_id}')
async def device_stream(websocket: WebSocket, device_id: str):
    """Per-viewer stream: binary messages are JPEG frames (newest only),
    text messages are JSON events (settings applied, status, fit results)."""
    with devices_lock:
        adapter = devices.get(device_id)
    if adapter is None:
        await websocket.close(code=4004)
        return
    await websocket.accept()
    listener = adapter.add_listener()
    loop = asyncio.get_running_loop()
    last_frame_id = 0
    try:
        while True:
            # forward queued events
            while True:
                try:
                    event = listener.get_nowait()
                except Exception:
                    break
                await websocket.send_text(json.dumps(event))
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
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8090)
