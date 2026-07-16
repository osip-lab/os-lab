# Kalishlot — adding a new device type

Kalishlot is the lab's unified browser GUI: a canvas of boxes, one box per
device, served by `server.py` (FastAPI, port **8090** — on this PC port 8000
is reserved by Windows and 8080 is occupied). This document is the recipe for
adding a new device type. It is written for a future Claude session (or any
developer) with no context from the session that created this code.

## Architecture in one minute

```
device layer (pure python)   adapter (kalishlot/adapters/)   frontend box (static/boxes/)
  e.g. basler_cam/            wraps ONE device instance,       plain-JS renderer for
  basler_cameras.py           translates it to the uniform     that device type's box
  — no GUI imports,     -->   vocabulary: open/close,     -->  — talks REST + WebSocket,
  threads + callbacks         describe, command, frames,       registered in app.js
                              events
```

- **The server owns devices.** They stay connected when no browser is open.
  Browsers re-attach on reload (`GET /api/devices` on page load).
- **Multiple viewers** may watch the same device; state changes are broadcast
  as events so all viewers stay in sync.
- **No build step.** Frontend is plain JS ES-modules + vendored libs in
  `static/vendor/` (currently Gridstack). Do not introduce npm/webpack.

## Step 1 — the device layer (pure Python, no GUI imports)

Put the actual instrument control in its own module *outside* kalishlot
(pattern: `basler_cam/basler_cameras.py`). Rules, per the lab's standing
decision (see memory `device-gui-decoupling`):

- No Qt / matplotlib / HTML imports. Plain classes, methods, properties.
- JSON-friendly types at the boundaries (numbers, dicts, numpy arrays).
- If the device streams or needs a background loop, use `threading` with
  callbacks (`on_frame(...)`, `on_error(...)`), not Qt signals. See
  `CameraStreamer` in `basler_cam/basler_cameras.py` for the reference
  pattern, including its `submit(command)` queue: **if the underlying SDK is
  not thread-safe, all device access must happen in the device's own thread**,
  and setting changes are submitted as callables executed between operations.

## Step 2 — the adapter (`kalishlot/adapters/<name>.py`)

Subclass `DeviceAdapter` from `adapters/base.py`. It already provides
thread-safe frame storage and event fan-out; you implement the vocabulary:

```python
from .base import DeviceAdapter

class MyDeviceAdapter(DeviceAdapter):
    type_name = 'my_device'            # api identifier, snake_case
    display_name = 'My device (nice name)'

    @staticmethod
    def list_available():
        # enumerate connectable instances (serial numbers, COM ports, ...)
        return [{'address': '12345', 'label': 'My device s/n 12345'}]

    def __init__(self, address):
        super().__init__(address)
        # build, don't connect yet

    def open(self):
        # connect; raise on failure — server turns it into a clean HTTP 409
        # whose message is shown to the user (e.g. "in use by another program")

    def close(self):
        # stop threads, release the device; must be safe to call twice

    def describe(self):
        return {'type': self.type_name,
                'label': f'{self.display_name} — {self.address}',
                'commands': ['set_setting', ...],   # what command() accepts
                'settings': [                        # rendered as number inputs
                    {'name': 'power', 'label': 'power', 'unit': 'mW',
                     'min': 0, 'max': 100, 'decimals': 1, 'value': 10.0},
                ],
                # cameras also send: 'frame_shape': [h, w], 'playing': bool
                }

    def command(self, name, args):
        # execute quickly or delegate to the device thread; return JSON-able
        # dict; raise ValueError for unknown commands (server -> HTTP 400)
        ...
```

Conventions the existing frontend already understands:

- **Command `set_setting`** with `args={'name': ..., 'value': ...}` changes a
  setting. After applying, `self.emit({'type': 'setting_applied', 'name': n,
  'value': v_actually_accepted})` — emit the value the *hardware* accepted
  (clamped/rounded), the GUI displays it back.
- **Cameras**: commands `play` / `pause` / `snap`; emit
  `{'type': 'status', 'playing': bool}` on play/pause.
- **Cameras from a NEW MANUFACTURER** (Ximea, ...): do NOT write commands or
  describe() yourself — subclass `CameraAdapterBase` from
  `adapters/camera_base.py` and implement only its hardware hooks
  (`_open/_close/_play/_pause/_snap/_apply_setting/_settings_schema/
  _sensor_shape`, plus `_store_camera_frame(frame, display)` from the frame
  thread; override `LEVELS_MAX`, `PIXEL_SIZE_MM`, `DISPLAY_DOWNSAMPLE`).
  The frontend camera box is manufacturer-agnostic: register the new
  type_name in server.py DEVICE_TYPES and point it at the existing
  `createCameraBox` in app.js BOX_RENDERERS — no new JS needed.
  `adapters/dummy_camera.py` is the minimal reference implementation;
  `adapters/basler.py` shows the pattern for a thread-unsafe SDK.
- **Gaussian fit** (cameras): mix in `CameraFitMixin` from
  `adapters/camera_fit.py` — call `_init_fit()` in `__init__`,
  `_store_fit_frame(frame)` from the frame thread with the full-resolution
  frame, `_stop_fit()` in `close()`, merge `fit_describe()` into `describe()`
  and delegate to `fit_command(name, args)` first in `command()`. That adds
  commands `fit_on` / `fit_off` / `set_guess` / `clear_guess` /
  `set_fit_threshold` and events `fit_status`, `fit` (params + row/column
  cuts), `guess`, `fit_threshold`, `brightness`. All fit coordinates are
  full-resolution sensor pixels; the frontend scales them using
  `describe()['sensor_shape']` (add it next to the possibly-downsampled
  `frame_shape`).
  The **fit trigger** handles blinking beams (e.g. resonator transmission
  during a scan): frames whose brightness — mean inside the guess circle's
  bounding square, else a 99th-percentile measure, both in counts above
  background (`gaussian_fit.beam_brightness`) — is below the threshold are
  never submitted to the fit. The check runs in the frame thread *before*
  `FitLoop.submit()` on purpose: the loop keeps only the newest frame, so a
  dark frame arriving right after a bright one would otherwise overwrite it
  before the fit thread wakes. The throttled `brightness` events feed the
  live readout next to the threshold input (it is in `COALESCE_EVENT_TYPES`,
  like `scope_data`).
- **Frames** (optional — control-only devices simply never call this): from
  the device thread call `self._store_display_frame(img)` with a display-ready
  `uint8` grayscale numpy array (12-bit data: `(frame >> 4).astype(np.uint8)`;
  downsample large sensors ~2x, e.g. `frame[::2, ::2]`). The server encodes
  JPEG and streams it; only the newest frame is ever sent (slow viewers skip
  frames, they never lag).
- **Errors** during operation: `self.emit({'type': 'error', 'message': ...})`.
- **Settings persistence**: implement `settings_snapshot()` (JSON dict) and
  `restore_settings(snapshot)` — the server records the snapshot after every
  command and on close (`kalishlot/device_state.json`, gitignored) and
  restores it right after `open()`, so a re-opened device comes back with
  its last-used settings even across server restarts. `CameraAdapterBase`
  implements both generically from the settings schema (set
  `RESTORE_SETTLE_S` if `_apply_setting` is asynchronous); the PicoScope
  restores only what differs (each change restarts streaming). Return None
  from `settings_snapshot()` (the default) for instruments that keep their
  own state, like the Rigol — restoring would overwrite the hardware.
- **Control-only devices** (no frames): see `adapters/rigol_dg.py` +
  `static/boxes/rigol_dg.js` — the reference for instruments that are pure
  settings/state (function generators, power supplies, ...). Broadcast every
  applied change as an event with the instrument-accepted state so all
  viewers stay in sync. NOTE the DG822 rule: connecting must be read-only
  (CH1 often drives the laser temperature scan, hard-capped at 5 Vpp in the
  adapter).
- **Streaming-plot devices** (continuous data points, e.g. oscilloscopes,
  power meters): see `adapters/picoscope.py` + `static/boxes/picoscope.js`.
  Never send or draw per-sample: the device layer accumulates into numpy
  ring buffers (`picoscope/ps4000a_scope.py` `RingBuffer`), an adapter
  thread emits the visible window at ~20 Hz **min/max-envelope-decimated**
  to <= ~1000 points per channel as a JSON event, and the box redraws once
  per event with the vendored uPlot (`static/vendor/uPlot.iife.min.js`,
  loaded globally in index.html). Device addresses may contain '/' (the
  PicoScope serial does) — the server routes use `{device_id:path}` for
  that reason. Register the bulky periodic event type in
  `COALESCE_EVENT_TYPES` in `server.py`: a stalled viewer then receives
  only the newest one instead of a backlog burst.
- **Analysis on a snapshot** (fits on the streamed data): pause captures the
  visible window at FULL resolution into the adapter (`self._snapshot` in
  `adapters/picoscope.py`) and broadcasts one chunk built from it, so every
  viewer's chart shows exactly the frozen data. The analyses themselves are
  pluggable extensions — never written into the device module: server
  classes in `adapters/analyses/`, client modules in
  `static/boxes/extensions/`. **See `ADDING_ANALYSES.md`** for the recipe
  (and for the litmus test deciding whether a feature is an extension at
  all — the camera's live Gaussian fit, woven into the frame pipeline, is
  the canonical non-extension).
- **Frontend sockets**: always connect through `connectDeviceStream` in
  `static/boxes/stream.js` (see any box for usage) — it reconnects
  automatically with backoff and calls your `onReattach(describe)` with a
  fresh `describe()` so the box can resync state it missed while offline.
  Never hand-roll `new WebSocket(...)` in a box.

Register the class in `server.py`:

```python
DEVICE_TYPES = {cls.type_name: cls
                for cls in (DummyCameraAdapter, MyDeviceAdapter)}
```

That is ALL the server needs — the REST/WebSocket plumbing is generic.

## Step 3 — the frontend box (`static/boxes/<name>.js`)

A box renderer is one exported function; look at `static/boxes/camera.js` as
the template:

```js
export function createMyDeviceBox(device, container, sendCommand) {
  // `device` = the describe() dict + device_id
  // `container` = the box body <div>; fill it with plain DOM
  // `sendCommand(deviceId, name, args)` -> Promise of the command result

  // subscribe to the stream (JSON events always; JPEG frames if the adapter
  // produces them — binary messages):
  const ws = new WebSocket(`ws://${location.host}/ws/devices/${encodeURIComponent(device.device_id)}`);
  ws.onmessage = (m) => {
    if (typeof m.data === 'string') { /* JSON event: status/setting_applied/... */ }
    else { /* Blob: JPEG frame -> createImageBitmap -> canvas */ }
  };

  return function cleanup() { ws.close(); };   // MUST close what you opened
}
```

Register it in `app.js`:

```js
import { createMyDeviceBox } from './boxes/my_device.js';
const BOX_RENDERERS = { ..., my_device: createMyDeviceBox };
```

UI conventions: settings inputs commit on **Enter or focus loss** (the lab's
explicit preference), and after a `setting_applied` event the input shows the
value the hardware accepted.

## Step 4 — test without hardware first

`adapters/dummy_camera.py` exists exactly for this: a synthetic device
exercising the whole pipeline. For a new device type, either add a dummy mode
to the adapter or a second dummy adapter. Then extend `smoke_test.py` (it
boots its own server on port 8765 — no hardware, no browser needed):

    python kalishlot/smoke_test.py

Manual test: `python kalishlot/server.py`, open http://localhost:8090, click
`+ add device`. Other lab computers: http://<this-pc>:8090 (Windows Firewall
must allow inbound Python — it prompts once).

## The HTTP/WS protocol (for reference)

| Endpoint | Meaning |
|---|---|
| `GET /api/device-types` | registered types (from `DEVICE_TYPES`) |
| `GET /api/device-types/{type}/available` | adapter's `list_available()` |
| `POST /api/devices {type, address}` | open (or attach to already-open); 409 + message on failure |
| `GET /api/devices` | open devices, for browser re-attach |
| `DELETE /api/devices/{id}` | close device |
| `POST /api/devices/{id}/command {name, args}` | adapter's `command()` |
| `WS /ws/devices/{id}` | per-viewer stream: binary = JPEG frame, text = JSON event |

`device_id` is always `f'{type_name}:{address}'`.

## Roadmap context

The desktop GUI `basler_cam/basler_gui.py` is the functional reference for the
camera box (fit overlay, cross-sections, circle annotation — kalishlot phases
2–4 in the approved plan at the time of writing). Keep feature parity in mind
but never import Qt code here.
