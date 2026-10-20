# Live capture: measurements cut out of the running streams

Status (2026-10-20): built. The recorder and `alignment/live_capture.py` run the alignment
measurements (`adapters/live_measure.py`) and the **quick synced capture** of the synced pipeline
box (section 4). Tested on simulated devices and, with the laser off, on the lab's XIMEA +
PicoScope 4424A (the sync refuses without blinks, as it should). **Still to do in the lab:** a sync
with the laser on, and one quick capture compared with a borrowed one (end of section 4).
Background on the alignment side: `MECHANICAL_CONTROL.md`.

## 1. The problem it removes
Today a synced video + spectrum capture (`adapters/synced_pipeline.py` ->
`pico_scope/mode_video_capture.capture_synchronized`) works like this:
1. kalishlot **lends** the camera and the scope to a child process (`loan_client.py`); their boxes
   stop streaming.
2. The child reconfigures both: locate an ROI, binning, pixel format, frame rate, scope auto-range.
3. It records a camera burst (FIFO, every frame) and a scope block (100 kS/s).
4. It writes the session to disk, and `mode_video_sync.py` fits the time offset afterwards.
5. The devices go back to kalishlot and restart streaming.

This takes many seconds every time, and the live view is gone meanwhile. The alignment loop needs a
measurement every few seconds, and a person often just wants "save what I see now".

## 2. The mechanism (built)
While kalishlot runs, the scope and camera stream continuously anyway. A measurement takes what it
needs from those streams. Nothing is borrowed or reconfigured.

| Piece | Where | What it does |
|---|---|---|
| Frame timing in the live stream | `ximea_cam/ximea_cameras.py` `XimeaCamera.get_frame()` | leaves `last_frame_meta` = `{block_id, camera_timestamp_ns, host_time_s}` for the frame it returned (same keys as a `record_burst` meta row); `get_frame()`'s return value is unchanged |
| **Recording flag** | `kalishlot/adapters/camera_base.py` `CameraAdapterBase.start_recording(max_s, max_bytes)` / `stop_recording() -> Recording` | while armed, `_store_camera_frame` also appends every frame (after the ROI crop, before the display trigger, so dark frames and blinks are kept) with `_frame_meta()`. Off by default: zero memory and zero cost. Caps: 10 s / 1 GB by default. Stops early with `Recording.truncated` set if the ROI changes. The stream is never paused |
| Per-camera timing hook | `adapters/ximea.py` `_frame_meta()` | `last_frame_meta` + `exposure_us`. Other cameras get only `host_time_s` (base class); the Basler would need chunk timestamps enabled (`basler_cameras.enable_timestamps`) and the same hook |
| Scope history | `pico_scope/ps4000a_scope.py` ring buffers | always the last ~70 s per channel (cap 8 M samples, 16 MB), addressed by absolute sample index (`read_span`, `samples_written`). Needs no flag |
| Scope host clock | `PicoScope4000A.sample_at_host_time(t)` | (host time, samples written) at every driver callback, last ~10 s; returns the sample index taken at host time `t` (min-latency estimate) |
| Placing frames on the scope timebase | `kalishlot/alignment/live_capture.py` `LiveCapture(scope, camera).capture(duration_s)` | arm, wait, disarm; host clocks give frame 0's time; `mode_video_sync.fit_time_offset` refines it optically (+-0.1 s the first time, then +-15 ms around the kept correction); returns `LiveCaptureData` (`t, signal, aux, frames, windows, meta, roi, sync, saturated`) |
| Tests | `kalishlot/alignment/test_live_capture.py` | recorder on the synthetic camera adapter; LiveCapture on a fake scope + camera with a known clock error |

`LiveCapture` takes the **device-layer** scope (`PicoScopeAdapter.scope`, a `PicoScope4000A`) and
the **camera adapter** (it needs `start_recording`/`stop_recording`). So it must run in the kalishlot
server process, not in a child process.

## 3. Measured (2026-10-19)
XIMEA s/n QXNAA2506000, ROI 720x546 Mono10 (0.79 MB/frame), 8.8 ms exposure, 113 fps; scope B (ramp)
and D (transmission) at 20 kHz.
- 2 s recording: 227 frames, no gap in the block ids, host-vs-camera latency spread 1.6 ms.
- Host-clock placement vs the optical fit: +1.50 +- 0.22 ms over 12 captures in 3 min, no trend; fit
  locked every time (depth 28-58, locked > 1.5).
- The scope clock drifts -13 ppm against the host clock. Every capture re-ties both clocks to the host
  and is re-fitted, so the drift never accumulates.
- Scope callbacks arrive in bursts (~60 ms spread); the min-latency estimate is unaffected.
- One 3 s capture costs 3.7 s of wall time (record + scope span + fit).

## 4. Quick synced capture (built)
**In the box.** The synced pipeline box has two ways to capture:
- **sync camera and scope ⟲**: records 3 s from the running streams (`SYNC_DURATION_S`) and fits
  the camera-to-scope clock correction on the mode blinks (+-0.1 s search). It needs blinks: laser
  on, scan running, cavity transmitting. Without them it refuses ("did not lock") and keeps any
  earlier sync.
- **start capture ▶**: greyed out until a sync exists. Asks the long arm and folder label (the same
  pop-up as before), records `measurement length` from the running streams, places the frames with
  the **kept** correction, and saves a session. Nothing is lent; the live views keep running.
- **forget sync**: drops it by hand.
- **full capture (borrow devices) ▶**: the old pipeline run, unchanged. It is still the way to get
  locate / binning / auto-range / 100 kS/s / the trailing scope capture.

The status line shows the sync (`+1.52 ms, depth 41, 12 min ago`), the last capture's check, the
longest capture memory allows now, or why there is no sync ("cleared: the camera exposure changed").

**The sync** (`alignment/live_capture.py`):
- `ClockSync`: the correction (fitted minus host-clock placement of frame 0), the stream key it
  holds for (scope `stream_generation`, camera ROI, exposure), depth, when set, why cleared.
- `shared_sync(scope_id, scope, camera_id, camera)`: one per pair of open devices, **shared** by
  the quick capture and `live_measure` (an alignment measurement refreshes it too: method
  `tracked`). Another device object under the same id = the box was reopened = a fresh, empty
  sync ("the camera box was reopened").
- Cleared when: a box is reopened (device off/disconnected means reopening it), the scope restarts
  its stream (rate or range change), the camera ROI or exposure changes, or by hand. Kept in memory
  only, so a server restart clears it too.
- `LiveCapture.capture(placement=...)`: `'track'` (alignment: fit +-15 ms and keep), `'establish'`
  (the button: fit +-0.1 s, `SyncError` unless locked), `'kept'` (the quick capture: place with the
  kept correction; a +-15 ms fit only **checks** it). The check is saved, and a check more than 1 ms
  off (`CHECK_WARN_S`) shows "re-sync recommended". A capture with no blinks at all is placed by the
  kept correction like any other.
- The camera records for one capture at a time (`start_recording` refuses a second), so an
  alignment measurement and a quick capture cannot steal each other's frames.

**The session** (`pico_scope/mode_video_capture.save_live_session`): the same files and keys as a
borrowed capture, written through `save_session` (which now takes its settings as an argument, so
the server never rewrites that module's constants). Every reader takes it unchanged.
- `<stamp>_scope.npz`: `t` starts at 0 (`session['scope']['stream_t_origin_s']` = where in the
  scope stream), `signal`, `aux`.
- `sync.method = 'live_stream'`; `t0_host_s` = host clocks alone; `t0_fitted_s` = with the kept
  correction (what `session_windows` uses); `check_locked` / `check_residual_s` / `check_depth`.
  `refine_session` still works on it (it searches around `t0_host_s`).
- `camera` = `XimeaCamera.describe()`, read in the streaming thread (`XimeaCameraAdapter
  .hardware_info()`); `function_generator` from the open Rigol box; `live` = duration, channels,
  scope rate and the sync record used.
- No `run_config` dump: it describes a script run, which this is not.

**What the box parameters mean for it** (`LIVE_PARAMS`): measurement length, frames compression,
h264 quality, mask threshold, scope transmission/aux channels and label, scope padding. ROI,
exposure, gain, frame rate and scope range are whatever the camera and scope boxes run at.

**Length limit.** Frames sit raw in memory until the capture is written, exactly as in the borrowed
capture (`record_burst` allocates the whole burst, then `save_frames` encodes it), so compression
only changes the file size, not the possible length. The quick capture may use half the free memory
(`RECORD_RAM_FRACTION`) and at most 60 s (the scope ring holds ~70 s): ~19 s at 720x546 Mono10,
113 fps with ~3.5 GB free. Longer needs a narrower ROI or real-time encoding (not built: h264 would
need its brightness scale chosen before the capture, and 113 fps encoding inside the server competes
with the live streams).

**Tests:** `alignment/test_live_capture.py`: establish / kept / no-blink placement, the clearing
reasons, a reopened box, and a sync + quick capture through the box adapter on a fake scope and
camera, read back with `load_session` / `session_windows`.

**Still to check in the lab (laser on):**
1. Sync: depth, and the correction (~+1.5 ms on 2026-10-19).
2. A quick capture and a borrowed capture of the same scene, both through `evaluate_session`;
   peaks, frame windows and metrics should agree.
3. A quick capture an hour after the sync: the check residual says whether the correction drifts.
4. The Basler cannot do this yet: it needs chunk timestamps in its stream and a `_frame_meta` hook.
