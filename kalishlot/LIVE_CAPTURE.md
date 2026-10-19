# Live capture: measurements cut out of the running streams

Status (2026-10-19): the recorder and `alignment/live_capture.py` exist and work on the lab's
XIMEA + PicoScope 4424A. The **quick synced capture** for kalishlot's "Start Capture" (section 4)
is designed here but **not built**. This file is the handover for whoever builds it. Background
on the alignment side: `MECHANICAL_CONTROL.md`.

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

## 4. Quick synced capture (to build)
**Goal.** kalishlot's "Start Capture" writes a session that every existing tool reads unchanged
(`mode_video_sync.load_session`, `session_windows`, `mode_marking.py`, `mode_video_sync_show.py`,
`alignment/metrics.evaluate_session`), in about the capture duration, with no device borrowed.

**Where.** A second run mode of `adapters/synced_pipeline.py`, e.g. a "live" toggle next to the
existing run. Keep the current borrowed capture: it is still the way to get locate / binning /
auto-range / 100 kS/s.
- The mode runs in a thread in the server process, not through `subprocess` and `loan_client`.
- It finds the open scope and camera through `self.registry()` (the same lookup the box uses to
  list cameras and the scope).

**Steps.**
1. Check that both devices are open and streaming, and that the camera supports recording (XIMEA now;
   the Basler needs the timing hook). Refuse with a clear message otherwise.
2. `data = LiveCapture(scope_adapter.scope, camera_adapter).capture(CAPTURE_DURATION_S)`. Keep one
   `LiveCapture` per (scope, camera) pair on the adapter, so the sync correction carries over between
   captures.
3. If `data.sync['locked']` is False, still save, but say so in the box. The first capture of a stream
   is the one most likely to need the wide search.
4. Write the session in the existing format. Reuse `mode_video_capture.save_session` (frames file,
   mask, brightness series, `_json_default`) instead of re-implementing it:
   - `t` in `<stamp>_scope.npz` = `data.t - data.t[0]` (the saved trace starts at 0), plus `signal`
     and `aux`;
   - `session['meta']` = `data.meta` rows (`block_id`, `camera_timestamp_ns`, `host_time_s`,
     `exposure_us`);
   - `session['exposure_s']` = `meta[0]['exposure_us'] / 1e6`;
   - `session['sync']` = `{'method': 'live_stream', 't0_fitted_s': data.windows[0, 0] - data.t[0],
     't0_host_s': ..., 'fit_depth': ..., 'fit_locked': ...}`. Check `session_windows(prefer_fitted=True)`
     reads `t0_fitted_s` the same way as for a refined session (it must be in the saved trace's
     time coordinate);
   - `session['camera']`: the ROI as `offset_x/offset_y`, `binning_x/binning_y = 1` (this is what
     `evaluate_session` turns into sensor coordinates), plus pixel size, exposure and gain from the
     camera adapter's settings;
   - `session['scope']`: range per channel from `scope.channels`, `sample_interval_s = 1 /
     sample_rate_hz`, overflow = `data.saturated`;
   - `long_arm_m` from the box's pop-up, as today.
5. Do the writing (video encoding) in the worker thread, after the frames are taken. The streams keep
   running throughout.

**Limits to show in the box / handle:**
- Settings are whatever the boxes run at. The scope streams at its box rate (20 kHz is enough per
  the lab; up to 100 kHz offered), and the camera at its box ROI and exposure. There is no locate step
  and no auto-range.
- `CAPTURE_DURATION_S` must stay well under the ~70 s scope ring. Frames sit in RAM until written:
  0.8 MB x 113 fps = 90 MB/s at the current ROI, so the recorder's byte cap matters.
- The live stream gets the camera's newest frame (`enable_recent_frame`). Under load a frame can be
  skipped; it shows as a gap in `block_id` and the timestamps keep the timing right. Record and report
  it (`mode_video_sync.dropped_frames`).
- An ROI or exposure change during the capture truncates the recording (`Recording.truncated`).
  Changing them between captures resets the sync correction.
- The scope range must not change during a capture. A restart of its stream changes
  `stream_generation` and resets the correction.

**Tests to add:** `smoke_test.py` with the dummy camera and a fake scope (the dummy camera has no
camera timestamps, so give it a `_frame_meta` returning `camera_timestamp_ns` from its own clock).
Then a lab check: a quick capture and a borrowed capture of the same scene, both through
`evaluate_session`; peaks, frame windows and metrics should agree.
