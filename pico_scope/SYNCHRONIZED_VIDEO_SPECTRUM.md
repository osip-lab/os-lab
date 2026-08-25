# Synchronized mode video + cavity spectrum

**Status:** design, not yet implemented. Written 2026-08-25.

**How to use this document:** Part 1 is for a human at the lab computer — checks
that must be made at the hardware before any code is worth writing, because two
of them can invalidate the design. Part 2 is the implementation brief for a
Claude Code session on the lab machine; it assumes Part 1 has been done and its
answers are written down.

---

## The problem

A measurement is currently two separate recordings: a mode video from the Pylon
Viewer, then a spectrum from PicoScope 7. When coupling is good the spectrum
alone is readable — one 0th order, one 1st order, fit them, done. When coupling
is bad the peak identity is ambiguous (which peak is 0th, which is 1st, which are
higher order), and the video is consulted for relative intensities and spacing.
Because the two recordings were made at different times, that consultation is
guesswork.

The goal: record both at once, so that any point on the spectrum maps to a
definite video frame. Hover a peak, see the transverse pattern that produced it.

## The idea

**Do not synchronize the two clocks.** Wire the camera's `ExposureActive` output
line into a spare PicoScope channel. Every frame's exposure window then appears
as a pulse *in the same record as the spectrum*, in the scope's own timebase.

Consequences worth understanding, because they are why this approach was chosen
over the alternatives:

- No clock drift to model and no host/USB latency term. The alignment is
  **measured**, not inferred.
- The alignment is stored permanently in the data file — a `.psdata` from three
  months ago can still be re-aligned exactly.
- **The frame-0 anchor is free.** The camera is off before the burst starts, so
  the first rising edge in the record is unambiguously frame 0. No cross-device
  arming is needed, which is why Phase 1 below needs no scope code at all.

### Why not the other routes

- **Driving the native apps: impossible.** PicoScope 7's only external interface
  is the `BatchConvert` command line already used in `utilities/utils.py`
  (`psdata_to_csv`); there is no arm/trigger/record API. The Pylon Viewer has no
  scripting interface. Neither can be started at a known instant.
- **Software timestamps only (no cable): not good enough.** Alignment would be
  limited by USB and OS latency — milliseconds of jitter, comparable to the peak
  spacing being resolved.
- **Kalishlot: right building blocks, wrong host today.** It has both adapters
  (`kalishlot/adapters/basler.py`, `adapters/picoscope.py`) and is the only place
  in the repo where a camera and a scope are open simultaneously. But nothing in
  it is time-correlated: `DeviceAdapter._store_display_frame` records a frame
  *counter*, not a clock; scope events carry a duration, not an absolute time; an
  analysis is constructed with exactly one host adapter and cannot see a second
  device; and the browser only receives JPEG-compressed, 2×-downsampled 8-bit
  frames — full-resolution data never leaves the server process. Doing this in
  kalishlot means inventing a cross-device "session" concept in `server.py`.
  Worth revisiting later, once Phase 1 has shown what the data needs to look like.

### Decisions already taken

| Question | Answer |
| --- | --- |
| Who drives the capture | A Python script (pypylon + picosdk are already in `requirements.txt`) |
| Sync method | Camera `ExposureActive` → spare scope channel |
| Frame timing | Frame period ≈ 10–30 ms, exposure = frame period (no dead time) |

---

# Part 1 — Checks to make at the lab, before any code

Write the answers down; Part 2 needs them. Checks 1 and 2 can invalidate the
design, so do those first.

### 1. Does the camera expose `ExposureActive`? *(can invalidate the design)*

Some newer ace 2 / dart models dropped `ExposureActive` as a `LineSource`.

In the Pylon Viewer, with the camera open, find the **Digital I/O Control**
category and check:

- Which output lines exist (`Line2` is the usual opto-isolated output;
  `Line3`/`Line4` are GPIO on models that have them).
- Set `Line Selector` to that line, `Line Mode` = `Output`, and open the
  `Line Source` dropdown. **Is `ExposureActive` in the list?**

Record: camera **model name**, **serial number**, which **line** you will use, and
**whether `ExposureActive` is available**.

If it is *not* available, look for `Timer1` in the same dropdown, plus a
**Timer Control** category with `Timer Trigger Source` = `ExposureStart` and a
`Timer Duration`. That combination is the standard substitute and produces the
same pulse train. Note which of the two you have.

### 2. Is a scope channel free? *(can invalidate the design)*

The cavity transmission is on **Channel D** (`SIGNAL_COLUMN = 'Channel D'` in
`pico_scope/mode_map_2d.py`). The sync pulse train needs its own channel.

Record: which channel is free (A, B or C), and how many channels your unit has.
Also note whether anything else is currently occupying A/B/C in your normal
measurement.

### 3. Cabling

Work out the physical connection from the camera's I/O connector to a scope BNC.
Record what connector the camera uses and what adapter/cable is needed.

Note for the opto-isolated output (`Line2` on ace cameras): it is an open
collector and needs a **pull-up resistor to a supply** (typically a few kΩ to
3.3 V or 5 V) to produce a voltage swing. The GPIO lines (`Line3`/`Line4`) can
usually drive a level directly. Check which type your chosen line is.

### 4. See the pulse train on the scope — the real proof that this works

Configure the line as in check 1, set the camera free-running at ~50 fps with
exposure ≈ 20 ms, and look at the sync channel on the PicoScope screen.

You should see a square wave with:

- **period** = the frame period (≈ 20 ms at 50 fps),
- **high time** = the exposure time.

Record: the **voltage levels** (low and high), so the right scope range and
trigger threshold can be chosen, and confirm period and high time match the
camera settings. **If this square wave is not there, stop — nothing downstream
works, and it is a wiring or `LineSource` problem, not a software one.**

### 5. Frame rate vs bandwidth — four independent levers

`BaslerCamera.THROUGHPUT_LIMIT_BPS = 150_000_000` (`basler_cam/basler_cameras.py:39`)
against a 2048×2048 sensor means full-frame Mono12 is ~8.4 MB/frame, i.e.
**≤17 fps** — short of the 33–100 fps that a 10–30 ms frame period needs. Payload
per frame has to come down. There are four ways to do it, and they multiply.

Two things can be the actual bottleneck, and they respond differently:
the **link** (payload bytes per second, capped by `DeviceLinkThroughputLimit`)
and the **sensor readout** (row-by-row, so it scales with the number of *rows*).
Reducing height helps both; reducing width mostly helps the link.

| Lever | Node(s) | Gain | Cost |
| --- | --- | --- | --- |
| **ROI crop** | `Width`, `Height`, `OffsetX`, `OffsetY` | area ratio | mode must stay inside it |
| **Binning 2×2** | `BinningHorizontal`, `BinningVertical`, `BinningHorizontalMode`, `BinningVerticalMode` | **4×** | half the spatial resolution |
| **Mono8 instead of Mono12** | `PixelFormat` | ~2× | 8-bit levels |
| **Raise the throughput cap** | `DeviceLinkThroughputLimit` | up to ~2.5× | only safe with one camera on the bus |

Notes that matter for this measurement:

- **Binning must be done on the camera, not on the host.** `BinningHorizontal`/
  `BinningVertical` are applied inside the camera before the data reaches the
  link, which is what makes them reduce bandwidth. Rebinning a full-size frame
  after it arrives saves nothing.
- **Set `BinningMode = 'Sum'`, not `'Average'`.** Sum gives 4× the signal for 2×2.
  The badly-coupled measurements this whole feature exists for are also the dim
  ones, so binning helps the images, not just the data rate. Watch for saturation
  on a bright 0th order — if it clips, either drop to `'Average'` or shorten
  exposure, and note which you chose.
- **Losing spatial resolution is cheap here.** The task is telling a round spot
  from a two-lobed pattern, not measuring a waist. 2×2, possibly 4×4, is fine.
- **Prefer binning to decimation.** `DecimationHorizontal`/`DecimationVertical`
  cut bandwidth by the same factor but skip pixels instead of summing them —
  same bandwidth saving, less light, no SNR benefit.
- **The 150 MB/s cap is conservative on purpose.** The comment at
  `basler_cameras.py:78-84` explains it: it was chosen so *two* cameras could share
  the USB3 bus without discarding frames. If only one camera is connected for these
  captures, it can be raised — but confirm nothing else is on the bus first.

At the camera, work out the smallest ROI that comfortably contains the mode spot
across the whole sweep — including the higher-order patterns, which are larger
than the 0th order, so do **not** crop to the TEM00 spot.

Record: the **ROI width and height**; whether the mode ever moves outside it during
a sweep; the **binning factor** you settled on and whether `Sum` saturated; whether
**Mono8** is acceptable; and whether **only one camera** is on the USB bus.

### 6. Sweep timing — how many frames you actually get

This decides whether the whole idea gives useful images or blended mush.

With exposure = frame period, each frame is an average over 10–30 ms. That is
what guarantees no peak is missed, but **two resonances crossed within one frame
blend into a single image**. So the frame period must be shorter than the time
between adjacent peaks.

From a normal spectrum recording, measure:

- the **total sweep duration** (one full ramp, in ms),
- the **time between the 0th-order peak and the 1st-order peak** of the same FSR
  (in ms) — the tightest spacing that must stay resolved,
- the **time between consecutive 0th-order peaks** (one FSR, in ms).

Record all three. If the 0th→1st spacing turns out to be shorter than ~10 ms,
say so — the frame rate has to go up and the ROI has to shrink further, and it
is better to know that before the code is written.

### 7. Bench-test parts (for the end-to-end check in Part 2)

The verification below drives an LED from the Rigol generator and points the
camera at it. Confirm you have an LED that can be driven from the generator and
a spare scope channel to record the same drive signal.

---

# Part 2 — Implementation brief

Everything below is new work. It is split so that Phase 1 delivers the full
capability while leaving the scope side in PicoScope 7 — so every existing loader
and analysis script keeps working untouched.

**Order of operations during a capture:** start the PicoScope 7 recording, *then*
start the camera burst from Python. The camera being off beforehand is what makes
the first rising edge frame 0.

## Phase 1a — extend `basler_cam/basler_cameras.py`

`BaslerCamera` already handles open/close/exposure/gain cleanly. Add:

- `frame_rate_hz` property (get/set `AcquisitionFrameRate`) and a
  `resulting_frame_rate` read-back (`ResultingFrameRate`). Today
  `MAX_FRAME_RATE = 10.0` is written unconditionally at open
  (`basler_cameras.py:90`) — make it the default of a settable property.
- `set_roi(width, height, offset_x=None, offset_y=None)`, centred by default.
- `set_binning(factor_x, factor_y=None, mode='Sum')` — `BinningHorizontal` /
  `BinningVertical` plus `BinningHorizontalMode` / `BinningVerticalMode`.
  **Apply binning before the ROI**: binning changes what one pixel means, so
  `Width`/`Height`/`OffsetX`/`OffsetY` are expressed in binned pixels and the
  usable maximum shrinks by the binning factor. Setting them in the other order
  silently gives the wrong field of view. Read `Width`/`Height` back after
  applying both and record the true frame shape in the session file — the viewer
  and any later fit need to know the effective pixel size, which is
  `PIXEL_SIZE_MM × binning` (see `kalishlot/adapters/basler.py`,
  `PIXEL_SIZE_MM = 5.5/1000.0`).
- `set_pixel_format(fmt)` — `Mono8` roughly halves the payload versus `Mono12`
  and is plenty for mode identification. `open()` currently hard-codes
  `pixel_format='Mono12'` (`basler_cameras.py:60`); keep that default, make it
  changeable afterwards. If `Mono8` is used, `record_burst` should return uint8
  and the session file should say so, rather than everything assuming uint16.
- `set_throughput_limit(bytes_per_second)` — so the deliberately conservative
  `THROUGHPUT_LIMIT_BPS = 150_000_000` can be raised when only one camera is on
  the bus. Keep the existing value as the default and leave the explaining
  comment at `basler_cameras.py:78-84` intact.
- `set_exposure_active_output(line='Line2')` — `LineSelector` / `LineMode='Output'`
  / `LineSource='ExposureActive'`, with the `Timer1` + `TimerTriggerSource=
  'ExposureStart'` fallback if check 1 said so.
- `enable_chunks()` — `ChunkModeActive`, `ChunkSelector='Timestamp'` +
  `ChunkEnable`, so each grab carries `result.TimeStamp` and `result.BlockID`.
- `record_burst(n_frames)` → `(frames, meta)`, where `meta` has one row per frame:
  `block_id`, `camera_timestamp_ns`, `host_time_s`.
  **Must use `pylon.GrabStrategy_OneByOne`.** The existing `start_streaming()`
  uses `GrabStrategy_LatestImageOnly` (`basler_cameras.py:161`), which silently
  discards frames — fatal here.
- `max_frame_rate_for(roi, pixel_format, binning=1)` — applies the throughput cap
  to the *effective* payload (ROI ÷ binning², times bytes per pixel) and raises a
  clear error when the requested rate is unreachable, rather than letting the
  camera quietly drop frames. Cross-check the estimate against
  `ResultingFrameRate` read back from the camera after configuring: that is the
  camera's own answer and it also accounts for readout time, which the payload
  arithmetic alone does not.

Leave `CameraStreamer` and the existing `self_test()` alone — kalishlot uses them.

## Phase 1b — new `pico_scope/mode_video_capture.py`

Configuration block at the top in the established style (see the knobs block of
`pico_scope/mode_map_2d.py:70-130`): serial number, ROI, frame rate, exposure,
frame count, output folder.

Flow: open camera → apply ROI, exposure = frame period, frame rate, exposure-active
line, chunks → print the bandwidth and peak-blending checks (from Part 1 checks 5
and 6) → prompt *"start the PicoScope recording now, then press Enter"* →
`record_burst` → save.

Saves a session folder:

- `<stem>_frames.npy` — uint16 stack (100 frames of 512×512 ≈ 52 MB).
- `<stem>_session.json` — camera settings, ROI, the per-frame `meta` rows, the
  sync channel name, and the computed checks.

The scope side stays a `.psdata` exported from PicoScope 7, exactly as today.

## Phase 1c — new `pico_scope/mode_video_sync.py` (pure, testable)

No hardware, no GUI — this is the piece that earns a self-test.

- `frame_windows_from_sync(t, sync_volts, threshold=None)` → rising/falling edges
  → `[(t_start, t_end), ...]`. Threshold defaults to the midpoint of the observed
  high and low levels.
- `align_frames(windows, frame_meta)` → index mapping. Assert
  `len(windows) == len(frame_meta)`; when they differ, locate the drop by
  comparing edge periods against `camera_timestamp_ns` differences and raise a
  message naming the frame, rather than silently mis-mapping.
- `frame_at_time(mapping, t)` → the frame whose exposure window contains `t`.

Read the scope CSV through the existing helpers: `utilities.utils.psdata_to_csv` /
`psdata_buffer_csvs` / `choose_buffer_csv`, and the
`pd.read_csv(..., skiprows=[1, 2])` convention in `pico_scope/mode_map_2d.py:182`.

One gotcha to document rather than "fix": PicoScope's `Time` column units row is
discarded by the current loaders, and nothing in the repo pins where `t = 0` sits.
Because the spectrum and the sync pulses come from the *same* record, the origin
cancels. Say so in the docstring.

## Phase 1d — new `pico_scope/mode_video_sync_show.py` (the viewer)

Reuse, don't reinvent:

- Layout and cursor idiom from `utilities/media_tools/plot_video.py:146-205`
  (`VideoInspector`: image axes + full-width trace axes + an `axvline` marking the
  displayed frame).
- Hover mechanics from `utilities/media_tools/postprocessing_camera_video.py:310`
  — the `motion_notify_event` + `copy_from_bbox` / `restore_region` / `blit`
  template. It is the only live-hover pattern in the repo and is fast enough.

Behaviour: spectrum across the top, mode image beside it; moving the cursor over
the spectrum shows the frame covering that instant; frame boundaries drawn as
light shading so the time granularity is visible; click pins a frame; arrow keys
step frame by frame.

Follow the repo convention of a `_self_test()` run under `matplotlib.use('Agg')`
(see `pico_scope/mode_marking.py`).

## Phase 2 — one-keypress capture (optional, only after Phase 1 proves out)

Move the scope side into Python so nothing is started by hand. Extend
`pico_scope/ps4000a_scope.py`, which is streaming-only today with no trigger:

- `configure_trigger(source, threshold_v, direction, delay, auto_trigger_ms)` →
  `ps4000aSetSimpleTrigger`.
- `capture_block(duration_s, pre_fraction, sample_interval_s)` → `GetTimebase2` /
  `RunBlock` / `IsReady` poll / `SetDataBuffers` / `GetValues`. There is a legacy
  block-mode implementation to copy from in
  `pico_scope/pico_scope_control_gui.py` (`ps_load` / `ps_start`; the trigger
  tuple order is built at line 730).
- **Return `t` with zero at the trigger** (`t = (arange(n) - pre_samples) * dt`).
  The legacy `pico_scope/__init__.py:adc2mv` puts zero at the first sample
  instead — a trap worth not repeating.

Leave `start_streaming` / `read_window` untouched: kalishlot depends on them.

---

# Verification

### Offline, no hardware

- `python pico_scope/mode_video_sync.py --self-test` — synthesize a pulse train
  plus frame metadata (including a deliberately dropped frame) and assert the
  mapping, the drop detection, and `frame_at_time` at window edges.
- `python pico_scope/mode_video_sync_show.py --self-test` — build the figure under
  Agg and drive the hover handler with synthetic events.
- `python basler_cam/basler_cameras.py` — the existing connectivity self-test must
  still pass.

### On the bench — the test that validates the whole chain

Drive an LED from the Rigol generator (already wired into this repo via
`rigol_gen`), point the camera at it, and feed the same drive signal into another
scope channel. Capture.

**The frames that come out bright must be exactly the frames whose exposure
windows overlap the LED-on intervals.** If they are, the mapping is correct. If it
is off by one, this shows it immediately and unambiguously — which a cavity
measurement would not.

### Asserted on every real capture

- number of rising edges == number of recorded frames;
- measured pulse width ≈ `ExposureTime` read back from the camera;
- measured pulse period ≈ 1 / `ResultingFrameRate`.

### On a real measurement

Take one deliberately bad-coupling spectrum, hover the ambiguous peaks, and
confirm the frames show the transverse patterns expected — a clean spot on the
0th order, a two-lobed pattern on the 1st.
