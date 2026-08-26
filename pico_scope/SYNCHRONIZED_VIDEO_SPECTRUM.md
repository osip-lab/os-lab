# Synchronized mode video + cavity spectrum

**Status:** design, not yet implemented. Written 2026-08-25.
**Part 1 was carried out on 2026-08-26** — see *Part 1 — results* near the end of
this file. The approach is confirmed viable. The sync cable of check 3 does not
exist in the lab, so the `ExposureActive` wire was **replaced by an optical
correlation** that fits the offset from the data itself — see *Synchronization
method*. Nothing is blocked. Several premises in Part 1 below turned out to be
different at the hardware; the results section says which, and supersedes them.

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

> **Answered 2026-08-26: yes.** See *Part 1 — results*.

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

> **Answered 2026-08-26: Channel A.** See *Part 1 — results*.

The cavity transmission is on **Channel D** (`SIGNAL_COLUMN = 'Channel D'` in
`pico_scope/mode_map_2d.py`). The sync pulse train needs its own channel.

Record: which channel is free (A, B or C), and how many channels your unit has.
Also note whether anything else is currently occupying A/B/C in your normal
measurement.

### 3. Cabling

> **Answered 2026-08-26: no cable exists in the lab, and the design no longer
> needs one** — see *Synchronization method*. Kept because a cable would still be a
> useful independent check. The pull-up advice below is superseded: the line that
> would be used (Line3, GPIO) has an internal ≈2 kΩ pull-up and needs no external one.

Work out the physical connection from the camera's I/O connector to a scope BNC.
Record what connector the camera uses and what adapter/cable is needed.

Note for the opto-isolated output (`Line2` on ace cameras): it is an open
collector and needs a **pull-up resistor to a supply** (typically a few kΩ to
3.3 V or 5 V) to produce a voltage swing. The GPIO lines (`Line3`/`Line4`) can
usually drive a level directly. Check which type your chosen line is.

### 4. See the pulse train on the scope — the real proof that this works

> **Not done, and no longer required** — the optical correlation replaces the pulse
> train. Still the right first test if a cable is ever obtained.

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

> **Answered 2026-08-26, and the premise below is wrong:** the cameras are at
> 212 MB/s, not 150 MB/s, and reach 50.6 fps at full frame Mono8. Bandwidth is
> not the constraint. `BinningHorizontalMode` / `BinningVerticalMode` do not
> exist on this model. See *Part 1 — results*.

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

> **Answered 2026-08-26: 0th→1st spacing is 24.7–116.8 ms (median 34.3), FSR
> 179.6–286.5 ms, record 1.000 s.** The frame period must therefore be ~10 ms,
> the short end of the range assumed here. See *Part 1 — results*.

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

> **Superseded 2026-08-26.** The LED bench test was there to prove the electrical
> mapping. Its replacement is the beam-block marker of option B, which needs no parts.

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

---

# Part 1 — results (measured 2026-08-26, lab PC `OsipLab`)

Checks 1, 2, 5 and 6 are done. Check 3 is blocked (no I/O cable exists), and
check 4 is blocked behind it.

## Camera — check 1: `ExposureActive` is available ✅

Two `acA2040-90umNIR` on the USB3 bus, s/n **24916690** and **25173136**;
**25173136 is the cavity-mode camera**. Both report identical capabilities.
Probed with pypylon, not the Pylon Viewer.

| Line | Pin | Format | Modes | `ExposureActive` |
| --- | --- | --- | --- | --- |
| Line1 | 2 | OptoCoupled | Input only | – |
| Line2 | 4 | OptoCoupled | Output | yes |
| **Line3** | **1** | **TTL / GPIO** | Input, **Output** | **yes** |
| **Line4** | **3** | **TTL / GPIO** | Input, **Output** | **yes** |

`LineSource` options on Line2/3/4: `ExposureActive`, `FrameTriggerWait`,
`FrameBurstTriggerWait`, `Timer1Active`, `UserOutput*`. The `Timer1` fallback is
present too (`TimerSelector = ['Timer1']`, `TimerTriggerSource = ['ExposureStart']`)
but is not needed.

**Use Line3 or Line4, not Line2.** The GPIO output is open collector with an
**internal ≈2 kΩ pull-up**, so into the PicoScope's 1 MΩ input it swings on its own
— no external pull-up or supply, unlike the opto-isolated Line2. Levels from the
Basler spec: "on" residual ≈0.4 V, safe operating range 3.3–24 V, max load 50 mA.

`ChunkSelector` offers `Timestamp`, `LineStatusAll`, `CounterValue`, `ExposureTime`
— `enable_chunks()` in Phase 1a works as designed.

⚠ **Deviation from the plan:** `BinningHorizontalMode` / `BinningVerticalMode`
**do not exist** on this model (`LogicalErrorException: Node not existing`).
`BinningHorizontal` / `BinningVertical` do, range 1–4. So `Sum` vs `Average` is
not selectable — the mode is fixed in firmware and must be determined empirically
(same scene at binning 1 vs 2; mean level ×4 means Sum, unchanged means Average).
`set_binning(..., mode=...)` must tolerate the node's absence.

## Camera — check 3: connector and cabling ⚠ no cable (worked around)

- Camera receptacle: **Hirose HR10A-7R-6PB**; mating plug: **Hirose HR10A-7P-6S**.
- Pinout (from the acA2040-90umNIR page of the pylon 8 docs):

  | Pin | Line | Function |
  | --- | --- | --- |
  | 1 | Line 3 | GPIO |
  | 2 | Line 1 | opto-coupled input |
  | 3 | Line 4 | GPIO |
  | 4 | Line 2 | opto-coupled output |
  | 5 | – | ground for the **opto-coupled** lines |
  | 6 | – | ground for the **GPIO** lines |

- Wiring wanted: **pin 1 (Line 3) → BNC centre, pin 6 (GPIO ground) → BNC shield.**
  Pin 5 is the *opto* ground and must not be used with a GPIO line.
- **No such cable exists in the lab.** One must be sourced before check 4.
- Basler's warning, worth obeying: configure `LineMode` to `Output` in software
  *before* connecting anything to a GPIO line. They are inputs by default.

## Scope — check 2: Channel A is free ✅

PicoScope 4424A. Cavity transmission stays on **Channel D**; the sync pulse train
goes on **Channel A**. Confirmed against yesterday's recordings, where Channel A is
already enabled but sits at −25 mV ± 5 mV — unconnected-input noise, nothing on it.

## Bandwidth — check 5 ✅ (not a constraint)

Measured on the camera, not estimated: `DeviceLinkThroughputLimit` is already
**212 MB/s** (max settable 419 MB/s, link speed 500 MB/s), and `ResultingFrameRate`
reads back **50.6 fps at full frame 2048×2048 Mono8**.

| Config | Payload/frame | fps at 212 MB/s |
| --- | --- | --- |
| 2048², Mono12 | 8.4 MB | 25 |
| 2048², Mono8 | 4.2 MB | 51 |
| 1024² (2×2 binned), Mono8 | 1.05 MB | 202 on the link — but see below |

⚠ **That last row is misleading, and the plan's whole bandwidth model with it.**
Measured while implementing Phase 1a (2026-08-26): the link is *not* usually the
constraint, and **binning does not raise the frame rate at all** on this sensor.
See *Frame rate — what actually limits it* below.

Exposure range 42 µs – 10 s, so a 10 ms exposure is well inside. Pixel formats:
`Mono8`, `Mono12`, `Mono12p`.

**Two cameras are on the USB bus**, so the `THROUGHPUT_LIMIT_BPS` comment at
`basler_cameras.py:78-84` still applies and the cap should not be raised blindly.
It does not need to be: binning gets us past 100 fps with room to spare.

## Sweep timing — check 6 ✅ and it constrains the frame rate

From the 15 `.modemarks.json` sidecars of 2026-08-23 (33–48 cm), whose `x0` values
are peak centres in the CSV `Time` column, plus one reconverted CSV to pin the
units — the units row reads `(s)`, so these are **seconds**.

| Quantity | min | median | max |
| --- | --- | --- | --- |
| **0th→1st spacing** | **24.7 ms** | 34.3 ms | 116.8 ms |
| FSR (0th→0th) | 179.6 ms | 221.8 ms | 286.5 ms |
| 0th-order FWHM | 0.69 ms | 1.53 ms | 3.47 ms |
| 1st-order FWHM | 0.63 ms | 2.14 ms | 6.26 ms |

Record length 1.000 s, 100 004 samples, dt = 10 µs.

The 0th→1st spacing shrinks with arm length: ~117 ms at 33 cm down to **~25 ms at
44–45 cm**, widening again by 47–48 cm.

**Consequence for the design:** the frame period must sit at the *short* end of the
plan's 10–30 ms range, not the middle. A 30 ms frame would blend the closest 0th
and 1st orders into one image. Target **10 ms / 100 fps**, which puts ~2.5 frames
between the tightest pair and 100 frames across a 1 s record.

That needs 2×2 binning (1024×1024, Mono8, 1 MB/frame, 100 MB/s). Note also that a
1.5 ms FWHM peak inside a 10 ms exposure means the resonance lights only ~15% of the
integration window, so frames will be dim — check the empirical binning mode and be
ready to raise `Gain`.

## Check 7 — bench-test LED

Superseded. The LED existed to validate the electrical frame-to-pulse mapping; with
the optical method the equivalent end-to-end test is the beam-block marker of
option B, which needs no parts at all.

## Decisions taken as a result of Part 1

These supersede the corresponding rows of *Decisions already taken* and the
Phase 1a/1b text above.

| Question | Decision | Because |
| --- | --- | --- |
| Which camera | s/n **25173136** | it is the one pointed at the cavity mode |
| Which line | **Line3** (GPIO, pin 1), ground on pin 6 | internal ≈2 kΩ pull-up drives a 1 MΩ scope input unaided; Line2 (opto) would need an external pull-up and supply |
| Which scope channel | **Channel A** | free and confirmed empty in real recordings; Channel D stays the signal |
| Frame period / rate | **10 ms / ~99.6 fps**, exposure 9.9 ms | the tightest measured 0th→1st spacing is 24.7 ms, so the plan's 30 ms end of the range would blend adjacent peaks. Exposure must sit just *under* the period, or it becomes the cap itself |
| Pixel format | **Mono8** | halves the payload; 8 bits is ample for telling a spot from two lobes |
| Binning | **2×2**, firmware mode measured to be **Sum** | 4× the signal, which the dim badly-coupled measurements need. It does *not* buy frame rate — see below |
| ROI | **1024×512 binned = 2048×1024 sensor** | full sensor width, half its height. Height is the only dimension that costs frame rate |
| Throughput cap | **raise to 212 MB/s** for the capture | the class default of 150 MB/s caps this configuration at 74.7 Hz. 212 MB/s is the cameras' own factory value, and only one camera streams during a capture |
| Frames per capture | **~100**, covering the 1.000 s record | 1024×512 uint8 → 0.52 MB/frame, ~52 MB per session |

## Synchronization method — decided 2026-08-26: optical correlation, no cable

**The cable is no longer on the critical path.** The offset between the two
records is recovered from the data itself. This supersedes the "software
timestamps only: not good enough" rejection in *Why not the other routes* — that
argument was about *host* timestamps, and it is still correct about those. The
route below is a different thing and was not considered when the plan was written.

### The insight

Without a wire, the only observable the camera and the scope share is **light** —
every other channel (USB, host clock, PicoScope 7's start instant) carries
millisecond latency, and PicoScope 7's record start is a human pressing a button
in a GUI, so the host does not know it at all. There is therefore **no clock prior
of any kind**, coarse or fine.

That turns out not to matter, because the camera and the Channel D photodiode watch
the *same cavity transmission*. Frame k's total brightness is, by construction, the
Channel D trace boxcar-integrated over that frame's exposure window. The offset can
be **fitted**.

Note what the camera's own clock does and does not do here. `ChunkSelector =
'Timestamp'` plus `BlockID` give exact *relative* frame times and independent
dropped-frame detection — which collapses the problem to **one unknown scalar**,
the offset between the two records. The optical fit supplies that scalar. Seen this
way, the sync cable existed to measure a single number.

### It works, on real data

Tested by taking a real Channel D trace (2026-08-23, 33 cm), synthesising the
brightness sequence a 100 fps camera would report, adding noise, and fitting the
offset back with gain and dark level marginalised out (counts-per-volt is unknown,
so only the *shape* of the sequence carries timing). The offset was searched blind
over a full 400 ms window — no prior.

| frame-brightness noise | RMS error | wrong frame | margin over rival |
| --- | --- | --- | --- |
| 1% | **0.050 ms** | 0.0% | 26× |
| 3% | **0.134 ms** | 0.0% | 4.5× |
| 10% | 0.478 ms | 4.0% | 1.7× |
| 30% | 1.208 ms | 21.0% | 1.1× |

"Wrong frame" is an error above half a frame period — the only error that matters.
"Margin" is how much worse the best rival minimum more than 30 ms away is; above 1
means the true offset wins.

At ≤3% noise this is sub-0.15 ms with no failures — finer than the cable needs to
be, and it carries its own quality flag (the margin), which the cable does not.

It works because peaks that straddle a frame boundary split their light between two
frames in a ratio that pins the offset hard, while the ~40 peaks that do not
straddle each still assert "I am in frame *n*", and 40 such constraints at
irregular spacings intersect to almost nothing.

### The one real caveat: saturation

The failure mode is *model mismatch* — the camera saturating while the photodiode
does not. Measured by clipping only the synthesised camera response:

| samples clipped | RMS error (1% noise) | wrong frame |
| --- | --- | --- |
| 0% | 0.043 ms | 0.0% |
| 1% | 0.116 ms | 0.0% |
| **5%** | 0.493 ms | **17%** |
| 10% | 0.987 ms | 29% |

A slightly clipped 0th order is survivable; a properly saturated one is not.
Two fixes, both cheap:

1. sum frame brightness over **unsaturated pixels only**; or
2. apply the same clip to the model, with the cap a fitted parameter — the level is
   known exactly (255 in Mono8). This turns mismatch back into a matched model.

### Option B — an engineered optical marker (complement, not replacement)

During the record, **block the beam once** with a card. Both the camera and
Channel D see the same dropout edge, and cross-correlating just those two edges
gives the offset directly — independent of peak structure and immune to saturation,
since a dark edge cannot clip. Doing it at both ends of the record also gives the
clock-rate ratio.

Its value is as an *independent check* on the correlation fit, which is otherwise
exactly what the cable would have been for.

### What the cable would still buy, if one ever turns up

- It works when the camera sees nothing at all (dark record, blocked beam,
  misaligned cavity) — though then there is no video worth synchronizing.
- It is independent of the science signal, where option A infers timing from it.
- Option A degrades if the ROI clips the higher-order modes, i.e. precisely for the
  modes this feature exists to identify. Keep the ROI generous.

### Assumption to confirm at the bench

That the camera and the Channel D photodiode look at the same transmitted beam (a
split of it). Anti-correlation is fine — the fitted gain simply comes out negative.
A genuinely different port would not be.

### What still has to be measured

Everything above is real on the scope side and synthetic on the camera side. The one
number that decides which row of the table applies is the **actual per-frame
brightness noise**. It needs no scope, no cable and no correlation: run the camera
at 100 fps / 10 ms exposure, record a burst, plot total brightness per frame, and
see whether the resonances stand out cleanly. The same session settles the two
questions left over from Part 1 — whether the firmware binning is Sum or Average,
and whether the frames are too dim.

## The cable — wanted, but no longer blocking

Checked 2026-08-26: there is no I/O cable for either camera, and the I/O
connector has never been used. Since the optical route above was adopted this no
longer blocks anything, but a cable would still be worth having as an independent
check — see *What the cable would still buy*.

What to obtain — one of:

1. A **Basler I/O cable, HRS 6p / open end** (the standard accessory for ace
   cameras with the HR10A-7R-6PB receptacle), then solder a BNC onto the two
   wires that land on pins 1 and 6.
2. A bare **Hirose HR10A-7P-6S** plug plus a short length of two-core cable and a
   BNC.
3. Whatever shipped in the cameras' original boxes — worth looking before ordering.

Wiring, again, because getting it wrong is the one way to damage the camera:

- **pin 1 (Line 3, GPIO) → BNC centre**
- **pin 6 (GPIO ground) → BNC shield**
- pin 5 is the *opto* ground and must not be used with a GPIO line
- **set `LineMode = 'Output'` on Line3 in software before connecting anything.**
  GPIO lines default to Input, and Basler's manual warns that applying the wrong
  signal to a GPIO input can damage the camera.

## How Part 1 was actually done (for whoever repeats it)

Not through the Pylon Viewer. Two throwaway scripts, kept only long enough to
produce the numbers above:

- **checks 1 and 5** — a pypylon probe that enumerates, for every connected
  camera, the `LineSelector` / `LineMode` / `LineSource` tree (temporarily forcing
  `LineMode = 'Output'` so the `LineSource` entries report as available, then
  restoring it), plus sensor size, pixel formats, binning and decimation ranges,
  `ResultingFrameRate`, the throughput cap and its maximum, and the chunk
  selectors. Faster and more complete than clicking through the Viewer, and it
  reports both cameras at once.
- **check 6** — read the `.modemarks.json` sidecars that
  `pico_scope/mode_map_2d.py` already leaves next to each marked measurement.
  Their `marks` entries are `[0th, 1st]` pairs of Lorentzian fits, so `x0`
  differences give the 0th→1st spacing directly and successive 0th orders give the
  FSR, with no re-marking. One `.psdata` was reconverted through
  `utilities.utils.psdata_buffer_csvs` purely to read the units row and confirm
  the `Time` column is in seconds — it is worth confirming per file, because other
  cached exports in the same tree use milliseconds.

---

# Phase 1a — done (2026-08-26)

`basler_cam/basler_cameras.py` now carries everything the brief asked for:
`frame_rate_hz` / `resulting_frame_rate`, `set_roi`, `set_roi_full`,
`max_frame_size`, `set_binning`, `set_pixel_format`, `set_throughput_limit`,
`set_exposure_active_output`, `enable_chunks`, `record_burst`,
`max_frame_rate_for`, `assert_frame_rate_reachable` and `describe`, plus the
module-level `burst_timing` and the node helpers `_snap` / `_clip` /
`_entry_available`. `CameraStreamer` and `self_test()` are untouched, as required
— the new checks live in `burst_self_test()` and `probe_binning_mode()`, reachable
as `python basler_cam/basler_cameras.py --burst-test [serial]` and `--binning-mode`.

## Frame rate — what actually limits it

The plan assumed the link was the constraint and that binning relieved it. Both
halves are wrong on this camera. Measured on 25173136, Mono8, exposure 1 ms, link
cap 419 MB/s, sweeping the ROI:

| ROI | resulting | ROI | resulting |
| --- | --- | --- | --- |
| 2048 × 2048 | 90.0 Hz | 2048 × 512 | 350.4 Hz |
| 1024 × 2048 | 90.0 Hz | 1024 × 512 | 350.4 Hz |
| 512 × 2048 | 90.0 Hz | 2048 × 256 | 676.1 Hz |
| 2048 × 1024 | 178.4 Hz | 512 × 256 | 676.1 Hz |
| 512 × 1024 | 178.4 Hz | 256 × 256 | 676.1 Hz |

**Readout is paced per row — about 5.4 µs each — and depends on nothing else.**
Width is free. And binning does *not* shorten it: 2×2 at 1024×1024 reads the same
2048 sensor rows as 1×1 at 2048×2048 and runs at the same rate. Binning happens
after readout on this sensor, so it reduces the payload but not the readout time.

So there are three independent ceilings, and the useful thing is to know which one
is low:

1. **exposure** — the rate can never exceed 1/exposure, so a 10 ms exposure caps
   at ~99 Hz. Set exposure just under the period, not equal to it.
2. **readout** — set by ROI *height* alone; crop rows, keep columns.
3. **link** — `DeviceLinkThroughputLimit` ÷ payload; the only one binning helps.

`assert_frame_rate_reachable()` works out which of the three is binding and says
so, instead of listing remedies that cannot help.

## The chosen configuration, verified

Binning 2×2, frame **1024×512** (= 2048×1024 sensor: full width, half height),
Mono8, throughput cap 212 MB/s, exposure 9.9 ms, requested 100 Hz.
`ResultingFrameRate` 99.6 Hz. A real 100-frame burst:

- **0 dropped frames**
- frame period from the camera's own timestamps: **10.0406 ms ± 0.0000 ms**
  (99.60 Hz) — the camera's clock is essentially jitter-free over a second
- timestamps confirmed to be in **nanoseconds**, by comparing the implied period
  against the requested rate rather than assuming it
- 0.52 MB per frame, ~52 MB for a 1 s capture

## Binning mode: measured to be Sum

`BinningHorizontalMode` / `BinningVerticalMode` do not exist on this model, so the
mode cannot be read — but it can be measured, and `probe_binning_mode()` does:
the same dark scene at binning 1 and binning 2, identical exposure and gain, gave
mean levels of **5.79 and 20.61**, a ratio of **3.56**.

That is Sum (which would give exactly 4), not Average (which would give 1). The
shortfall from 4 is the black-level pedestal, which is added per *output* pixel and
so does not multiply: with mean = S + P at binning 1 and 4S + P at binning 2, a
ratio of 3.56 implies P ≈ 0.8 counts, which is a plausible dark offset.

**This is the good outcome** — it is the 4× signal the plan hoped for, and the
badly-coupled measurements this feature exists for are exactly the dim ones.

## Measured on the live cavity, 2026-08-26 — Option A validated

Laser on, transmission good, camera 25173136 on the cavity output. One 1 s burst
at 99.60 Hz, binning 2×2, 1024×512 frame, Mono8, exposure 9.9 ms, gain 0 dB:

- **0 dropped frames**, period 10.0406 ms ± 0.0000 ms
- peak pixel **50 of 255**, saturation **0.000%** — plenty of headroom, and well
  clear of the 5%-clipped regime that breaks the offset fit
- **noise / span = 2.37%** on a plain full-frame mean

2.37% sits between the 1% and 3% rows of the offset-fit table, i.e. **~0.1 ms
expected error with no wrong-frame assignments**. Option A works on this setup.

The brightness sequence also validates itself against check 6 without involving
the scope at all: the resonances come in **pairs 30–35 ms apart, with ~200 ms
between pairs**, against the 34.3 ms median 0th→1st spacing and 179–287 ms FSR
measured from the `.modemarks.json` sidecars. And the frames on different peaks
show visibly different transverse patterns — a clean two-lobed 1st order next to
compact higher-order structure — which is the whole point of the feature.

### Mask the brightness; do not raise the gain

The mode covers roughly 1% of the frame, so a full-frame mean spends 99% of its
pixels accumulating noise with no signal in them. Restricting the sum to pixels
that actually vary during the sweep (peak-to-peak above 15% of its maximum,
~14 000 of 524 288 pixels) is worth **5–7×**. Gain, by contrast, makes things
worse — the noise is shot-limited, not read-limited, so gain amplifies both and
adds its own:

| gain | peak pixel | saturated | full-frame noise/span | masked | gain from masking |
| --- | --- | --- | --- | --- | --- |
| **0 dB** | 49 | 0.000% | 2.17% | **0.40%** | 5.4× |
| 6 dB | 104 | 0.000% | 3.54% | 0.51% | 6.9× |
| 12 dB | 214 | 0.000% | 4.50% | 0.64% | 7.0× |
| 18 dB | 255 | 0.001% | 3.68% | 1.25% | 2.9× |

**Use gain 0 dB.** At 0.40% masked, the offset fit is better than the best row of
the simulated table.

One tension worth understanding rather than optimising away: masking improves the
noise but slightly *breaks the model*, because the photodiode sums all transmitted
light while a masked camera sum does not. If a mode ever falls outside the mask,
the two sequences diverge — the same class of failure as saturation. Since the
unmasked 2.37% is already comfortably good enough, the capture should **store both
series** and let the fit report its margin for each, rather than committing to one.
A generous mask (low threshold, dilated) gets most of the noise benefit with
little of the mismatch risk.

### The ROI needs a vertical offset — the mode is not centred

Measured from a whole-sensor burst: the light that varies during the sweep occupies
**sensor rows 1002–1358 and columns 1244–1462**, centred at sensor (1180, 1352)
against a sensor centre of (1024, 1024).

So the 2048×1024 sensor window is taken at **`offset_y = 334` in binned pixels**
(sensor rows 668–1692), which centres it on the mode and leaves ~330 binned rows of
margin on each side for the larger higher-order patterns. Full width is kept, since
width costs nothing.

This is per-alignment, not a constant — re-run the reconnaissance whenever the
cavity is realigned.
