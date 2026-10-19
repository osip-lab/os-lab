# Mechanical control: design record

Status: **plan only** (written 2026-10-19 on the laptop, no hardware connected). Nothing here is
implemented yet. Read this before starting the work in the lab, and update the "Decisions" and
"Open questions" sections as things get settled.

## Goal
Let Kalishlot drive motorized optomechanics (linear stages, rotation stages, goniometers, piezos) to
reach and keep a cavity alignment. Feedback is always the photodiode (scope) and the camera, and
sometimes the motor's own readback (closed-loop devices).

Core pieces:
1. Motor control with safety built in by construction.
2. Feedback: motor readback, and optical metrics from the scanned spectrum + synced camera.
3. Smart control: an optimizer that moves the knobs, respecting a DoF hierarchy, banned DoFs,
   hysteresis and limited repeatability.

## Decisions
- **Hardware is deliberately left undecided.** Candidates:
  Standa motorized positioners (at least 3 axes; standa.lt catalog item 525),
  PI E-872.401 Q-Motion PiezoMike drive electronics, Thorlabs MDT693B 3-axis piezo driver,
  Thorlabs Kinesis motors. So the design is vendor-neutral: one `Axis` interface, one thin driver per
  vendor, added when the hardware is chosen.
- **Safety violations:** targets outside the allowed range are **rejected** (raise `ValueError`, never
  silently clamped, so the optimizer is never fooled about where it is). Legal but large jumps are
  **automatically ramped** in small steps at a limited slew rate.
- **Build a simulator early.** Its physics core is built on `cavity_design` (sibling repo
  `../cavity-design`) so the cavity-design project can use it too. Starting points there:
  the `cavity-calculations` skill, `examples/perturbation_tolerance_analysis.py`, and the
  `playground.ipynb` cell under "# Set ups exploration - small perturbation given a setup params - two
  dynamical widgets". Lockstep rule applies: pull both repos together.
- **Metric 2.b.3** (0th/1st-order intensity ratio) is **deferred**; rarely used for alignment.
- **Optical metrics are developed offline first**, on recorded synced sessions in Dropbox:
  `Labs Dropbox/Laser Phase Plate/Daily measurements and notes/2026-10-11`, `2026-10-12`, `2026-10-15`.
  Many sessions have hand-made `*_scope.modemarks.json` files (pairs of 0th/1st-order Lorentzian fits:
  `x0`, `gamma`, `height`, `y0`). These are the ground truth for automatic mode identification.

## 1. Motor control and safety
Safety by construction, as proposed. Specifics:

- **Enforce safety in the device layer, not in the kalishlot adapter.**
  The existing Rigol 5 V cap is a lesson here. It lives in `kalishlot/adapters/rigol_dg.py`
  (`AMPLITUDE_LIMIT_VPP`), as a silent clamp on amplitude only:
  - any script calling `rigol_fg/rigol_dg.py` directly bypasses it;
  - offset is not limited, so amplitude + offset can exceed 5 V;
  - channel 2 and the JS box have no cap.

  For motors, a `SafeAxis` wrapper owns the only handle to the raw driver. The kalishlot adapter,
  standalone scripts and the optimizer all go through it. (Side task: fix the Rigol cap the same way,
  as a peak-voltage limit `|offset| + amplitude/2` in the device layer.)
- **Rules per axis**, from a versioned config file (Python, like `pico_scope/run_config.py`), not
  editable from the UI:
  - absolute soft range: reject outside it;
  - max single step: larger legal moves are auto-ramped;
  - max slew rate (V/s, um/s, deg/s);
  - settle time after a move, before any measurement;
  - optional travel budget per session;
  - `frozen` flag: reject any move. **This is where "ban a DoF" lives**, so the optimizer cannot
    bypass it.
- **Known state before the first move.** On open, read position/voltage and never write (as
  `RigolDG.open()` does). Refuse to move an axis whose position is unknown (unhomed stepper, unread
  piezo voltage).
- **Never jump on close, error or E-stop.** Close = hold where it is (ramp somewhere only if asked).
  Zeroing a piezo on exit is itself a sudden jump. The UI gets an E-stop: stop motion, hold.
- **Log every move**: commanded target, read-back, time, and who asked (UI, script, optimizer).
- Hardware limits are a second layer, not a replacement (e.g. MDT693B's 75/100/150 V limit switch).
- Otherwise follow `ADDING_DEVICES.md`: pure-Python device layer outside kalishlot, thin adapter.
  `rigol_dg.py` is the template for a control-only adapter.

## 2. Feedback

### 2.a Motor readback
Read and log it. For open-loop devices treat it as **untrusted**: open-loop piezo voltage, stick-slip /
inertia drives, steppers without encoders (there "position" is a step count). Each axis declares
capability flags: `has_readback`, `closed_loop`, `units`.

### The biggest undefined item: one evaluation is slow
The recorded sessions use a 0.05 Hz, 3 Vpp ramp (20 s period, so 10 s per sweep) and 5.6 s captures
at about 14 fps with 70 ms exposure. One metric value therefore costs about one sweep plus move and
settle time, roughly 10 s. A few hundred evaluations is about an hour. This drives the whole optimizer
design: it must be sample-efficient and noise-aware.

To decide in the lab: can the scan be narrowed and sped up during optimization, so it covers only the
0th + 1st order region? The fixed turnaround dead time makes very short sweeps wasteful.

### 2.c Scan nonlinearity after turnarounds (proposed design)
- Split each capture into monotonic sweeps using the extrema of the aux channel (the ramp, Channel B
  "Temperature modulation Voltage"). Drop a configurable guard window after each turnaround. Start at
  0.5 s and measure the real value from data.
- **Never mix up- and down-sweeps.** Thermal lag shifts peak positions and heights between them.
  Optimize on one direction only, or treat the two as separate samples.
- The aux channel is the **drive voltage, not the laser frequency**. For time-to-frequency conversion
  use the modes themselves (FSR / sideband spacing) as a local ruler.
- **Reject saturated sweeps.** The 2026-10-12 sample session has `overflow_channels: ['D']`.
  Keep the scope range fixed during a run: autoranging between evaluations silently changes the gain.

### 2.b.1 Intensity
Use the fitted Lorentzian height (and area) of the 0th-order peak, not the raw maximum, which is
biased by noise. Report the spread over the sweeps within one capture.
Open: is there a reference photodiode to normalize out laser-power drift?

### 2.b.2 How Gaussian the 0th order is
Two changes to "the most Gaussian frame is the 0th order":
- **Select the frame by spectrum + sync, then score it.** Candidates are the frames whose exposure
  window contains the 0th-order peak (`pico_scope/mode_video_sync.py`: `frame_windows`,
  `frame_at_time`). Picking by best fit is fragile: dim and partly-exposed frames can fit "well".
- **Use a scale-free score.** R² and MSE change with brightness, so a brighter but worse mode can win.
  Prefer the residual energy fraction (1 - overlap with the fitted Gaussian), plus ellipticity
  sigma_x/sigma_y and the ellipse angle. Astigmatism shows up directly in those.
  Reuse `basler_cam/gaussian_fit.py` and the session mask.
- **Check timing.** A 70 ms exposure integrates the ~1 ms transit plus background. The 0th-to-1st order
  spacing in the hand marks is ~130 ms, about two frames, so a frame can sometimes contain both modes.
  Flag such frames.

### 2.b.4 Splitting of the first-order mode
The proposed approach is sound. Additions:
- **Use a physics prior.** The expected 0th-to-1st order offset (transverse mode spacing, Gouy phase)
  can be predicted with `cavity_design` from `long_arm_m`, which every session already records. That
  narrows which peaks to test.
- **Validate candidates on the camera.** A 1st-order frame shows two lobes (HG10/HG01). Score it
  against lobe templates or with second moments.
- **Fit the pair** with the existing `fit_lorentzian_pair` in `pico_scope/mode_analysis.py` and convert
  the splitting to frequency with the local scan calibration. Report it relative to the birefringence
  baseline of the molded lens.
- **Risk.** Near degeneracy, or when the transverse spacing is close to a multiple of the FSR, the
  "which side in time" rule is ambiguous. Then rely on the camera check, and report "not identified"
  rather than guess.
- **Validate the detector against the hand marks** before trusting it in a loop.

### Additions to the metrics
- **Mode centroid on the camera** as a guard metric, so the loop does not walk the beam off the sensor
  or the mask.
- **A validity flag on every metric** (no peak found, saturated, ambiguous identification). The
  optimizer must see "invalid", not a zero.
- **A composite objective needs defining.** For example: maximize intensity subject to
  gaussianity >= threshold and splitting <= limit, rather than a weighted sum.

## 3. Smart control

### 3.a Hierarchy and banned DoFs
- Config: ordered tiers of axes. Each axis has enabled/frozen, a step scale, and a per-run trust region
  (a subset of its safety range).
- Staged search: tier 1 alone (1D scans first), then tiers 1+2 jointly, and so on. "The gradient
  points to a combination of DoFs" means a joint local optimization over the active set, using a
  finite-difference gradient with steps larger than the noise floor.
- **Virtual axes** for known coupled moves, e.g. "beam walk" = tilt M1 plus compensating tilt M2, or
  translation plus tilt about a pivot. These are often the right DoFs for alignment and are cheap to
  build on top of `Axis`. A virtual axis passes through the safety rules of every real axis it moves.

### 3.b Hysteresis and repeatability
Hysteresis is only one source of repeatability error. **Every motor also has an inherent random
positioning error**: commanding the same target twice, even from the same direction, lands in two
slightly different places (stepper microstep error, stick-slip step-size scatter, piezo creep, encoder
resolution, thermal drift of the mount). So each axis is modelled as: systematic, path-dependent part
(hysteresis/backlash, partly removable by approach direction) + random part (not removable; only
characterized). Consequences:
- Each axis config carries its measured **repeatability sigma** (in axis units). Optimizer steps smaller
  than ~2-3 sigma are meaningless and are not proposed.
- Optimizers must model the *landed* position as noisy, not only the metric. With readback, use the
  read-back position for the model; without it, widen the uncertainty.
- Approach every target from the same direction (overshoot, then come back), configurable per axis.
- **The metric is the truth, not the commanded position.** Re-measure the incumbent best periodically.
  The final "go to best" must be verified, with a local re-search if it does not reproduce.
- Estimate the noise floor at the start (N measurements without moving). Accept an improvement only if
  it is above that floor.
- Re-measure a reference point periodically to separate drift from progress.
- Use closed-loop readback when available. Log commanded vs read-back to characterize each axis.
- In the lab, before any optimizer run: a repeatability / hysteresis characterization script per axis.

### 3.c M-LOOP
A reasonable **later** backend, not the first one.
- Good: its Gaussian-process learner is sample-efficient and noise-aware, which suits ~10 s evaluations.
- Poor fit: it assumes the cost is a function of the parameters, which hysteresis breaks. It can propose
  large jumps across the whole range, and it has no notion of tiers. Its neural-net learner pulls in
  TensorFlow.
- Plan: a small pluggable ask/tell optimizer interface. First backends: 1D scan, and coordinate search
  with anti-backlash approach and noise-aware acceptance. Then M-LOOP's GP learner per stage,
  restricted to the active set and a tight trust region.

### Also needed
- Abort that returns (ramped) to the start point or the best point.
- A "signal lost" policy: a small spiral search, or stop.
- A per-run log folder: `evaluations.jsonl` (commanded and read-back positions, metrics, validity,
  timestamps) plus raw data for selected evaluations, so runs can be replayed offline.

## Proposed architecture
Fits the existing patterns.
- **`motion/`**, a pure-Python package at the repo root (like `rigol_fg/`):
  - `axis.py`: the `Axis` interface and capability flags;
  - `safety.py`: `SafeAxis` and its rules;
  - `axes_config.py`: versioned limits and tiers;
  - `simulated.py`: `SimulatedAxis`;
  - later, one driver per vendor: `standa.py`, `pi_e872.py`, `thorlabs_mdt693b.py` (port the ASCII
    protocol from `mode_position_feedback/piezo_feedback_gui.py`), `kinesis.py`.
- **kalishlot adapter** `adapters/motion_axes.py`, control-only like `rigol_dg.py`, plus a JS "jog" box:
  step buttons, read-back, freeze toggle, E-stop, and the live intensity next to them.
- **Metrics package** (e.g. `alignment/sweeps.py`, `alignment/metrics.py`) with an offline CLI that
  runs on saved sessions and compares against `*.modemarks.json`.
- **Aligner** as a virtual adapter like `adapters/synced_pipeline.py`. It launches a child process that
  borrows axes, scope and camera via `loan_client.py`, and moves axes through `device_command`
  (precedent: the trailing scope capture switching the Rigol on), so every move passes `SafeAxis`.
  It reuses `capture_synchronized` (`pico_scope/mode_video_capture.py`). That function takes an
  `output_root` and saves a session; check whether it can return arrays without writing video to disk
  on every evaluation, or add that option.
  Scope-only metrics skip the camera for speed.
- **Simulator:**
  - physics in `cavity_design`: element shifts/tilts give a perturbed mode, hence coupling (overlap
    with the input beam), transverse mode spacing, and x/y astigmatic splitting;
  - the kalishlot side synthesizes scope traces (Lorentzians on a ramp, turnaround nonlinearity, noise,
    saturation) and camera frames (HG modes);
  - `SimulatedAxis` injects backlash, hysteresis and drift;
  - cache physics results if a cavity solve is slow.

## Phasing
1. Laptop: sweep segmentation and metrics, offline on recorded sessions, validated against hand marks.
2. Laptop: `motion/` interface, `SafeAxis`, config, simulated axis, and a unit test for every safety
   rule. Fix the Rigol cap the same way.
3. Laptop: simulator (cavity_design physics plus synthetic traces and frames).
4. Laptop: kalishlot adapter, jog box, and `smoke_test.py` checks with simulated axes.
5. Laptop: aligner with 1D scan and coordinate search on the simulator; then the M-LOOP GP backend.
6. Lab: first real drivers once the hardware is chosen. Per-axis characterization (noise floor,
   repeatability, hysteresis, settle time, safe limits). First closed-loop run on a single axis.

## Open questions (for the lab)
- Final hardware, and the DoF mapping: which element, which motion, units, safe range per axis.
- Which candidates are open-loop (PiezoMike on the E-872? Standa with or without encoder?) and which
  SDKs they need.
- Can the scan be narrowed and sped up during optimization? What is the real turnaround dead time?
- Is there a reference photodiode for laser-power normalization?
- Camera plane (near field or far field), and exposure time vs mode transit time.
- The composite objective and its thresholds; acceptable run time; the "signal lost" policy.
