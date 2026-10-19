# Mechanical control: design record

Status (2026-10-19, laptop, no hardware connected): the laptop phases 1-5 are implemented and tested
on recorded data and on a simulator; see **Implementation status** at the end. Nothing has touched real
hardware yet. Read this before starting the work in the lab, and update the "Decisions" and
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
  - settle time after a move, before any measurement (see "Measure only between moves" below);
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

### Measure only between moves (vibrations)
Motor motion shakes the setup, so a spectrum recorded while (or just after) a motor moves looks messy
and must not be used. Rules for the alignment loop:
- Never capture while any axis is moving. A move is "done" only when the driver reports it finished
  (closed-loop: in position; open-loop: last step sent and acknowledged).
- Then wait a per-axis **vibration settle delay** before the capture starts. Start with a conservative
  value (e.g. 0.5-1 s) and measure it in the lab: repeat captures at increasing delays after a move
  and take the delay after which the metrics stop changing beyond their noise floor.
- The settle delay belongs to the axis config (different mounts ring differently), and the loop takes
  the maximum over the axes that just moved.
- Optionally check for residual vibration in the data itself (peak jitter / extra peaks relative to a
  reference capture) and re-capture if it is present.
- This adds to the evaluation cost (see above), so it is one more reason to keep evaluations few.

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
- **Mode position as a target** (put the mode on a camera marker): see 3.d.

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

### 3.d Target objective: put the mode on a camera marker
In the lab the knobs of one element are often turned until the centre of the 0th-order Gaussian fit
sits on a kalishlot camera marker left earlier (a past configuration). This is an objective with a
goal rather than an optimum, next to maximizing intensity, minimizing splitting, etc.
- **Objective** `alignment/targets.py: MarkerTarget(x, y, tolerance_px)`; `from_markers(path, label)`
  reads the marker by label (or id) from a saved marker file or from kalishlot's `device_state.json`.
  Score = minus the distance from the measured centre to the marker. The axes it may use are the loop's
  tiers (freeze the rest), e.g. only the two tilts of one mirror.
- **Units.** Markers are in uncropped, unbinned sensor pixels; capture frames are a binned ROI. The
  metrics add `sensor_x/_y` = the fit centre converted with the session's `camera` block
  (`offset_x/_y` in binned px, `binning_x/_y`): sensor = b * (offset + index) + (b - 1)/2.
  kalishlot's XIMEA runs unbinned, so its markers are true sensor pixels. *Open:* check this for the
  Basler if it is ever used binned in kalishlot.
- **Noise (the fit lands somewhere slightly different in every blink).**
  - One evaluation = the median centre over all usable 0th-order blinks of a capture (~10 per 5.6 s),
    not one fit.
  - The loop only accepts a move that beats the incumbent by more than the measured noise floor of the
    distance, so it does not chase jitter.
  - "Reached" = the **mean** centre of the hit plus `confirm_repeats` (default 2) more captures in
    place is within `tolerance_px`. One lucky capture does not end the run. The loop then stops where
    it stands; no verification move is needed.
  - A tolerance tighter than the noise, or than what one minimum motor step (3 sigma repeatability)
    moves the spot, cannot be confirmed. The run then ends on budget or stall with
    `summary['target_reached'] = False`. Pick the tolerance from the jitter seen in the noise floor.
- **Guard the resonance.** Walking the mode costs coupling (the input beam does not follow), so pass a
  constraint such as `('intensity', '>=', x)`. Without the 0th order there is no position.
- **The marker itself is noisy.** A marker placed from one live fit carries that fit's jitter. Place
  it from an averaged fit when the tolerance is tight.
- **Search: Jacobian steering, then coordinate search.** The centre moves almost linearly with the
  knobs, so each stage first probes every axis, growing the probe step until the centre moves by more
  than 3x its noise. It then moves straight at the marker by least squares (minimum-norm when there are
  more axes than 2), within a trust region of 8x the probe, refining the Jacobian after every move
  (Broyden). It falls back to the coordinate search if a move fails three times.
  - Why: the coordinate search alone failed on the cavity_design simulator. One 1 V step moved the
    spot ~0.8 px, about the per-capture noise, so steps were rejected as noise and shrank until it gave
    up (13.6 px -> 14.1 px).
  - With steering: 13.6 -> 2.2 px (true distance) in 13 evaluations, confirmed over 3 captures, coupling 0.978.
  - Hysteresis and landing error make each move approximate; the Broyden update and the confirmation
    absorb that.

### Also needed
- Abort that returns (ramped) to the start point or the best point.
- A "signal lost" policy: a small spiral search, or stop.
- A per-run log folder: `evaluations.jsonl` (commanded and read-back positions, metrics, validity,
  timestamps) plus raw data for selected evaluations, so runs can be replayed offline.

## Proposed architecture
Fits the existing patterns.
- **`motion/`**, a pure-Python package (GUI-free, like `rigol_fg/`). It lives in `kalishlot/motion/`,
  next to `kalishlot/alignment/` and `kalishlot/simulation/`, to keep the repo root clean. All three
  import as top-level packages (`import motion`) with `kalishlot/` on `sys.path`:
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
  (Recorded data: the frequency turnaround lags the drive by ~0.5-0.9 s; guard set to 1.0 s.)
- **Which side family is the first order?** Every 2026-10-12 session shows two side families of
  comparable strength at s ~ +0.31 and s ~ -0.31 (s = signed FSR fraction, see `alignment/spectrum.py`).
  The camera favours +0.31 (HG10/donut template overlap 0.73-0.76 vs 0.54-0.66). The hand marks pick
  +0.31 in two sessions but -0.30 in 113244. On 44.5 cm (2026-10-11) the strongest family is -0.14 in one
  session and +0.13 in the next, four minutes apart, same scan direction. Confirm by eye in the lab, then
  set `MetricsConfig.first_order_s`.
- Is the Rigol CH1 envelope right (|offset| + amplitude/2 <= 2.5 V, see Implementation status)?
- Where should the simulator physics live long term: os-lab `simulation/cavity_physics.py` (now) or
  the cavity-design repo?
- Is there a reference photodiode for laser-power normalization?
- Camera plane (near field or far field), and exposure time vs mode transit time.
- The composite objective and its thresholds; acceptable run time; the "signal lost" policy.

## Implementation status (2026-10-19, laptop)

Everything below runs without hardware.

### Tests to run
Paths are relative to `kalishlot/`. Run from there (the `-m` forms need it).
```
python alignment/sweeps.py             # sweep segmentation self-test
python alignment/spectrum.py           # peaks, FSR, families, splitting, saturation, reassignment
python alignment/camera.py             # Gaussian score, first-order templates, purity
python -m alignment.metrics            # whole-capture metrics
python alignment/validate_marks.py     # real sessions vs hand marks (edit ROOTS at the top)
python -m motion.test_safety           # 14 safety-rule tests
python ../rigol_fg/test_safety_limits.py  # Rigol envelope against a fake instrument
python -m simulation.cavity_physics    # physics (cavity_design + analytic stand-in)
python -m simulation.setup             # simulated scope + camera + metrics
python test_motion_axes.py             # kalishlot adapter (also hooked into smoke_test.py)
python alignment/test_loop.py          # loop: recovers, frozen, settle, abort, signal lost, 1D scan,
                                       # marker target, marker lookup + sensor units
python alignment/run_alignment_sim.py  # demo run on the cavity_design physics (edit the globals)
python alignment/visualize_alignment_sim.py  # GIF: loop aligning a cavity with a hidden misalignment
```

### What exists
- **`alignment/`** (pure numpy/scipy):
  - `sweeps.py`: turnarounds from the aux ramp (zig-zag with hysteresis), sweeps with direction,
    1.0 s guard after each turnaround.
  - `spectrum.py`: 0th-order peaks (tallest per FSR; Lorentzian fit with an adaptive window; saturated
    samples excluded and flagged), FSR from a periodicity test, local FSR per peak, scan-nonlinearity
    flag, side families as signed offsets s, first-order peak and splitting (one vs two Lorentzians),
    reassignment of misidentified 0th orders (see Lessons).
  - `camera.py`: scale-free Gaussian overlap and ellipticity (refined fit on a crop: the live fit's
    tolerances are too loose for astigmatism), HG10/LG01 template overlap for first orders, exposure
    purity from the scope.
  - `metrics.py`: `evaluate_capture` / `evaluate_session` give metrics with value, spread, n, valid and
    reason, per scan direction (never pooled): intensity, gaussian_overlap, ellipticity, centroid_x/y,
    first_order_s, split_linewidths (separation / component FWHM), split_fsr.
  - `loop.py`: `AlignmentLoop` (tiers, noise floor, noise-aware coordinate search plus a gradient step
    along a linear combination, min step = 3 x repeatability sigma, outlier confirmation, periodic
    re-measure of the incumbent, verified final return, abort to best/start, JSON-lines log),
    `Objective` (metric, direction, constraints), `scan_axis` (1D scan, one-sided approach).
    Target objectives stop the loop once `reached()` is confirmed in place.
  - `targets.py`: `MarkerTarget` (section 3.d), marker lookup from a marker file or `device_state.json`.
  - `metrics.py` also gives `sensor_x/_y` (centre in the markers' units) when the sensor geometry is
    known; `evaluate_session` takes it from the session's `camera` block.
- Axis names follow cavity_design (optical axis = x): simulated axes `sim_mirror_phi` (end mirror,
  xy-plane tilt), `sim_mirror_theta` (xz-plane tilt), `sim_lens_x` (lens along the optical axis).
  Paths to the cavity-design project and to Dropbox come from `local_config.py`
  (`PATH_CAVITY_DESIGN_PROJECT`, `PATH_DROPBOX`), so the scripts run on any lab computer.
- **`motion/`**: `Axis` interface; `SafeAxis` (reject out of range, auto-ramp, slew, settle, vibration
  settle via `ready_at`, freeze in config / at run time, E-stop latch, unknown-position refusal,
  anti-backlash approach, travel budget, move log); `VirtualAxis` (all components checked before any
  moves); `SimulatedAxis` (backlash, random landing error, drift, readback modes); `axes_config.py`
  (3 simulated axes, tiers, a template for a real axis).
- **`simulation/`**: `cavity_physics.py` (cavity_design: coupling = |overlap|^2, per-axis Gouy phase gives
  s_x and s_y, ellipticity, beam tilt; ~50 ms per evaluation; NaN = no stable mode = signal lost; plus an
  analytic stand-in); `setup.py` (scan through a thermal low-pass, which reproduces the turnaround stall;
  HG_nm at n*s_x + m*s_y as in `Cavity.plot_spectrum` but per axis; saturation; camera frames that
  integrate each mode over the exposure; vibration jitter when capturing too soon after a move;
  the frame is a binned ROI of a sensor; the 0th-order centre moves with the beam tilt (far field:
  shift in beam radii = tilt / NA) and jumps by `pointing_jitter_px` between blinks).
  `run_alignment_sim.py` with `MARKER_FROM` rehearses the marker objective.
- **kalishlot**: `adapters/motion_axes.py` (a box per driver group; blocking `wait` moves for scripts)
  and `static/boxes/motion_axes.js` (jog box: E-STOP, jog +/-, go-to, freeze, declare position, lamps),
  registered in `server.py` and `app.js`. **Not tried in a browser**: the server's dependencies
  (uvicorn, websockets) are not installed on the laptop. Only the adapter test and a JS syntax check ran.
- **Rigol**: the CH1 cap moved into the device layer (`rigol_fg/rigol_dg.py`, `SAFETY_LIMITS`):
  amplitude <= 5 Vpp **and** |offset| + amplitude/2 <= 2.5 V, refused with ValueError for every caller.
  The box still clamps to the allowed envelope. **Behaviour change:** a non-zero offset combined with a
  large amplitude is now refused or clamped; the old cap ignored the offset.

### Findings from the recorded sessions (2026-10-11/12/15)
- All 10 hand-marked 0th orders are detected; unsaturated heights match the marks within a few %.
- Time per FSR: ~0.42 s on up-sweeps, ~0.52 s on down-sweeps, drifting by up to +-25% within one sweep
  right after the guard. Every quantity therefore uses a local FSR.
- 113244 is saturated (2 V range): its intensity is invalid by design, since fitted heights of clipped
  peaks are extrapolations.
- 0th-order frames: Gaussian overlap ~0.97 on 2026-10-12, 0.72-0.85 on the low-finesse 2026-10-11.
- Analysis cost: ~3 s per capture with camera metrics, ~0.2 s scope-only.

### Lessons from the simulator
- **Mode misidentification is the main failure mode.** When misaligned, the first order can be taller
  than the 0th order. "Tallest peak = 0th order" then makes the loop maximize the wrong mode and walk
  away; it did so in the first test. The fix in place: with a known first-order offset
  (`MetricsConfig.first_order_s`), a pick whose +s neighbour is weak (< 20% of it) while a taller peak
  sits at -s is reassigned to that peak. Limit: when the 0th order is at the noise level the scope alone
  cannot tell; then the camera guard (`min_fundamental_overlap`, off by default) or a better start is
  needed.
- A fit inconsistent with its own samples once became the incumbent. Such fits are now rejected, and
  large improvements are re-measured before they are accepted.
- Visualization (`alignment/visualize_alignment_sim.py`): the true aligned point is offset by a hidden
  misalignment only the simulator knows; the loop sees only synthetic scope captures. With +25/-18 urad
  and 60 um hidden, hidden coupling went 0.931 -> 0.9997 in 80 evaluations. `Cavity.plot` raises at its
  last step (mirror origin/image markers, "Central line is not well defined") for the example cavity;
  the script catches it and keeps the modes. The xz plane cannot be plotted (aspheric surfaces are
  xy-only in cavity_design).
- With backlash, the best *commanded* position differs from the true one by ~backlash/2. Trusting the
  metric (re-measure, verify) handles it.

### Not done yet
- **M-LOOP backend**: not written. M-LOOP is not installed here and pulls heavy dependencies; the
  stage/objective structure of `AlignmentLoop` is where a GP backend would plug in.
- **Lab measurement function** for the loop: an in-memory `capture_synchronized` (no video written per
  evaluation) wrapped as `measure()`; today `capture_synchronized` takes an output folder.
- **Aligner box** in kalishlot (a virtual adapter launching the loop in a child process via loan_client).
- Real drivers (Standa, PI E-872, MDT693B port, Kinesis) and per-axis characterization in the lab:
  `scan_axis` up and down for hysteresis, repeats for repeatability, delays for the vibration settle.
