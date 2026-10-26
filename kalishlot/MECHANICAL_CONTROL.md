# Mechanical control: design record

Status (2026-10-19): the laptop phases 1-5 are implemented and tested on recorded data and on a
simulator. In the lab, the first real driver (Thorlabs MDT693B, large-mirror tilts) works with the piezos
still disconnected. See **Implementation status** at the end. Update the "Decisions" and
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
- **Continuous capture for the loop (2026-10-19).** Scope and camera keep streaming as in kalishlot;
  an evaluation cuts its data out of the streams instead of a grab-configure-record-sync cycle
  (`alignment/live_capture.py`, see Implementation status). The camera records only on demand,
  capped in seconds and bytes. The scope ring buffer always holds the last ~70 s. A partial sweep
  is enough for most feedback.
- **The aligner runs inside the kalishlot server** (a virtual adapter reading the adapters' streams
  directly), not as a child process borrowing devices through `loan_client`, so the devices never
  stop streaming.
- **Scope rate:** 20 kHz is enough (the lab's manual measurements use it); streaming allows 100 kHz.
- **Camera for alignment:** the XIMEA (s/n QXNAA2506000).
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

### 3.e Nested optimization: ridges and kept targets (2026-10-24, laptop)
Two situations where one joint optimization over all axes fails:
- **Narrow ridge.** Almost every direction drops the target sharply and only one combination of axes
  gains, slowly. Single-axis steps (and the per-axis-scaled gradient step) fall off the crest every time,
  so the loop shrinks its steps and concludes it is at the optimum. Noise makes it worse.
- **A hierarchy of targets**, e.g. maximize the coupling while *keeping* the mode symmetric. A constraint
  (`Objective.constraints`) only rejects moves that break symmetry; it never restores it.

`alignment/nested.py: NestedLoop(axes, measure, levels)`, levels outermost first, each
`Level(axes, objective, steps, max_evaluations)`. For each trial position of level k the levels below are
optimized again (recursive, any depth; the innermost is a plain `AlignmentLoop`), and only then is level
k's objective measured: theta_k* = argmax F_k(theta_k, theta_below*(theta_k)). The same objective at every
level is the ridge case: the profile max_inner F(outer, inner) is smooth along the ridge.
- **The inner axes must be the stiff ones.** An inner problem without a sharp optimum makes the outer
  profile noise.
- **Predictor:** K = d(theta_below*)/d(theta_k): how far the optimum of the lower axes moves per unit
  step of level k's axes. Fitted by least squares to the last `predictor_window` (10) trials, accepted or
  not; each single trial's shift is noisy, the fit averages it out (a damped update that followed the
  latest trials gave 0.67 on a ridge of slope 0.5; the fit gave 0.48). Each trial starts the levels below
  at theta_below* + K delta. On a ridge, (delta, K delta) is the
  ridge direction (implicit function theorem: K -> -H_bb^-1 H_bk).
- **Outer search:** coordinate search on the profile (+step, -step, x1.5 / x0.5, never below 3 sigma of
  the repeatability), a parabola through (-step, 0, +step) when both sides fail, and a gradient step when
  the level has >= 2 axes.
- **Profile noise** (`profile_noise=True`, default; `False` turns it off). A step of level k is kept only
  if level k's target beats the best value so far by more than a threshold (`accept_sigmas` x the noise).
  The measurement noise alone (4 captures without moving) is too small for that: the levels below never
  stop exactly at their optimum, and where they stop differs from run to run, so level k's target
  measured at one fixed position varies more than the measurement noise. A lucky stop then looks like an
  improvement; on a flat profile every step can, and the level never decides it has finished. So the
  first time a level is optimized, the levels below are solved twice at the same position, the second
  time from a start shifted by one step on each of their axes. The difference d of the two results gives
  the profile noise |d| / sqrt(2); the threshold uses the larger of the two noises. Cost: one more solve
  of the levels below per level, once per run. A single difference is a rough estimate.
- **Noise:** each level's measurement noise comes from the same noise_repeats captures at the start. A level's value
  at a point is the last capture of the levels below (taken there) plus `outer_repeats` more, not the
  inner's best capture (biased upwards). Large jumps are confirmed; the incumbent is re-measured.
- **"Keep" = good enough:** `Objective(metric, good_enough=thr)` is reached once the metric is past thr;
  a loop started there measures once and stops. Maximizing an already-flat metric would chase noise.
- **Shared:** one hard evaluation budget, one history (every capture tagged with its level), stop(),
  and the best point is the full position. Every capture waits for all axes to settle. With 3 levels give
  the middle level a `max_evaluations`, or its first solve takes the whole budget.
- Simulator: new axis `sim_lens_y` (lens across the beam, `DoF(1, 'y', 1e-7)`, limits as `sim_lens_x`,
  not in `TIERS`). `AnalyticPhysics(curvature=ridge_curvature(...))` makes a rotated elongated quadratic,
  `signed_split` a splitting that one axis can cancel. `run_alignment_sim.py`: set `LEVELS`.
- `AlignmentLoop` gained `known_noise` (skip the noise floor on warm restarts), `stall_passes` (1 for warm
  inner runs), `Objective.good_enough`, steering goals of any length, and its final fine pass now only
  touches the tiers' axes.

Simulator results (`alignment/test_nested.py`, analytic physics, 150 captures allowed, 3 seeds; profile
noise on, K by least squares):
- Ridge (1/e at 40 units along (1, 0.5) and at 4 across, start on the crest 30 units out, coupling
  0.55-0.58): the plain loop stops after 25-42 captures at 0.61-0.71. Nested: 0.991-0.995, K 0.47-0.50
  (true slope 0.5). Here the profile noise came out below the measurement noise, so it changed nothing;
  the runs use all 150 captures.
- Keep symmetric (outer phi + theta on intensity, inner lens_y on asymmetry <= 1 linewidth, start 0.37-0.39):
  nested 0.996-0.998 in 103-122 captures, splitting kept below 1 linewidth; the joint loop with the
  symmetry as a constraint ends at 0.66-0.96 (one seed at 4.5 linewidths).
- cavity_design (phi + theta outer, lens_y inner, start 0.81): plain 1.000 in 55 captures. Nested with
  profile noise off: 0.999, all 150 captures used. On: profile noise 0.0083 V, 10x the measurement noise
  (0.0008 V); it stops by itself after 100 captures at 0.987 (the higher threshold also rejects small real
  gains). In the example cavity phi and lens_y are not a sloped ridge but an exactly flat valley (lens_y
  -5.5 cancels phi +30: coupling 1.000), so nesting only adds cost there. Use it when the joint loop stalls.

Known limits:
- **The scope's `split_linewidths` cannot be the symmetry metric in the simulator:** it reads 0 above
  ~2 linewidths, because a one-plane tilt feeds only HG10 and `fit_split` counts a partner below 10% as
  "not split". So 0 means "symmetric" or "cannot tell". The tests inject the truth as `asymmetry`. For
  the lab, check this, or use the camera's ellipticity / gaussian_overlap (slower).
- **K needs outer steps large enough** that the shift they cause below (K delta) is larger than the
  smallest step the lower axes can resolve (3 x their repeatability). On cavity_design K came out -0.53
  (profile noise on) / -0.96 (off), true -0.18, from 1-2 V phi steps.
- With a good-enough inner level the inner optimum is not unique, so K is noisier (one seed learned a
  junk theta column); it still helped.

**Probe steps** (`Level(..., probe_steps={axis: small step})`). On a narrow line an outer step without
the right lens correction loses the mode, so the first large steps all fail before K is known and the
steps only shrink. Each axis first takes one small trial (the levels below re-optimized, judged like any
trial), which gives K; the large steps then start already predicted.

#### A lens error hidden by the mirror tilt (cavity_design, 2026-10-24)
Example cavity with the end mirror 1 mm towards the lens (`example_cavity(end_mirror_shift=1e-3,
paraxial=False)`), end-mirror phi tilt and lens y shift, both in the xy plane:
- A mirror tilt of **53.3 urad per um of lens shift** keeps the mode almost unchanged, along a straight
  line, to hundreds of um. Across the line the mode is lost within ~1.5 um of lens or ~70 urad of tilt
  (0.2 um or 10 urad already cost ~1% coupling). Within the simulated axes' old ranges (+-37.5 urad) this
  looked like an exact compensation; it is not.
- Along the line the coupling barely changes but the mode becomes astigmatic (exact tracing):

  | lens off | coupling | splitting [peak widths] |
  |---|---|---|
  | 50 um | 0.99999 | 0.12 |
  | 100 um | 0.99979 | 0.46 |
  | 150 um | 0.99889 | 1.03 |
  | 200 um | 0.99621 | 1.79 |
  | 300 um | 0.97835 | 3.54 |

  The paraxial trace loses the mode beyond ~30 um along the line; use `paraxial=False` there.
- **cavity_design limit (not fixed, cavity-design repo):** refractive surfaces get different ABCD terms in
  the two planes from the angle of incidence (`SphericalRefractiveSurface.ABCD_matrix`), but the aspheric
  lens surface (`AsphericRefractiveSurface.ABCD_matrix`) always uses its vertex radius (9.07 mm). Off-centre,
  the real asphere is curved differently in the two planes: relative difference ~0.57 x (surface slope)^2,
  the same size as the angle term. The astigmatism above may be off by up to ~2x (rough estimate).
- **Who does what.** Start: lens -200 um, mirror tilted to hide it (coupling 0.996, splitting 1.9).
  - Outer = mirror tilt -> coupling, inner = lens -> least splitting: **fails** (lens -200 -> -210 um,
    coupling 0.996 -> 0.963). The coupling gains only ~0.4% along the whole line, less than its variation
    between inner runs (0.27%), and the least-split lens position is ~0.5 um off the line, so the inner
    level trades coupling for symmetry.
  - Outer = mirror tilt -> least splitting, inner = lens -> coupling: **works on 3 of 5 seeds** (lens ends at
    -181, -13, -157, -48, -1 um; seed 5: splitting 1.89 -> 0.00, coupling 0.9999, 167 captures). K =
    -0.01877 um/urad on every seed (true 1/53.3). The stalls: the splitting changes fast across the line
    (white lines on the map lie close together) while the coupling peak there is flat, so where the lens
    stops varies the splitting by ~0.17 peak widths between inner runs; that sets the outer threshold.
  - Symmetry is the true splitting + 0.05 noise here; the lab needs a working symmetry metric (above).
- Animation: `alignment/visualize_nested_sim.py` (writes `alignment/figures/nested_<roles>.gif` and a final
  PNG for both role choices; `LIVE = True` shows it in a window).

### 3.f Targets that are procedures, and vector targets (2026-10-26, laptop)
**Problem.** Some target values are not one capture but a procedure that ends in a measurement. Example:
align the optical axis to the linear stage under the large mirror. Move the stage and see whether the mode
moves on the mirror. Moving the stage should change only the NA. If the stage axis is at an angle to the
optical axis, the mode's spot on the mirror drifts as well.

**Why the drift measures it.** Every line through the large mirror's centre of curvature meets the mirror
at normal incidence. When the stage moves that centre along the optical axis, the same axis still closes on
itself, so the spot stays put. When the stage moves it at an angle, the axis turns and the spot drifts.

**Design.**
- `Level(axes, objective, measure=procedure)`. The procedure is any function `procedure(run)` that returns
  a CaptureMetrics; the level's objective then scores it as usual. `run` gives:
  - `capture(label)`: waits for every axis to settle, counts against the shared budget, and is logged with
    the level;
  - `move_to` / `move_by`: through the SafeAxes;
  - `positions()`, `move_back()`.
- The loop puts every axis the procedure moved back where it was, also on stop() or an error in the middle.
- A procedure level's noise floor comes from `noise_repeats` procedure calls, and it never reuses another
  level's capture.
- `alignment/procedures.py: move_and_compare(axis, travel, metrics)`: capture, move, capture, move back.
  It returns `drift_<m>` (after − before) and `drift` (the length of that vector).
- `targets.py: VectorTarget(metrics, goal, tolerance)`: bring a measured vector to a goal (score = minus the
  distance). `MarkerTarget` is now a VectorTarget of (sensor_x, sensor_y).

**Vector targets: steering, not a new search.** Minimizing |drift| with the old one-axis-at-a-time search
works without any change, but slowly. A vector target also tells the loop which way to go. For an outer
level whose objective has `position`/`goal`, `NestedLoop._steer`:
1. probes each axis (a full nested trial; the step is doubled until the vector moves by more than 3× its
   noise);
2. fits a straight line, vector = a + J·x, over **all** trials of the level;
3. moves to where the fit meets the goal, and repeats until that point stops moving;
4. moves there and keeps it (`adopt`: the fit over many trials is a better judge than one more noisy
   capture). Up to 3 rounds; then the coordinate search, if the goal is still not reached.

The fit matters here. Every trial's vector carries the scatter of where the inner level ended. Comparing
single trials then keeps a lucky one (first version: recorded 2.2 µm, true 4.1 µm).

**Simulator.**
- `short_arm_cavity()`: the example cavity with the small mirror and the aspheric lens as one OpticalSystem
  (element 0), turned about its mechanical centre.
- `stage_dofs(tilt_y, tilt_z)`: a DoF entry may be a tuple, so one axis moves several parameters. The stage
  moves the end mirror along x plus the hidden tilt.
- `ModeState.spot_y/_z`: where the long arm's central line meets the end mirror, relative to its centre, in µm.
- Facts (cavity_design):
  - a stage tilted 100 µrad from the axis gives a 13 µm drift per cm of travel towards the lens. Moving away
    from the lens passes the concentric limit, where there is no mode;
  - |split| against the large-mirror tilt is one quadratic bowl, about 10 ppm of the FSR per µrad²;
  - its minimum follows the short arm: mirror phi ≈ −1.08 × arm phi, mirror theta ≈ +1.08 × arm theta;
  - with the mirror re-tilted, the axis follows the arm about 1:1, so the drift changes only ~0.13 µm per
    µrad of arm rotation.

**Example** `alignment/run_stage_alignment_sim.py` (the user's procedure). Outer = short arm (phi, theta),
target = drift (dy, dz) → (0, 0). Inner = large mirror (phi, theta), target = least |split|. Hidden stage
tilt (150, −80) µrad, 1 cm of travel. Noise: split 2 ppm, spot 1 µm per capture, landing error 0.2 µrad.

| outer target | axis vs stage at the end (3 seeds) | captures |
|---|---|---|
| vector (steering) | 13, 17, 11 µrad (start 170) | 389, 400, 297 |
| scalar \|drift\| | 106 µrad (seed 1), stalls | 297 |

The scalar search stalls because a 40 µrad arm step changes |drift| by ~5 µm, about the scatter between
inner runs. On a linear test model (`test_vector_target_steering_beats_scalar`), steering used 52–120
captures against 115–159.

**What limits the accuracy: the large mirror's landing.** 1 µrad of large-mirror tilt turns the optical axis
by ~46 µrad. The tilt moves the mirror's centre of curvature sideways, and the point the long-arm axis pivots
about is only ~4 mm away from it. The drift is therefore ~45× more sensitive to the large mirror (~6 µm/µrad)
than to the short arm. The inner level ends a few tenths of a µrad from the symmetric point (landing error,
3σ minimum step, the flat bottom of |split|), so every trial's drift scatters by a few µm. The fit averages
that. Less splitting noise (0.5 ppm) did not improve the result, so the landing sets the limit. In the lab:
- the large mirror's repeatability matters more than the short arm's;
- more travel gives proportionally more drift signal;
- more outer trials average more.

**cavity_design bug, fixed locally (cavity-design repo, not committed).** `find_central_line_standing_wave`
used `optimize.root(...).success`. hybr reports "not making good progress" when it already sits on the root
(residual 1e-17 m). The trace then counted as failed: the overlap was NaN and the mode was "lost", at most
points once the short arm is turned. The fix also accepts a residual below 1e-12 m.

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
  **Lab rule (2026-10-19):** aux voltage rising -> the first order passes *after* the 0th order;
  falling -> *before* (`sweeps.scan_direction`, `sweeps.first_order_time_sign`). In `s` terms the first
  order has s > 0 in both directions, i.e. the +0.31 family, as the camera favoured. Live captures
  that day picked -0.36 on up-sweeps (against the rule) and +0.31 on down-sweeps (with it).
  **Implemented as a sign, not a value:** `MetricsConfig.first_order_sign = +1` (default) keeps only
  side families with s > 0. The magnitude of s still comes from the data in every sweep, since it
  changes with the long-arm length. s is a fraction of the local FSR, so the scan rate does not enter.
  Nothing in the lab path hard-codes a value (0.3 appears only in synthetic self-tests and in the
  analytic simulator stand-in).
  **Check against the hand marks (2026-10-11/12):** 9 of 12 first-order marks have s > 0. The 3 that
  do not: 110007 (44.5 cm, both marks s = -0.14) and 113244 (33.5 cm, saturated, s = -0.29/-0.31).
  Either those marks are mislabelled or the rule has exceptions. Check them by eye before trusting the
  rule for mode identification in a loop.
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
python -m motion.test_mdt693b          # MDT693B driver against a fake instrument
python alignment/test_live_capture.py  # camera recorder + live capture sync (fakes, no hardware)
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
  - `nested.py`: `NestedLoop` / `Level` (section 3.e): outer axes stepped, inner axes re-optimized at
    every step, any depth, with a learned d(inner*)/d(outer) predictor; tests in `test_nested.py`.
    Levels can measure through a procedure, and vector targets are steered at (section 3.f).
  - `procedures.py`: `move_and_compare` (section 3.f). `targets.py` also has `VectorTarget`.
  - `run_stage_alignment_sim.py`: align the optical axis to the large mirror's stage (section 3.f).
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

### Lab, 2026-10-19: first real driver, Thorlabs MDT693B
- **Hardware:** large mirror in a POLARIS-K1S3P. Horizontal tilt = controller X = `large_mirror_phi`;
  vertical tilt = Y = `large_mirror_theta`; Z unused. The MDT693B is on COM4
  (USB VID 1313 / PID 1003), firmware 1.17, s/n 2408096171-09, limit switch at 150 V.
- **Driver** `motion/thorlabs_mdt693b.py` (pyvisa, 115200 baud, CR). Written from the protocol, not by
  importing `mode_position_feedback/` (other team members own that folder and another project uses it;
  never edit it). Its `inst.clear()` fails on this port (VI_ERROR_SYSTEM_ERROR), so the driver flushes
  the read buffer instead.
  - The X and Y axes share one VISA session per port.
  - On open: checks that `id?` names an MDT693B and that the soft max is not above `vlimit?`.
  - Before every write it re-reads the output and refuses to move if the output differs from the last
    known value by more than 0.5 V (front-panel knob turned).
  - The readback is the controller's measured output: ~+0.07 V above the command at 3-10 V.
- **Tried with the piezos disconnected:** X 0 -> 5 -> 10 V and Y 0 -> 3 V through `SafeAxis`, confirmed
  on the front display. 80 V was refused by the 0-75 V placeholder soft range.
- **Limits are placeholders** (`axes_config.py`): 0-75 V, 0.5 V steps, 5 V/s, approach off. Next:
  connect the piezos, measure tilt per volt on the camera, then hysteresis, repeatability and settle time.
- **First scans with the piezos connected** (2026-10-19, through kalishlot with the live view on:
  `motion_axes` box + `live_measure` fast mode; 0, 5, 10, 15, 20 V and back to 0, ~55 s per scan):
  - X (`large_mirror_phi`): spot -0.25 px/V horizontally, -0.07 px/V vertically; intensity
    0.52 -> 0.47 V.
  - Y (`large_mirror_theta`): spot -0.46 px/V vertically, +0.05 px/V horizontally; intensity
    0.52 -> 0.60 V and Gaussian overlap 0.956 -> 0.975 (Y up improves the coupling).
  - The tilt per volt is small: the full 0-75 V moves the spot ~20-35 px. The manual knobs stay the
    coarse adjustment.
  - The X and Y cables were swapped at first (X moved the spot vertically); fixed at the controller.
    After a controller power cycle, close and reopen the piezo box (new serial session; opening only
    reads).
  - Piezos do not shake the setup: settle delays set to 0.
  - Each fast measurement takes ~6 s (3 s recording + ~3 s overhead, not yet broken down).

### Lab, 2026-10-19: live capture from the running streams
- **What was added:**
  - XIMEA `get_frame()` leaves the frame's timing in `last_frame_meta` (block id, camera timestamp,
    host time). `get_frame()` itself is unchanged.
  - `PicoScope4000A.sample_at_host_time()` gives a coarse host clock from (host time, samples written)
    at every driver callback.
  - `CameraAdapterBase.start_recording()/stop_recording()` keep every frame between the calls: blinks
    included, the stream never paused, capped (default 10 s / 1 GB). Recording stops early if the ROI
    changes.
  - `alignment/live_capture.py: LiveCapture.capture(duration_s)` returns arrays ready for
    `evaluate_capture`.
- **Sync:** the host clocks place frame 0 to a few ms. The first capture fits the exact offset
  optically over +-0.1 s and keeps the correction; later captures search +-15 ms around it. The
  correction is dropped when the scope stream restarts or the ROI or exposure changes.
- **Measured** (XIMEA ROI 720x546, 8.8 ms exposure, 113 fps; scope B + D at 20 kHz):
  - 2 s recording: 227 frames, no dropped frames, host-vs-camera latency spread 1.6 ms.
  - Three 3 s captures: correction +1.6, +1.3, +1.1 ms (about 0.15 frame), fit depth 41-58 (locked).
    Each capture costs 3.7 s of wall time.
  - Metrics valid: intensity 0.62-0.65 V, Gaussian overlap 0.98, ellipticity 0.83-0.84, centre
    stable to ~1 px.
  - Drift, 12 captures over 3 min: the scope clock runs -13 ppm against the host clock (-2.4 ms in
    3 min). The correction stayed +1.50 +- 0.22 ms with no trend, because both device clocks are re-tied
    to the host clock in every capture and every capture is re-fitted. Scope driver callbacks arrive in
    bursts (~60 ms spread); the min-latency estimate ignores that.
  - The ramp that day: 0.05 Hz, 1.5 Vpp (aux -0.76..+0.76 V, 0.151 V/s). `scan_direction` matched
    every segment of a 25 s capture.
- **Synced captures from the streams (proposal, not built; full handover in `LIVE_CAPTURE.md`):**
  kalishlot's "Start Capture" could use
  the same mechanism: arm the recorder, cut the scope span, fit the sync, and write the session in the
  existing format, without borrowing the devices. Differences from today's capture:
  - the streams run at the boxes' settings: scope at its streaming rate (20 kHz now; the capture uses
    100 kS/s blocks), camera at its ROI and binning (no locate step, no auto-range probe);
  - the live stream hands over the newest frame, so under load a frame can be skipped (visible in
    the block ids; none in 2 s at 113 fps);
  - frames sit in RAM until written (~0.8 MB each at 720x546 Mono10), and the scope span must still
    be in the ~70 s ring.
- **Open:**
  - `evaluate_capture` took 3-11 s per capture (camera metrics), now longer than the capture itself.
    Speed it up, e.g. score only the frames around the 0th-order peaks.
  - One of the three captures scattered (overlap 0.93 +- 0.07, centre y +- 61 px, n=4). Not yet
    understood; maybe a sweep segment near a turnaround.
  - `first_order_s` came out -0.36 on up-sweeps and +0.31 on down-sweeps. Data for the "which side
    family" open question.

### Not done yet
- **Nested loop**: a symmetry metric that works in the
  lab (section 3.e, known limits); a first nested run on real axes.
- **Stage alignment in the lab** (section 3.f): a driver for the large mirror's stage, a spot measurement
  (the camera behind the large mirror), and a check of the large mirror's repeatability, which sets the
  accuracy.
- **M-LOOP backend**: not written. M-LOOP is not installed here and pulls heavy dependencies; the
  stage/objective structure of `AlignmentLoop` is where a GP backend would plug in.
- **Lab measurement function** for the loop: wrap `LiveCapture` as `measure()` (wait for the axes'
  `ready_at`, capture, evaluate).
- **Aligner box** in kalishlot: a virtual adapter running the loop in the server process, on the
  scope and XIMEA adapters' streams and the motion adapter's axes.
- Real drivers (Standa, PI E-872, Kinesis) and per-axis characterization in the lab (MDT693B included):
  `scan_axis` up and down for hysteresis, repeats for repeatability, delays for the vibration settle.
