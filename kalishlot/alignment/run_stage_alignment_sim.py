"""Align the cavity's optical axis to the linear stage under the large mirror (simulator).

The cavity is cavity_design's example with the small mirror and the lens as one
rigid body, the short arm (simulation/cavity_physics.py: short_arm_cavity). The
large mirror sits on a linear stage whose mechanical axis is tilted from the
optical axis by STAGE_TILT_URAD, which the loop does not know.

Why the drift measures it: every line through the large mirror's centre of
curvature meets the mirror at normal incidence. If the stage moves that centre
along the optical axis, the same axis still closes on itself: only the NA
changes, and the mode stays where it was on the mirror. If the stage moves at an
angle to the axis, the axis turns and the mode's spot on the mirror moves. The
spot here is where the long arm's central line meets the large mirror,
relative to the mirror's centre.

One outer iteration (alignment/nested.py, alignment/procedures.py):
1) turn the short arm (outer level: sim_arm_phi, sim_arm_theta);
2) re-tilt the large mirror until the first-order splitting is smallest (inner
   level: sim_mirror_phi, sim_mirror_theta);
3) record the spot on the mirror, move the stage by STAGE_TRAVEL_MM, record it
   again, move the stage back (the procedure move_and_compare); the drift
   (dy, dz) is the outer target, goal (0, 0);
4) choose the next short-arm orientation from the drift, repeat.

OUTER picks how the outer level uses the drift:
- 'vector': VectorTarget on (dy, dz). The loop probes each short-arm axis once,
  then steers straight at zero drift with a Jacobian (2x2 here).
- 'scalar': minimize |drift| with the usual one-axis-at-a-time search.
- 'both': run both from the same start and compare.

The measurements are the truth plus noise: |split| in ppm of the FSR
(SPLIT_NOISE_PPM per capture) and the spot in um (SPOT_NOISE_UM). The scope's
split metric reads 0 in the simulator (MECHANICAL_CONTROL.md, section 3.e), so
the scope and camera are not simulated here.

Run from PyCharm (edit the globals) or:  python alignment/run_stage_alignment_sim.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alignment.loop import Objective  # noqa: E402
from alignment.metrics import CaptureMetrics, Metric  # noqa: E402
from alignment.nested import Level, NestedLoop  # noqa: E402
from alignment.procedures import move_and_compare  # noqa: E402
from alignment.run_alignment_sim import FakeClock  # noqa: E402
from alignment.targets import VectorTarget  # noqa: E402
from motion.safety import AxisLimits, SafeAxis  # noqa: E402
from motion.simulated import SimulatedAxis  # noqa: E402
from simulation.cavity_physics import CavityPhysics, short_arm_cavity, stage_dofs  # noqa: E402

OUTER = 'both'                     # 'vector', 'scalar' or 'both'
STAGE_TILT_URAD = (150.0, -80.0)   # hidden: stage axis vs the aligned optical axis (y, z)
STAGE_TRAVEL_MM = -10.0            # towards the lens (away from it there is no mode)
ARM_START_URAD = (0.0, 0.0)        # short-arm orientation at the start (phi, theta)
ARM_STEP_URAD = 40.0               # the axis follows the arm ~1:1, and the drift is ~0.13 um per urad of it
ARM_PROBE_URAD = 10.0              # first small trial per arm axis: learns how the mirror follows (predictor)
MIRROR_STEP_URAD = 2.0
DRIFT_TOLERANCE_UM = 1.5           # stop when the mean drift is this small
SPLIT_NOISE_PPM = 2.0              # per capture, ppm of the FSR
SPOT_NOISE_UM = 1.0                # per capture, each of y and z
CAPTURE_S = 1.0
MAX_EVALUATIONS = 400
SEED = 1

ARM = ['sim_arm_phi', 'sim_arm_theta']
MIRROR = ['sim_mirror_phi', 'sim_mirror_theta']
STAGE = 'sim_stage'


def make_physics(tilt=STAGE_TILT_URAD):
    dofs = stage_dofs(*tilt)
    return CavityPhysics(cavity_factory=short_arm_cavity, dofs=dofs, aligned={n: 0.0 for n in dofs},
                         cache_decimals=9)


def build(clock, seed=SEED, arm_start=ARM_START_URAD):
    tilt = dict(units='urad', kind='voltage', has_readback=True, repeatability_sigma=0.2, now=clock.now)
    raw = {ARM[0]: SimulatedAxis(ARM[0], start=arm_start[0], seed=seed, **tilt),
           ARM[1]: SimulatedAxis(ARM[1], start=arm_start[1], seed=seed + 1, **tilt),
           MIRROR[0]: SimulatedAxis(MIRROR[0], start=0.0, seed=seed + 2, **tilt),
           MIRROR[1]: SimulatedAxis(MIRROR[1], start=0.0, seed=seed + 3, **tilt),
           STAGE: SimulatedAxis(STAGE, units='mm', kind='position', start=0.0, closed_loop=True,
                                repeatability_sigma=0.002, sensor_noise=0.0005, now=clock.now, seed=seed + 4)}
    tilt_limits = AxisLimits(min=-2000, max=2000, max_step=50, max_slew=500, vibration_settle_s=0.5,
                             repeatability_sigma=0.2)
    limits = {ARM[0]: tilt_limits, ARM[1]: tilt_limits, MIRROR[0]: tilt_limits, MIRROR[1]: tilt_limits,
              STAGE: AxisLimits(min=-25, max=0.5, max_step=15, max_slew=5, vibration_settle_s=1.0,
                                repeatability_sigma=0.002)}
    axes = {n: SafeAxis(raw[n], limits[n], sleep=clock.sleep, now=clock.now) for n in raw}
    for a in axes.values():
        a.open()
    return axes


def true_positions(axes):
    return {n: a._raw.true_position for n, a in axes.items()}


def make_measure(axes, physics, clock, seed=SEED):
    rng = np.random.default_rng(seed)

    def measure():
        clock.sleep(CAPTURE_S)
        st = physics.evaluate(true_positions(axes))
        if not st.valid:
            return CaptureMetrics(metrics={m: Metric(m, reason='no mode') for m in ('split', 'spot_y', 'spot_z')})
        return CaptureMetrics(metrics={
            'split': Metric('split', abs(st.split) * 1e6 + rng.normal(0, SPLIT_NOISE_PPM), SPLIT_NOISE_PPM, 1, True),
            'spot_y': Metric('spot_y', st.spot_y + rng.normal(0, SPOT_NOISE_UM), SPOT_NOISE_UM, 1, True),
            'spot_z': Metric('spot_z', st.spot_z + rng.normal(0, SPOT_NOISE_UM), SPOT_NOISE_UM, 1, True)})
    return measure


def axis_vs_stage_urad(physics, positions, tilt=STAGE_TILT_URAD):
    """True angle (y, z) in urad between the long arm's optical axis and the stage axis."""
    pc = physics.perturbed_cavity(positions)
    end = pc.elements[-1]
    for arm in pc.arms:
        if arm.surface_1 is end:
            k = np.asarray(arm.central_line.k_vector, dtype=float)
            return k[1] / k[0] * 1e6 - tilt[0], k[2] / k[0] * 1e6 - tilt[1]
    return float('nan'), float('nan')


def true_drift_um(physics, positions, travel=STAGE_TRAVEL_MM):
    a = physics.evaluate(positions)
    b = physics.evaluate({**positions, STAGE: positions.get(STAGE, 0.0) + travel})
    return b.spot_y - a.spot_y, b.spot_z - a.spot_z


def levels_for(outer):
    if outer == 'vector':
        target = VectorTarget(('drift_spot_y', 'drift_spot_z'), (0.0, 0.0), tolerance=DRIFT_TOLERANCE_UM,
                              label='no drift', unit='um')
    else:
        target = Objective('drift', maximize=False, good_enough=DRIFT_TOLERANCE_UM, confirm_repeats=2)
    return [Level(ARM, target, steps={n: ARM_STEP_URAD for n in ARM}, probe_steps={n: ARM_PROBE_URAD for n in ARM},
                  measure=move_and_compare(STAGE, STAGE_TRAVEL_MM)),
            Level(MIRROR, Objective('split', maximize=False), steps={n: MIRROR_STEP_URAD for n in MIRROR})]


def run(outer, seed=SEED, print_fn=print, **loop_kw):
    clock = FakeClock()
    physics = make_physics()
    axes = build(clock, seed)
    loop = NestedLoop(axes, make_measure(axes, physics, clock, seed), levels_for(outer),
                      max_evaluations=MAX_EVALUATIONS, print_fn=print_fn, **loop_kw)
    start = true_positions(axes)
    summary = loop.run()
    end = true_positions(axes)
    result = {'outer': outer, 'summary': summary, 'loop': loop,
              'angle_start': axis_vs_stage_urad(physics, start), 'angle_end': axis_vs_stage_urad(physics, end),
              'drift_start': true_drift_um(physics, start), 'drift_end': true_drift_um(physics, end),
              'split_end_ppm': abs(physics.evaluate(end).split) * 1e6,
              'stage_end_mm': end[STAGE], 'evaluations': summary['evaluations'],
              'outer_trials': len(loop.profile[0]), 'hours': (clock.t - 1000.0) / 3600}
    return result


def report(r):
    print(f'\n=== outer = {r["outer"]} ===')
    print(f'axis vs stage [urad]  start ({r["angle_start"][0]:+.1f}, {r["angle_start"][1]:+.1f})'
          f'  end ({r["angle_end"][0]:+.1f}, {r["angle_end"][1]:+.1f})')
    print(f'true drift [um]       start ({r["drift_start"][0]:+.1f}, {r["drift_start"][1]:+.1f})'
          f'  end ({r["drift_end"][0]:+.2f}, {r["drift_end"][1]:+.2f})')
    print(f'splitting at the end  {r["split_end_ppm"]:.1f} ppm FSR;  stage back at {r["stage_end_mm"]:+.4f} mm')
    print(f'{r["evaluations"]} captures, {r["outer_trials"]} outer trials, {r["hours"]:.2f} h of simulated time, '
          f'target reached: {r["summary"]["target_reached"]}')


def main():
    for outer in (['vector', 'scalar'] if OUTER == 'both' else [OUTER]):
        report(run(outer))


if __name__ == '__main__':
    main()
