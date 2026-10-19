"""Run the alignment loop against the simulated cavity (laptop, no hardware).

Misaligns the simulated axes of motion/axes_config.py, then lets AlignmentLoop
recover. A shared fake clock drives the axes' ramps/settle delays and the
simulator, so nothing really sleeps; the physics is cavity_design's when the
cavity-design project is reachable (else the analytic stand-in).

Run from PyCharm (edit the globals) or:  python alignment/run_alignment_sim.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alignment.loop import AlignmentLoop, Objective  # noqa: E402
from alignment.metrics import MetricsConfig, evaluate_capture  # noqa: E402
from motion.axes_config import AXES, TIERS  # noqa: E402
from motion.safety import SafeAxis  # noqa: E402
from motion.simulated import SimulatedAxis  # noqa: E402
from simulation.cavity_physics import make_physics  # noqa: E402
from simulation.setup import SimulatedSetup  # noqa: E402

MISALIGN = {'sim_mirror_phi': +12.0, 'sim_mirror_theta': -9.0, 'sim_lens_x': 0.0}
OBJECTIVE = Objective('intensity', maximize=True)
WITH_FRAMES = False          # camera metrics cost ~3 s of analysis per capture
FROZEN = []                  # e.g. ['sim_lens_x'] to keep it fixed
USE_CAVITY_DESIGN = True
MAX_EVALUATIONS = 120
SEED = 3
LOG_DIR = None               # e.g. a folder: writes evaluations.jsonl there
FIRST_ORDER_S = 'aligned'    # known first-order offset for mode identification: 'aligned' (from
                             # the physics, like a value measured in the lab), a number, or None


class FakeClock:
    def __init__(self, t=1000.0):
        self.t = t

    def now(self):
        return self.t

    def sleep(self, dt):
        self.t += max(dt, 0.0)


def build(clock, seed=SEED, physics=None):
    axes = {}
    for k, (name, entry) in enumerate(AXES.items()):
        if entry['driver'] != 'simulated':
            continue
        raw = SimulatedAxis(name, now=clock.now, seed=seed + k, **entry['options'])
        axes[name] = SafeAxis(raw, entry['limits'], sleep=clock.sleep, now=clock.now)
        axes[name].open()
    setup = SimulatedSetup.from_config(axes, physics=physics or make_physics(USE_CAVITY_DESIGN),
                                       now=clock.now, seed=seed)
    return axes, setup


def make_measure(setup, clock, with_frames=WITH_FRAMES, duration=5.6, first_order_s=FIRST_ORDER_S):
    if first_order_s == 'aligned':
        first_order_s = setup.aligned_first_order_s()
    cfg = MetricsConfig(saturation_v=0.995 * setup.scan.range_v, first_order_s=first_order_s)

    def measure():
        cap = setup.capture(duration=duration, with_frames=with_frames)
        clock.sleep(duration)                      # the capture itself takes time
        return evaluate_capture(cap['t'], cap['signal'], cap['aux'], cap.get('frames'),
                                cap.get('windows'), cap.get('mask'), cfg)
    return measure


def main():
    clock = FakeClock()
    axes, setup = build(clock)
    for name, delta in MISALIGN.items():
        if delta:
            axes[name].move_by(delta, who='misalign')
    for name in FROZEN:
        axes[name].set_frozen(True, who='run_alignment_sim')
    before = setup.mode_state()
    print(f'misaligned: coupling {before.coupling:.3f} (truth)')

    loop = AlignmentLoop(axes, make_measure(setup, clock), OBJECTIVE, tiers=TIERS,
                         max_evaluations=MAX_EVALUATIONS, log_dir=LOG_DIR)
    t_sim0 = clock.t
    summary = loop.run()
    after = setup.mode_state()
    print(f'\nresult: coupling {before.coupling:.3f} -> {after.coupling:.3f} (truth), '
          f'{summary["evaluations"]} evaluations, {(clock.t - t_sim0) / 60:.1f} simulated minutes')
    print('positions:', {n: round(v, 3) for n, v in summary['best'].items()})
    print('true positions:', {n: round(v, 3) for n, v in setup.true_positions().items()})
    return summary, before, after


if __name__ == '__main__':
    main()
