"""Animate the alignment loop aligning a simulated cavity whose misalignment it cannot see.

The cavity is cavity_design's (simulation/cavity_physics.py). Its true aligned
point is moved by HIDDEN_MISALIGNMENT, which only the simulator knows; the
motors start at their nominal positions. AlignmentLoop gets nothing but
synthetic scope captures (and the metrics computed from them), exactly as it
would in the lab. Afterwards, every evaluation is replayed with the hidden
truth next to what the optimizer saw:

    top:    Cavity.plot of the aligned (red) vs the current (blue) cavity, full and zoomed;
            plot_2_cavity_perturbation_overlap(aligned, current)
    bottom: the scope trace the optimizer got; measured intensity vs the hidden
            coupling; the path over the hidden coupling landscape (tilt plane)

Writes OUTPUT_DIR/alignment_sim.gif and alignment_sim_final.png.
Run from PyCharm (edit the globals) or:  python alignment/visualize_alignment_sim.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

from alignment.loop import AlignmentLoop, Objective  # noqa: E402
from alignment.run_alignment_sim import FakeClock, build, make_measure  # noqa: E402
from motion.axes_config import TIERS  # noqa: E402
from simulation.cavity_physics import CavityPhysics  # noqa: E402

HIDDEN_MISALIGNMENT = {'sim_mirror_phi': 25.0, 'sim_mirror_theta': -18.0, 'sim_lens_x': 60.0}
OBJECTIVE = Objective('intensity', maximize=True)
MAX_EVALUATIONS = 120
SEED = 4
GRID_POINTS = 31                     # hidden coupling landscape resolution (per axis)
TRACE_WINDOW_S = 1.2                 # how much of each capture to show
FPS = 4
OUTPUT_DIR = Path(__file__).resolve().parent / 'figures'

TX, TY, LX = 'sim_mirror_phi', 'sim_mirror_theta', 'sim_lens_x'


def _safe_plot(cavity, **kw):
    """Cavity.plot draws the modes first and the mirror origin/image markers last;
    for this cavity the last step raises (central line continuation), so the
    markers are skipped and the modes kept."""
    try:
        cavity.plot(**kw)
    except ValueError:
        pass


def run():
    physics = CavityPhysics()            # puts cavity_design on the path
    nominal = dict(physics.aligned)
    physics.aligned = {n: v + HIDDEN_MISALIGNMENT.get(n, 0.0) for n, v in nominal.items()}

    clock = FakeClock()
    axes, setup = build(clock, seed=SEED, physics=physics)   # motors start at nominal
    inner = make_measure(setup, clock)
    record = []

    def measure():                    # the optimizer sees only cm; the truth goes to `record`
        cm = inner()
        cap = setup.last_capture
        m = cap['t'] < TRACE_WINDOW_S
        record.append({'true': setup.true_positions(), 'state': setup.mode_state(),
                       't': cap['t'][m][::4], 'signal': cap['signal'][m][::4]})
        return cm

    loop = AlignmentLoop(axes, measure, OBJECTIVE, tiers=TIERS, max_evaluations=MAX_EVALUATIONS)
    summary = loop.run()
    return physics, nominal, loop, record, summary


def landscape(physics, nominal):
    """Hidden coupling over the tilt plane, lens at its (hidden) aligned position."""
    span = 45.0
    tx = nominal[TX] + np.linspace(-span, span, GRID_POINTS) + HIDDEN_MISALIGNMENT[TX] * 0.5
    ty = nominal[TY] + np.linspace(-span, span, GRID_POINTS) + HIDDEN_MISALIGNMENT[TY] * 0.5
    c = np.full((ty.size, tx.size), np.nan)
    for i, y in enumerate(ty):
        for j, x in enumerate(tx):
            st = physics.evaluate({TX: x, TY: y, LX: physics.aligned[LX]})
            c[i, j] = st.coupling if st.valid else np.nan
    return tx, ty, c


def animate(physics, nominal, loop, record, summary):
    import cavity_design as cd
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    hist = loop.history
    n = min(len(hist), len(record))
    measured = np.array([h.score if h.score is not None else np.nan for h in hist[:n]])
    truth = np.array([r['state'].coupling if r['state'].valid else 0.0 for r in record[:n]])
    best_so_far = np.fmax.accumulate(np.nan_to_num(measured, nan=-np.inf))
    tx, ty, land = landscape(physics, nominal)
    path = np.array([[r['true'][TX], r['true'][TY]] for r in record[:n]])
    aligned = physics.aligned
    ymax = max(max(r['signal'].max() for r in record[:n]) * 1.1, 0.1)

    fig, ax = plt.subplots(2, 3, figsize=(17, 9.5))
    fig.subplots_adjust(hspace=0.45, wspace=0.38, top=0.88)
    tw = ax[1, 1].twinx()                # created once; cleared with the others every frame

    def draw(k):
        r, h = record[k], hist[k]
        for a in list(ax.flat) + [tw]:
            a.clear()
        current = physics.perturbed_cavity(r['true'])
        _safe_plot(physics.reference, ax=ax[0, 0])
        _safe_plot(current, ax=ax[0, 0], laser_color='b')
        ax[0, 0].set_title('modes, xy plane (optical axis x): aligned (red) vs current (blue)', fontsize=10)
        _safe_plot(physics.reference, ax=ax[0, 1])
        _safe_plot(current, ax=ax[0, 1], laser_color='b')
        ax[0, 1].set_xlim(-0.012, 0.006)
        ax[0, 1].set_ylim(-3e-4, 3e-4)
        ax[0, 1].set_title('zoom at the small mirror', fontsize=10)
        ax[0, 1].locator_params(axis='x', nbins=4)
        try:
            cd.plot_2_cavity_perturbation_overlap(cavity=physics.reference, second_cavity=current,
                                                 real_or_abs='abs', ax=ax[0, 2])
        except Exception as e:
            ax[0, 2].text(0.5, 0.5, f'no stable mode\n({type(e).__name__})', ha='center',
                          transform=ax[0, 2].transAxes)

        ax[1, 0].plot(r['t'], r['signal'], lw=0.6)
        ax[1, 0].set_ylim(0, ymax)
        ax[1, 0].set_xlabel('time in capture [s]')
        ax[1, 0].set_ylabel('photodiode [V]')
        ax[1, 0].set_title('what the optimizer sees (scope)', fontsize=10)

        ev = np.arange(n)
        ax[1, 1].plot(ev[:k + 1], measured[:k + 1], 'o', ms=3, color='C0', alpha=0.6,
                      label='measured intensity')
        ax[1, 1].plot(ev[:k + 1], best_so_far[:k + 1], '-', color='C0', label='incumbent')
        ax[1, 1].set_xlim(0, n)
        ax[1, 1].set_ylim(np.nanmin(measured) * 0.95, np.nanmax(measured) * 1.03)
        ax[1, 1].set_xlabel('evaluation')
        ax[1, 1].set_ylabel('intensity [V]', color='C0')
        tw.plot(ev[:k + 1], truth[:k + 1], '-', color='C3', lw=1, label='hidden coupling')
        tw.set_ylim(min(truth.min(), 0.8), 1.005)
        tw.yaxis.tick_right()                # clear() resets a twin axis to the left side
        tw.yaxis.set_label_position('right')
        tw.set_ylabel('hidden true coupling', color='C3')
        ax[1, 1].set_title('measured (optimizer) vs truth (hidden)', fontsize=10)

        ext = [tx[0] - nominal[TX], tx[-1] - nominal[TX], ty[0] - nominal[TY], ty[-1] - nominal[TY]]
        im = ax[1, 2].imshow(land, origin='lower', extent=ext, aspect='auto', cmap='viridis',
                             vmin=np.nanmin(land), vmax=1)
        ax[1, 2].contour(tx - nominal[TX], ty - nominal[TY], land, levels=[0.9, 0.95, 0.99],
                         colors='w', linewidths=0.6)
        ax[1, 2].plot(path[:k + 1, 0] - nominal[TX], path[:k + 1, 1] - nominal[TY], '.-',
                      color='orange', lw=0.8, ms=3)
        ax[1, 2].plot(path[k, 0] - nominal[TX], path[k, 1] - nominal[TY], 'o', color='red', ms=6)
        ax[1, 2].plot(aligned[TX] - nominal[TX], aligned[TY] - nominal[TY], '*', color='white',
                      ms=14, mec='k')
        ax[1, 2].set_xlabel('end mirror phi (xy tilt) from nominal [urad]')
        ax[1, 2].set_ylabel('end mirror theta (xz tilt) from nominal [urad]')
        ax[1, 2].set_title('hidden coupling, lens at its optimum (star = hidden optimum)', fontsize=10)

        st = r['state']
        off = {nm: r['true'][nm] - aligned[nm] for nm in (TX, TY, LX)}
        fig.suptitle(
            f'evaluation {k}/{n - 1} ({h.kind})   hidden coupling {st.coupling if st.valid else 0:.4f}   '
            f'remaining misalignment: phi {off[TX]:+.1f} urad, theta {off[TY]:+.1f} urad, '
            f'lens {off[LX]:+.0f} um (stage)', fontsize=12)
        return []

    anim = FuncAnimation(fig, draw, frames=n, blit=False)
    gif = OUTPUT_DIR / 'alignment_sim.gif'
    anim.save(gif, writer=PillowWriter(fps=FPS), dpi=60)
    draw(n - 1)
    png = OUTPUT_DIR / 'alignment_sim_final.png'
    fig.savefig(png, dpi=90)
    plt.close(fig)
    return gif, png


def main():
    physics, nominal, loop, record, summary = run()
    start, end = record[0]['state'], record[-1]['state']
    print(f'hidden coupling {start.coupling:.4f} -> {end.coupling:.4f} '
          f'in {summary["evaluations"]} evaluations')
    gif, png = animate(physics, nominal, loop, record, summary)
    print(f'wrote {gif}\nwrote {png}')


if __name__ == '__main__':
    main()
