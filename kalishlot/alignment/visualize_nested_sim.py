"""Animate the nested loop (alignment/nested.py) walking a hidden lens error out of a cavity.

The cavity is cavity_design's example (simulation/cavity_physics.py) with the large
end mirror 1 mm closer to the lens, traced exactly (paraxial=False). Two knobs, both
in the xy plane: the end mirror's tilt and the lens's sideways position.

What the physics does (MECHANICAL_CONTROL.md, section 3.e): a mirror tilt of 53.3 urad
per um of lens shift keeps the mode almost unchanged, so a mispositioned lens can be
hidden by a tilted mirror. Along that compensating line the coupling barely changes
(0.996 with the lens 200 um off), but the mode becomes astigmatic: the first-order
peak splits (1.9 peak widths at 200 um). Across the line the mode is lost within
~1.5 um of lens or ~70 urad of tilt.

The start: lens 200 um off, mirror tilted to compensate. ROLES picks who does what:
- 'tilt_symmetry' (works): inner = lens position maximizing the coupling (sharp across
  the line), outer = mirror tilt minimizing the splitting (strong signal along it).
- 'tilt_coupling': inner = lens position minimizing the splitting, outer = mirror tilt
  maximizing the coupling. Fails: the coupling gains only ~0.4% along the whole line,
  less than how much it varies between inner runs, and the least-split lens position
  is off the line, so the inner level trades coupling for symmetry.

The symmetry is measured as the true splitting (from the physics) in peak widths plus
noise (ASYMMETRY_NOISE): the scope's split metric reads 0 in the simulator when one
first-order mode dominates (section 3.e).

Panels: the true coupling over (mirror tilt, lens offset from the compensating line)
with lines of equal splitting and every capture; the outer and inner targets per
capture, measured and true; the predictor K = how far the lens optimum moves per mrad
of tilt, against the true slope; and what the loop is doing at that capture.

Writes OUTPUT_DIR/nested_<roles>.gif and nested_<roles>_final.png per entry of ROLES.
Run from PyCharm (edit the globals) or:  python alignment/visualize_nested_sim.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

LIVE = False                       # True: show the animation in a window instead of only saving it
import matplotlib  # noqa: E402
if not LIVE:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

from alignment.loop import Objective  # noqa: E402
from alignment.metrics import Metric  # noqa: E402
from alignment.nested import Level, NestedLoop  # noqa: E402
from alignment.run_alignment_sim import FakeClock, make_measure  # noqa: E402
from motion.safety import AxisLimits, SafeAxis  # noqa: E402
from motion.simulated import SimulatedAxis  # noqa: E402
from simulation.cavity_physics import CavityPhysics, DoF, example_cavity  # noqa: E402
from simulation.setup import SimulatedSetup  # noqa: E402

ROLES = ['tilt_symmetry', 'tilt_coupling']
END_MIRROR_SHIFT = 1e-3            # m towards the lens, away from the concentric limit
LENS_START_UM = -200.0             # the hidden lens error
SLOPE_URAD_PER_UM = 53.3           # compensating mirror tilt per um of lens shift (measured, section 3.e)
TILT_STEP_URAD = 1000.0            # outer step, after one probe step of TILT_PROBE_URAD
TILT_PROBE_URAD = 30.0
LENS_STEP_UM = 0.2
ASYMMETRY_NOISE = 0.05             # peak widths, per capture
MAX_EVALUATIONS = 200
SEED = 5                           # 5 seeds: lens ends at -181, -13, -157, -48, -1 um (seed 1..5; section 3.e)
GRID = (49, 25)                    # coupling map resolution (tilt, lens offset)
LENS_OFFSET_RANGE_UM = 1.5         # map: lens offset from the line, +-
FPS = 6
OUTPUT_DIR = Path(__file__).resolve().parent / 'figures'

T, L = 'mirror_tilt', 'lens_y'
OUTER_STEP_KINDS = ('trial', 'probe', 'parabola', 'gradient')


def make_physics():
    return CavityPhysics(cavity_factory=lambda: example_cavity(END_MIRROR_SHIFT, paraxial=False),
                         dofs={T: DoF(3, 'phi', 1e-6), L: DoF(1, 'y', 1e-6)},   # urad, um
                         aligned={T: 0.0, L: 0.0}, cache_decimals=9)


def build(physics, clock):
    raw = {T: SimulatedAxis(T, units='urad', kind='voltage', start=-SLOPE_URAD_PER_UM * LENS_START_UM,
                            has_readback=True, repeatability_sigma=0.5, now=clock.now, seed=SEED),
           L: SimulatedAxis(L, units='um', kind='position', start=LENS_START_UM, closed_loop=True,
                            repeatability_sigma=0.02, sensor_noise=0.005, now=clock.now, seed=SEED + 1)}
    limits = {T: AxisLimits(min=-3000, max=20000, max_step=500, max_slew=5000, vibration_settle_s=0.5,
                            repeatability_sigma=0.5),
              L: AxisLimits(min=-400, max=100, max_step=10, max_slew=50, vibration_settle_s=0.5,
                            repeatability_sigma=0.02)}
    axes = {n: SafeAxis(raw[n], limits[n], sleep=clock.sleep, now=clock.now) for n in raw}
    for a in axes.values():
        a.open()
    return axes, SimulatedSetup.from_config(axes, physics=physics, now=clock.now, seed=SEED)


def split_widths(state, setup):
    """|first-order splitting| in peak widths (FWHM); below ~1 the two parts cannot be told apart."""
    return abs(state.split) / (2 * setup.scan.linewidth_fsr)


def levels_for(roles):
    coupling, symmetry = Objective('intensity'), Objective('asymmetry', maximize=False)
    outer, inner = (symmetry, coupling) if roles == 'tilt_symmetry' else (coupling, symmetry)
    return [Level([T], outer, steps={T: TILT_STEP_URAD}, probe_steps={T: TILT_PROBE_URAD}),
            Level([L], inner, steps={L: LENS_STEP_UM})]


def run(physics, roles):
    clock = FakeClock()
    axes, setup = build(physics, clock)
    inner_measure = make_measure(setup, clock)
    rng = np.random.default_rng(SEED)
    truth = []                                     # one per capture, in the order of loop.history

    def measure():
        cm = inner_measure()
        st = setup.mode_state()
        truth.append({'coupling': st.coupling if st.valid else 0.0,
                      'split': split_widths(st, setup) if st.valid else np.nan,
                      **setup.true_positions()})
        cm.metrics['asymmetry'] = (Metric('asymmetry', split_widths(st, setup) + abs(rng.normal(0, ASYMMETRY_NOISE)),
                                          ASYMMETRY_NOISE, 10, True)
                                   if st.valid else Metric('asymmetry', reason='no mode'))
        return cm

    start = setup.mode_state()
    loop = NestedLoop(axes, measure, levels_for(roles), max_evaluations=MAX_EVALUATIONS, print_fn=None)
    summary = loop.run()
    end = setup.mode_state()
    print(f'{roles}: coupling {start.coupling:.4f} -> {end.coupling:.4f}, splitting '
          f'{split_widths(start, setup):.2f} -> {split_widths(end, setup):.2f} peak widths, lens '
          f'{LENS_START_UM:.0f} -> {setup.true_positions()[L]:.1f} um, {summary["evaluations"]} captures, '
          f'K {summary["predictor"].get(0, {}).get("K")}')
    return loop, truth, setup, summary


def offset_from_line(tilt, lens):
    """Lens position minus the position the tilt compensates [um]."""
    return lens + tilt / SLOPE_URAD_PER_UM


def coupling_map(physics):
    tilts = np.linspace(-500, -SLOPE_URAD_PER_UM * LENS_START_UM + 1000, GRID[0])
    offs = np.linspace(-LENS_OFFSET_RANGE_UM, LENS_OFFSET_RANGE_UM, GRID[1])
    C = np.zeros((offs.size, tilts.size))
    S = np.full_like(C, np.nan)
    for j, t in enumerate(tilts):
        for i, v in enumerate(offs):
            st = physics.evaluate({T: t, L: v - t / SLOPE_URAD_PER_UM})
            if st.valid:
                C[i, j] = st.coupling
                S[i, j] = abs(st.split) / (2 * 0.0022)
    return tilts, offs, C, S


def describe(loop, i, roles):
    """What the loop is doing at capture i, in words."""
    h = loop.history[i]
    outer_name = 'splitting' if roles == 'tilt_symmetry' else 'coupling'
    inner_name = 'coupling' if roles == 'tilt_symmetry' else 'splitting'
    kinds = [x['kind'] for x in loop.history]
    first_start = kinds.index('start') if 'start' in kinds else len(kinds)
    profile_at = kinds.index('profile noise') if 'profile noise' in kinds else -1
    if h['level'] is None:
        return 'Measuring the noise: 4 captures without moving anything.'
    trial = next((p for p in loop.profile[0] if p['first_capture'] <= i <= p['last_capture']), None)
    if trial is not None:
        d = trial['delta'][T]
        pre = trial['predicted'].get(L)
        lens_pre = (f'lens pre-moved {pre - trial["start"][L]:+.1f} um (predicted)' if pre is not None
                    else 'lens left in place (no prediction yet)')
        head = f'{trial["kind"].capitalize()} step of the mirror tilt: {d / 1000:+.3f} mrad,\n{lens_pre}.'
        if i < trial['last_capture'] - 1 or h['level'] == 1:
            return head + f'\nInner level: the lens re-optimizes the {inner_name}.'
        verdict = 'kept' if trial['accepted'] else 'rejected (not better by more than the threshold)'
        val = trial['metrics'].get('asymmetry' if roles == 'tilt_symmetry' else 'intensity')
        val = (f'{val:.2f} peak widths' if roles == 'tilt_symmetry' else f'{val / 1.6:.4f}') if val else 'no light'
        return head + f'\nOuter level: {outer_name} measured {val} -> {verdict}.'
    if i <= first_start:
        if h['level'] == 0:
            return f'Starting point: {outer_name} measured after the lens optimized the {inner_name} there.'
        return f'Starting point: inner level, the lens optimizes the {inner_name}.'
    if i <= profile_at:
        if h['level'] == 0:
            return (f'Same tilt, second inner run done: the {outer_name} differs from the first\n'
                    f'run by the "profile noise", which sets the threshold for the outer steps.')
        return (f'Same tilt again: the lens re-optimizes the {inner_name} from a shifted start,\n'
                f'to see how much the result varies between inner runs.')
    if h['kind'] == 'verify':
        return 'Finished: back at the best point, checking that it reproduces.'
    if h['kind'] in ('remeasure', 'confirm'):
        return 'Re-measuring the best point (drift check / confirming a large jump).'
    return 'This tilt step lost the mode (no light at all): back to the best point.'


def animate(physics, roles, grid):
    loop, truth, setup, summary = run(physics, roles)
    tilts, offs, C, S = grid
    hist = loop.history
    n = len(hist)
    tilt = np.array([t[T] for t in truth])
    off = offset_from_line(tilt, np.array([t[L] for t in truth]))
    cpl = np.array([t['coupling'] for t in truth])
    spl = np.array([t['split'] for t in truth])
    meas_c = np.array([(h['metrics'].get('intensity') or np.nan) / setup.scan.peak_v for h in hist])
    meas_s = np.array([h['metrics'].get('asymmetry') if h['metrics'].get('asymmetry') is not None else np.nan
                       for h in hist])
    level = [h['level'] for h in hist]
    trials = loop.profile[0]

    fig = plt.figure(figsize=(14, 8.4))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.35, 1], height_ratios=[1, 1, 1], hspace=0.55, wspace=0.3)
    ax_map = fig.add_subplot(gs[0:2, 0])
    ax_txt = fig.add_subplot(gs[2, 0])
    ax_c = fig.add_subplot(gs[0, 1])
    ax_s = fig.add_subplot(gs[1, 1], sharex=ax_c)
    ax_k = fig.add_subplot(gs[2, 1])
    outer_is_sym = roles == 'tilt_symmetry'
    fig.suptitle('Nested alignment: outer = mirror tilt -> ' + ('symmetry' if outer_is_sym else 'coupling')
                 + ',  inner = lens position -> ' + ('coupling' if outer_is_sym else 'symmetry')
                 + f'   (start: lens {LENS_START_UM:.0f} um off, hidden by the mirror tilt)', fontsize=12)

    im = ax_map.pcolormesh(tilts / 1000, offs, C, shading='auto', cmap='viridis', vmin=0, vmax=1)
    fig.colorbar(im, ax=ax_map, label='true coupling (0 = no mode)', pad=0.01)
    cs = ax_map.contour(tilts / 1000, offs, S, levels=[0.25, 0.5, 1, 1.5, 2], colors='w', linewidths=0.7)
    ax_map.clabel(cs, fmt='%.2g', fontsize=7)
    ax_map.axhline(0, color='w', lw=0.5, ls=':')
    ax_map.set_xlabel('mirror tilt [mrad]   (0 = aligned)')
    ax_map.set_ylabel('lens position minus the position\nthe tilt compensates [um]')
    ax_map.set_title('white lines: splitting of the first-order peak [peak widths]', fontsize=9)
    ax_map.set_ylim(offs[0], offs[-1])
    path_inner, = ax_map.plot([], [], '.', color='0.85', ms=3, label='capture (inner level)')
    kept, = ax_map.plot([], [], 'o', mfc='none', mec='lime', ms=8, mew=1.5, label='outer step kept')
    rejected, = ax_map.plot([], [], 'x', color='r', ms=7, label='outer step rejected')
    now, = ax_map.plot([], [], '*', color='orange', ms=16, mec='k', label='now')
    ax_map.legend(loc='lower left', fontsize=7, framealpha=0.6)

    true_lines = []
    for ax, ylabel in ((ax_c, 'coupling'), (ax_s, 'splitting [peak widths]')):
        true_lines.append(ax.plot([], [], '-', color='0.6', lw=1, label='true')[0])
        ax.set_xlim(-2, n + 1)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
    ax_c.set_ylim(max(0.9, np.nanmin(meas_c[np.isfinite(meas_c)]) - 0.01) if np.isfinite(meas_c).any() else 0.9, 1.003)
    ax_s.set_ylim(0, np.nanmax(spl) * 1.15 + 0.1)
    ax_s.set_xlabel('capture number')
    sc_c = ax_c.scatter([], [], s=9)
    sc_s = ax_s.scatter([], [], s=9)
    cur_c = ax_c.axvline(0, color='orange', lw=1)
    cur_s = ax_s.axvline(0, color='orange', lw=1)
    ax_c.set_title(('inner' if outer_is_sym else 'outer') + ' target   (dots: measured; grey: true)', fontsize=9)
    ax_s.set_title(('outer' if outer_is_sym else 'inner') + ' target', fontsize=9)

    true_k = -1000 / SLOPE_URAD_PER_UM
    ax_k.axhline(true_k, color='0.5', ls='--', lw=1, label=f'true slope of the line ({true_k:.2f})')
    ks = [(p['last_capture'], p['K'][0][0] * 1000) for p in trials if p['K'] is not None]
    k_line, = ax_k.plot([], [], 'o-', color='C1', ms=4, label='predictor K (learned)')
    ax_k.set_xlim(0, n)
    ax_k.set_ylim(min([true_k * 1.3] + [k for _, k in ks]) if ks else true_k * 1.3, 2)
    ax_k.set_xlabel('capture number')
    ax_k.set_ylabel('um per mrad')
    ax_k.set_title('predictor K: lens shift to pre-move per mrad of tilt', fontsize=9)
    ax_k.legend(fontsize=7, loc='upper right')
    ax_k.grid(alpha=0.3)
    ax_txt.axis('off')
    txt = ax_txt.text(0.0, 1.0, '', va='top', ha='left', fontsize=9, family='monospace', transform=ax_txt.transAxes)

    colors = np.array(['0.5' if lv is None else ('C0' if lv == 1 else 'C3') for lv in level])

    def draw(i):
        i = min(i, n - 1)
        m = np.arange(i + 1)
        valid = np.isfinite(off[m]) & (cpl[m] > 0)
        path_inner.set_data(tilt[m][valid] / 1000, off[m][valid])
        done = [p for p in trials if p['last_capture'] <= i]
        k_pts = [p for p in done if p['accepted']]
        r_pts = [p for p in done if not p['accepted']]
        kept.set_data([p['positions'][T] / 1000 for p in k_pts],
                      [offset_from_line(p['positions'][T], p['positions'][L]) for p in k_pts])
        rejected.set_data([p['positions'][T] / 1000 for p in r_pts],
                          [offset_from_line(p['positions'][T], p['positions'][L]) for p in r_pts])
        now.set_data([tilt[i] / 1000], [np.clip(off[i], offs[0], offs[-1])])
        for sc, meas, line, true in ((sc_c, meas_c, true_lines[0], cpl), (sc_s, meas_s, true_lines[1], spl)):
            sc.set_offsets(np.column_stack([m, meas[m]]))
            sc.set_color(colors[m])
            line.set_data(m, true[m])
        cur_c.set_xdata([i, i])
        cur_s.set_xdata([i, i])
        kk = [(c, k) for c, k in ks if c <= i]
        k_line.set_data([c for c, _ in kk], [k for _, k in kk])
        lv = {None: 'noise floor', 0: 'outer', 1: 'inner'}[level[i]]
        txt.set_text(f'capture {i + 1}/{n}   level: {lv}\n(dot colours on the right: red = outer, blue = inner)\n\n'
                     + describe(loop, i, roles)
                     + f'\n\ntrue now: coupling {cpl[i]:.4f}, splitting {spl[i]:.2f} peak widths,\n'
                       f'lens {truth[i][L]:.1f} um (0 = centred)')
        return path_inner, kept, rejected, now, sc_c, sc_s, cur_c, cur_s, k_line, txt

    frames = list(range(n)) + [n - 1] * (2 * FPS)
    anim = FuncAnimation(fig, draw, frames=frames, interval=1000 / FPS, blit=False)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if LIVE:
        plt.show()
        return
    gif = OUTPUT_DIR / f'nested_{roles}.gif'
    anim.save(gif, writer=PillowWriter(fps=FPS), dpi=65)
    draw(n - 1)
    png = OUTPUT_DIR / f'nested_{roles}_final.png'
    fig.savefig(png, dpi=100)
    plt.close(fig)
    print(f'wrote {gif} and {png}')


def main():
    physics = make_physics()
    print('computing the coupling map ...')
    grid = coupling_map(physics)
    for roles in ROLES:
        animate(physics, roles, grid)


if __name__ == '__main__':
    main()
