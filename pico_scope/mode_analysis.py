"""Pure math for resonator mode-spacing / NA extraction from scope traces.

Shared by the offline scripts (mode_spacing_extraction_sidebands.py, run on
saved PicoScope files) and the kalishlot web GUI (run live on the streaming
scope) so both produce identical numbers. No GUI, no file I/O, no prompts —
only numpy/scipy plus the (optional, cached) cavity-design NA interpolators.

    python mode_analysis.py    runs a synthetic-data self-test (no hardware,
                               no cavity-design project needed).
"""

import sys
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

MAX_FIT_POINTS = 500              # decimate fits to at most this many points
DEFAULT_SIDEBAND_FREQ_MHZ = 25.0  # single-side EOM modulation frequency [MHz]
SIDEBAND_AMP_RATIO_GUESS = 6.0    # r: main-peak / sideband-peak amplitude ratio
N_points = 200
LONG_ARM_LENGTH = 34.4e-2
MID_ARM_LENGTH = 1.5e-2
SHORT_ARM_LENGTHS = 4e-4          # the simulation's lens-scan parameter
SHORT_ARM_LENGTH = 0.7e-2         # the physical short arm: sets the cavity
                                  # length and with it the FSR
SPEED_OF_LIGHT = 299792458.0      # m / s

# The cavity's optical elements, named after the cavity-design catalog and listed in optical order.
# This is only the default for callers that do not pass their own (the kalishlot web GUI); the
# offline scripts define their own list in a config block at the top of the file.
# See list_cavity_elements() for the available names.
CAVITY_ELEMENTS = ['LASER_OPTIK_MIRROR',
                   'EDMUND_4p5MM_ASPHERIC_83580',
                   'COASTLINE_20CM_MIRROR']

SIX_LORENTZIAN_PARAMS = ['A0', 's0', 'x0', 'A1', 's1', 'x1', 'd', 'r', 'y0']
DOUBLE_LORENTZIAN_PARAMS = ['x01', 'gamma1', 'A1', 'x02', 'gamma2', 'A2', 'y0']


def cavity_fsr_mhz(long_arm=LONG_ARM_LENGTH, mid_arm=MID_ARM_LENGTH,
                   short_arm=SHORT_ARM_LENGTH):
    """FSR = c / (2 L) of the linear cavity, in MHz."""
    length = long_arm + mid_arm + short_arm
    return SPEED_OF_LIGHT / (2.0 * length) / 1e6


def decimate(x_arr, y_arr, max_points=MAX_FIT_POINTS):
    """Keep every n-th sample so that at most `max_points` points remain.

    PicoScope traces can hold ~100k points, far denser than the fit needs.
    A simple stride keeps the spectrum shape intact.
    """
    step = max(1, int(np.ceil(len(x_arr) / max_points)))
    return x_arr[::step], y_arr[::step]


def lorentzian(x, A, s, x0):
    """Peak value A, half-width-at-half-maximum s, centre x0."""
    return A / (1.0 + ((x - x0) / s) ** 2)


def six_lorentzian_model(x, A0, s0, x0, A1, s1, x1, d, r, y0):
    """0th- and 1st-order modes, each with two sidebands at +/- d, + offset."""
    return (
        lorentzian(x, A0, s0, x0)
        + lorentzian(x, A0 / r, s0, x0 - d)
        + lorentzian(x, A0 / r, s0, x0 + d)
        + lorentzian(x, A1, s1, x1)
        + lorentzian(x, A1 / r, s1, x1 - d)
        + lorentzian(x, A1 / r, s1, x1 + d)
        + y0
    )


def nearest_value(xs, ys, x0):
    return ys[int(np.argmin(np.abs(xs - x0)))]


def fit_six_lorentzians(x, y, x0_guess, x1_guess, d_guess,
                        r_guess=SIDEBAND_AMP_RATIO_GUESS, region=None):
    """Fit the 6-Lorentzian sideband model; return (params, errors) dicts.

    `x0_guess` / `x1_guess` are the clicked mode positions, `d_guess` the
    clicked sideband distance. `region` is the (min, max) of the selected
    window (defaults to the data's ends). Raises RuntimeError when the fit
    does not converge. Keys: A0, s0, x0, A1, s1, x1, d, r, y0.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if region is not None:
        region_min, region_max = float(region[0]), float(region[1])
    else:
        region_min, region_max = float(x[0]), float(x[-1])
    region_width = region_max - region_min

    y0_init = float(np.min(y))
    A0_init = max(nearest_value(x, y, x0_guess) - y0_init, 1e-9)
    A1_init = max(nearest_value(x, y, x1_guess) - y0_init, 1e-9)
    s_init = max(d_guess / 6.0, region_width / 200.0)

    p0 = [A0_init, s_init, x0_guess,
          A1_init, s_init, x1_guess,
          d_guess, r_guess, y0_init]

    # peaks => positive amplitudes; widths/distance positive
    eps = region_width / 1e6
    lower = [0.0,    eps,          region_min,
             0.0,    eps,          region_min,
             eps,    1.0,          -np.inf]
    upper = [np.inf, region_width, region_max,
             np.inf, region_width, region_max,
             region_width, np.inf, np.inf]

    popt, pcov = curve_fit(
        six_lorentzian_model, x, y,
        p0=p0, bounds=(lower, upper), maxfev=50000,
    )
    perr = np.sqrt(np.diag(pcov))
    return (dict(zip(SIX_LORENTZIAN_PARAMS, (float(v) for v in popt))),
            dict(zip(SIX_LORENTZIAN_PARAMS, (float(v) for v in perr))))


def sideband_results(x0, x1, d, s0, s1, f_sb_mhz, na_interp=None):
    """Scale the fitted geometry by the sideband frequency (the sidebands sit
    at +/- f_sb, i.e. +/- d in x-units) and look up the NA if an interpolator
    (mode spacing [Hz] -> NA) is given."""
    mode_spacing = abs(x1 - x0) / d * f_sb_mhz   # [MHz]
    linewidth_0 = s0 / d * f_sb_mhz              # [MHz] (HWHM)
    linewidth_1 = s1 / d * f_sb_mhz              # [MHz] (HWHM)
    na = float(na_interp(mode_spacing * 1e6)) if na_interp is not None else None
    if na is not None and not np.isfinite(na):
        # the interpolator returns NaN outside the simulated range; NaN is
        # not JSON-compliant and not a meaningful NA either
        na = None
    return {
        'mode_spacing_MHz': mode_spacing,
        'linewidth_0_HWHM_MHz': linewidth_0,
        'linewidth_1_HWHM_MHz': linewidth_1,
        'NA': na,
    }


# ------------------------------------------------ pairs (df / FSR) extraction
def area_lorentzian(x, x0, gamma, A, y0):
    """Area A, HWHM gamma, centre x0 (the pairs script's parametrization —
    the peak height is A / (pi * gamma))."""
    return A * gamma / (np.pi * ((x - x0) ** 2 + gamma ** 2)) + y0


def double_lorentzian(x, x01, gamma1, A1, x02, gamma2, A2, y0):
    """One fundamental + higher-order mode pair on a common offset."""
    return (area_lorentzian(x, x01, gamma1, A1, 0.0)
            + area_lorentzian(x, x02, gamma2, A2, 0.0) + y0)


def fit_lorentzian_pair(x, y, x1_guess, x2_guess, region=None):
    """Fit a pair of area-parametrized Lorentzians; return (params, errors)
    dicts keyed by DOUBLE_LORENTZIAN_PARAMS.

    `x1_guess` / `x2_guess` are the clicked peak positions, `region` the
    (min, max) of the selected window (defaults to the data's ends).
    Guess recipe as in extract_df_and_fsr_from_scope_csv.py: y0 = min,
    gamma = region width / 6, amplitudes from the samples nearest each
    clicked centre. Raises RuntimeError when the fit does not converge.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if region is not None:
        region_min, region_max = float(region[0]), float(region[1])
    else:
        region_min, region_max = float(x[0]), float(x[-1])
    width = region_max - region_min

    y0_init = float(np.min(y))
    gamma_init = width / 6.0
    A1_init = np.pi * gamma_init * max(nearest_value(x, y, x1_guess) - y0_init, 1e-9)
    A2_init = np.pi * gamma_init * max(nearest_value(x, y, x2_guess) - y0_init, 1e-9)
    p0 = [x1_guess, gamma_init, A1_init, x2_guess, gamma_init, A2_init, y0_init]

    popt, pcov = curve_fit(double_lorentzian, x, y, p0=p0, maxfev=50000)
    perr = np.sqrt(np.diag(pcov))
    return (dict(zip(DOUBLE_LORENTZIAN_PARAMS, (float(v) for v in popt))),
            dict(zip(DOUBLE_LORENTZIAN_PARAMS, (float(v) for v in perr))))


def pair_positions_results(positions, fsr_mhz=None, na_over_fsr_interp=None):
    """Per-pair df / FSR extraction from an ordered list of fitted pair
    positions [[x01, x02], ...] (>= 2 pairs, in scan order).

    The FSR in x-units is the spacing between consecutive pairs' first
    peaks (forward difference for the first pair, backward for the last
    analyzed one, centred otherwise); each analyzed pair contributes
    df = |x02 - x01| and df / FSR. With `fsr_mhz` (cavity_fsr_mhz()) df is
    scaled to MHz; with `na_over_fsr_interp` ((df / FSR) -> NA) each pair
    gets an NA (None outside the simulated range). Returns a list of
    len(positions) - 1 row dicts.
    """
    if len(positions) < 2:
        raise ValueError('need at least two fitted pairs to get an FSR')
    rows = []
    for i in range(len(positions) - 1):
        if i == 0:
            fsr = positions[i + 1][0] - positions[i][0]
        elif i == len(positions) - 2:
            fsr = positions[i][0] - positions[i - 1][0]
        else:
            fsr = (positions[i + 1][0] - positions[i - 1][0]) / 2.0
        if fsr == 0:
            raise ValueError('two pairs share the same first-peak position')
        df = abs(positions[i][1] - positions[i][0])
        df_over_fsr = df / fsr
        na = None
        if na_over_fsr_interp is not None:
            na = float(na_over_fsr_interp(df_over_fsr))
            if not np.isfinite(na):
                na = None  # outside the simulated range (and not JSON-safe)
        rows.append({
            'fsr': float(fsr),
            'df': float(df),
            'df_over_fsr': float(df_over_fsr),
            'NA': na,
            'df_MHz': (df_over_fsr * fsr_mhz) if fsr_mhz is not None else None,
        })
    return rows


def pair_summary(rows):
    """Mean / std over the pairs (ddof=1, as pandas does) of df / FSR, df [MHz]
    and NA; stds are None with a single row, and the NA / MHz fields are None
    when no pair got an NA / when the rows were built without an FSR in MHz."""
    ratios = np.array([row['df_over_fsr'] for row in rows], dtype=float)
    nas = np.array([row['NA'] for row in rows if row['NA'] is not None],
                   dtype=float)
    dfs_mhz = np.array([row['df_MHz'] for row in rows if row['df_MHz'] is not None],
                       dtype=float)
    return {
        'n_pairs': len(rows),
        'df_over_fsr_mean': float(ratios.mean()),
        'df_over_fsr_std': float(ratios.std(ddof=1)) if len(ratios) > 1 else None,
        'df_MHz_mean': float(dfs_mhz.mean()) if len(dfs_mhz) else None,
        'df_MHz_std': float(dfs_mhz.std(ddof=1)) if len(dfs_mhz) > 1 else None,
        'NA_mean': float(nas.mean()) if len(nas) else None,
        'NA_std': float(nas.std(ddof=1)) if len(nas) > 1 else None,
    }


# ------------------------------------------------- NA mapping (cavity design)
# The mode-spacing <-> NA relation comes from the external cavity-design
# project (path in local_config.py). Building the interpolators runs the whole
# lens-position simulation and takes a while, so they are cached per geometry.
_na_cache = {}  # (elements, long_arm, mid_arm, short_arms, N_points) -> (interp, interp, error)


def _add_cavity_design_to_path():
    """Put the repo root and the cavity-design project on sys.path."""
    repo_root = str(Path(__file__).resolve().parents[1])
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from local_config import PATH_CAVITY_DESIGN_PROJECT
    if PATH_CAVITY_DESIGN_PROJECT not in sys.path:
        sys.path.append(PATH_CAVITY_DESIGN_PROJECT)


def _import_simulation():
    """Import the cavity-design lens-position simulation (raises if unavailable)."""
    _add_cavity_design_to_path()
    import simple_analysis_scripts.mode_spacing_to_NA as simulation
    return simulation


def list_cavity_elements():
    """Names accepted in a cavity element list, from the cavity-design catalog.

        python -c "from pico_scope.mode_analysis import list_cavity_elements; print(*list_cavity_elements(), sep='\\n')"
    """
    return _import_simulation().available_element_names()


def mark_mode_spacing_on_dependencies(mode_spacing_MHz):
    """Mark a measured mode spacing [MHz] on the cavity-design dependencies figure.

    Draws a vertical line on each panel: at the spacing itself on the NA-vs-spacing panel, and at
    the small arm length that yields it on the other. Needs a dependencies figure to be open (i.e.
    get_na_interpolators(..., plot_dependencies=True) earlier in the run). Returns None on success
    or a '<why>' string, so a missing cavity-design project never breaks the analysis.
    """
    try:
        _import_simulation().mark_mode_spacing(mode_spacing_MHz)
    except Exception as error:
        return f'{type(error).__name__}: {error}'
    return None


def get_na_interpolators(long_arm=LONG_ARM_LENGTH, mid_arm=MID_ARM_LENGTH,
                         short_arm_lengths=SHORT_ARM_LENGTHS,
                         elements=CAVITY_ELEMENTS,
                         N_points=N_points,
                         plot_cavity=False, plot_spectrum=False,
                         plot_dependencies=False):
    """Return (mode_spacing_interp, mode_spacing_over_fsr_interp, error).

    mode_spacing_interp maps mode spacing [Hz] -> NA;
    mode_spacing_over_fsr_interp maps (df / FSR) -> NA.
    `elements` names the cavity's optical elements in optical order (see
    list_cavity_elements()); a name that is not in the catalog is an error the
    caller must fix, so it is raised, not reported.
    When the cavity-design project cannot be imported or the simulation
    fails, returns (None, None, '<why>') — callers report the MHz quantities
    and mark the NA unavailable.
    """
    key = (tuple(elements), long_arm, mid_arm, short_arm_lengths, N_points)
    # The cached interpolators carry no figures with them, so a caller that asked for plots has to
    # re-run the simulation to get them - otherwise a second run in the same session (a live
    # console re-running a script) would silently show nothing to mark the measurement on.
    if key in _na_cache and not (plot_cavity or plot_spectrum or plot_dependencies):
        return _na_cache[key]
    try:
        simulation = _import_simulation()
    except Exception as error:
        result = (None, None, f'{type(error).__name__}: {error}')
    else:
        try:
            mode_spacing_interp, mode_spacing_over_fsr_interp = \
                simulation.generate_lens_position_dependencies_output(
                    short_arm_lengths=short_arm_lengths,
                    long_arm_length=long_arm,
                    mid_arm_length=mid_arm,
                    elements=list(elements),
                    N_points=N_points,
                    plot_cavity=plot_cavity,
                    plot_spectrum=plot_spectrum,
                    plot_dependencies=plot_dependencies,
                )
            result = (mode_spacing_interp, mode_spacing_over_fsr_interp, None)
        except simulation.UnknownCavityElement:
            # A misspelled element name is a typo in the caller's config block, not a missing
            # simulation: fail loudly instead of silently dropping the NA from the report.
            raise
        except Exception as error:
            result = (None, None, f'{type(error).__name__}: {error}')
    _na_cache[key] = result
    return result


# ------------------------------------------------------------------ self-test
def _self_test():
    rng = np.random.default_rng(seed=7)
    true = {'A0': 1.0, 's0': 0.010, 'x0': 0.30,
            'A1': 0.6, 's1': 0.012, 'x1': 0.62,
            'd': 0.08, 'r': 6.0, 'y0': 0.05}
    x = np.linspace(0.0, 1.0, 5000)
    y = six_lorentzian_model(x, *(true[k] for k in SIX_LORENTZIAN_PARAMS))
    y += rng.normal(0.0, 0.005, len(x))

    x_dec, y_dec = decimate(x, y)
    assert len(x_dec) <= MAX_FIT_POINTS

    # deliberately rough clicks
    params, errors = fit_six_lorentzians(
        x_dec, y_dec, x0_guess=0.31, x1_guess=0.60, d_guess=0.075)
    for key in ('x0', 'x1', 'd', 's0', 's1'):
        assert abs(params[key] - true[key]) < 0.05 * max(abs(true[key]), 0.01), \
            f'{key}: fitted {params[key]:.5g}, true {true[key]:.5g}'

    results = sideband_results(params['x0'], params['x1'], params['d'],
                               params['s0'], params['s1'], f_sb_mhz=25.0)
    expected_spacing = abs(true['x1'] - true['x0']) / true['d'] * 25.0  # 100 MHz
    assert abs(results['mode_spacing_MHz'] - expected_spacing) < 1.0, results
    assert results['NA'] is None
    print(f"fit ok: mode spacing {results['mode_spacing_MHz']:.2f} MHz "
          f"(expected {expected_spacing:.2f}), "
          f"HWHM0 {results['linewidth_0_HWHM_MHz']:.3f} MHz, "
          f"HWHM1 {results['linewidth_1_HWHM_MHz']:.3f} MHz")

    # fake interpolator exercises the NA path without the cavity project
    results = sideband_results(params['x0'], params['x1'], params['d'],
                               params['s0'], params['s1'], f_sb_mhz=25.0,
                               na_interp=lambda hz: hz / 1e9)
    assert abs(results['NA'] - expected_spacing / 1e3) < 0.01
    print(f"NA lookup path ok (fake interpolator): NA = {results['NA']:.4f}")

    # ---- pairs (df / FSR): three synthetic pairs, FSR 0.3, df/FSR 0.2
    fsr_true, df_true = 0.3, 0.06
    firsts = [0.15, 0.45, 0.75]
    x = np.linspace(0.0, 1.0, 20000)
    y = np.full_like(x, 0.02)
    for x01 in firsts:
        y += double_lorentzian(x, x01, 0.004, 0.01, x01 + df_true, 0.004,
                               0.004, 0.0)
    y += rng.normal(0.0, 0.001, len(x))

    positions = []
    for x01 in firsts:
        region = (x01 - 0.05, x01 + df_true + 0.05)
        mask = (x >= region[0]) & (x <= region[1])
        x_dec, y_dec = decimate(x[mask], y[mask])
        params, _ = fit_lorentzian_pair(x_dec, y_dec, x1_guess=x01 + 0.005,
                                        x2_guess=x01 + df_true - 0.005,
                                        region=region)
        positions.append([params['x01'], params['x02']])
        assert abs(params['x01'] - x01) < 0.002, params
        assert abs(params['x02'] - (x01 + df_true)) < 0.002, params

    rows = pair_positions_results(positions, fsr_mhz=cavity_fsr_mhz(),
                                  na_over_fsr_interp=lambda ratio: ratio * 2)
    assert len(rows) == 2
    for row in rows:
        assert abs(row['fsr'] - fsr_true) < 0.005, row
        assert abs(row['df_over_fsr'] - df_true / fsr_true) < 0.02, row
        assert abs(row['NA'] - 2 * row['df_over_fsr']) < 1e-9, row
    summary = pair_summary(rows)
    assert summary['n_pairs'] == 2 and summary['df_over_fsr_std'] is not None
    print(f"pairs path ok: df/FSR = {summary['df_over_fsr_mean']:.4f} "
          f"± {summary['df_over_fsr_std']:.4f} (expected {df_true / fsr_true:.4f}), "
          f"FSR = {cavity_fsr_mhz():.1f} MHz")

    # a NaN-returning interpolator must yield NA = None, never NaN
    rows = pair_positions_results(positions, na_over_fsr_interp=lambda r: float('nan'))
    assert all(row['NA'] is None for row in rows)
    summary = pair_summary(rows)
    assert summary['NA_mean'] is None and summary['NA_std'] is None
    print('pairs NaN-NA path ok (None, not NaN)')
    print('mode_analysis self-test passed')


if __name__ == '__main__':
    _self_test()
