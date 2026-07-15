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
LONG_ARM_LENGTH = 34.4e-2
MID_ARM_LENGTH = 1.5e-2
SHORT_ARM_LENGTHS = 4e-4

SIX_LORENTZIAN_PARAMS = ['A0', 's0', 'x0', 'A1', 's1', 'x1', 'd', 'r', 'y0']


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


# ------------------------------------------------- NA mapping (cavity design)
# The mode-spacing <-> NA relation comes from the external cavity-design
# project (path in local_config.py). Building the interpolators runs the whole
# lens-position simulation and takes a while, so they are cached per geometry.
_na_cache = {}  # (long_arm, mid_arm, short_arms) -> (interp, interp, error)


def get_na_interpolators(long_arm=LONG_ARM_LENGTH, mid_arm=MID_ARM_LENGTH,
                         short_arm_lengths=SHORT_ARM_LENGTHS,
                         plot_cavity=False, plot_spectrum=False,
                         plot_dependencies=False):
    """Return (mode_spacing_interp, mode_spacing_over_fsr_interp, error).

    mode_spacing_interp maps mode spacing [Hz] -> NA;
    mode_spacing_over_fsr_interp maps (df / FSR) -> NA.
    When the cavity-design project cannot be imported or the simulation
    fails, returns (None, None, '<why>') — callers report the MHz quantities
    and mark the NA unavailable.
    """
    key = (long_arm, mid_arm, short_arm_lengths)
    if key in _na_cache:
        return _na_cache[key]
    try:
        repo_root = str(Path(__file__).resolve().parents[1])
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        from local_config import PATH_CAVITY_DESIGN_PROJECT
        if PATH_CAVITY_DESIGN_PROJECT not in sys.path:
            sys.path.append(PATH_CAVITY_DESIGN_PROJECT)
        import simple_analysis_scripts.mode_spacing_to_NA as simulation
        mode_spacing_interp, mode_spacing_over_fsr_interp = \
            simulation.generate_lens_position_dependencies_output(
                short_arm_lengths=short_arm_lengths,
                long_arm_length=long_arm,
                mid_arm_length=mid_arm,
                plot_cavity=plot_cavity,
                plot_spectrum=plot_spectrum,
                plot_dependencies=plot_dependencies,
            )
        result = (mode_spacing_interp, mode_spacing_over_fsr_interp, None)
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
    print('mode_analysis self-test passed')


if __name__ == '__main__':
    _self_test()
