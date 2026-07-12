"""
mode_spacing_extraction_sidebands.py

Extract the numerical-aperture-related quantities of a resonator mode from a
PicoScope intensity-vs-time trace, using EOM sidebands as a built-in frequency
ruler.

Pipeline
--------
1. Load a PicoScope trace (file path taken from the clipboard). Both .csv and
   .psdata files are accepted; .psdata files are converted to CSV on the fly
   via PicoScope 7's command-line BatchConvert.
2. Plot the spectrum in an interactive Qt window.
3. Drag a horizontal window to select the region of interest.
4. Click the zeroth-order mode.
5. Click one sideband of the central line (gives the sideband distance d).
6. Click the first-order mode.
7. Enter the (single-side) sideband modulation frequency [MHz] (default 20).
8. Fit a sum of 6 Lorentzians + constant offset:

       f_total = f(A0,    s0, x0)
               + f(A0/r,  s0, x0 - d) + f(A0/r,  s0, x0 + d)
               + f(A1,    s1, x1)
               + f(A1/r,  s1, x1 - d) + f(A1/r,  s1, x1 + d)
               + y0

   with f(A, s, x0; x) = A / (1 + ((x - x0)/s)^2)   (peak A, HWHM s, centre x0).

   Free parameters: A0, s0, x0, A1, s1, x1, d, r, y0.

9. Compute, scaled by the sideband frequency f_sb:
       (x1 - x0)/d * f_sb   -> 0th->1st-order mode spacing  [MHz]
       s1/d        * f_sb   -> 1st-order linewidth (HWHM)    [MHz]
       s0/d        * f_sb   -> 0th-order linewidth (HWHM)    [MHz]
10. Map the mode spacing to the numerical aperture (NA) using the cavity-design
    simulation (simple_analysis_scripts.mode_spacing_to_NA).
11. Print mode spacing, linewidths and NA together.
12. Append a one-line record (long arm length, mode spacing, NA) to
    numerical-results.txt in the folder of the original data file.

Note for future development: steps 4-6 already produce raw coordinate guesses
(x0_guess, x1_guess, d_guess). A future "coordinate-only" mode can skip the fit
(step 8) and call `report_results(...)` directly with the clicked values.
"""

# %% [Step 0] Imports and configuration -------------------------------------
import matplotlib

matplotlib.use('Qt5Agg')

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector
from scipy.optimize import curve_fit
from utilities.utils import append_numerical_result_line, wait_for_path_from_clipboard

# --- configuration the user may want to tweak ------------------------------
TIME_COLUMN = 'Time'              # x-axis column in the PicoScope CSV
SIGNAL_COLUMN = 'Channel D'       # intensity column to analyze
MAX_PLOT_POINTS = 500            # decimate plots/fit to at most this many points
DEFAULT_SIDEBAND_FREQ_MHZ = 20.0  # single-side EOM modulation frequency [MHz]
SIDEBAND_AMP_RATIO_GUESS = 6.0    # r: main-peak / sideband-peak amplitude ratio
LONG_ARM_LENGTH = 31e-2
MID_ARM_LENGTH = 1.5e-2
from local_config import PICOSCOPE_EXE
# PicoScope 7 executable, used to convert .psdata files to CSV on the fly.

# --- NA mapping (cavity-design project) ------------------------------------
# Plot toggles passed to generate_lens_position_dependencies_output(); the
# cavity / spectrum / dependency plots are shown non-blocking, so the final
# report prints without waiting for the windows to be closed.
SIM_PLOT_CAVITY = False
SIM_PLOT_SPECTRUM = False
SIM_PLOT_DEPENDENCIES = True


def decimate(x_arr, y_arr, max_points=MAX_PLOT_POINTS):
    """Keep every n-th sample so that at most `max_points` points remain.

    PicoScope exports can hold ~100k points, which is far denser than the fit
    needs and makes interactive plotting sluggish. A simple stride keeps the
    spectrum shape intact while staying responsive.
    """
    step = max(1, int(np.ceil(len(x_arr) / max_points)))
    return x_arr[::step], y_arr[::step]


def psdata_to_csv(psdata_path):
    """Convert a PicoScope .psdata file to CSV; return the CSV path.

    Uses PicoScope 7's command-line `BatchConvert` mode, which produces a CSV
    identical to the GUI's "Save as CSV". BatchConvert operates on folders,
    so the single file is copied to a temporary folder and converted there.

    If an up-to-date CSV with the same name already sits next to the .psdata
    file (e.g. from an earlier manual export or an earlier run), it is used
    directly and no conversion is performed.
    """
    psdata_path = Path(psdata_path)
    sibling_csv = psdata_path.with_suffix('.csv')
    if sibling_csv.is_file() and sibling_csv.stat().st_mtime >= psdata_path.stat().st_mtime:
        print(f"Using existing up-to-date CSV: {sibling_csv}")
        return str(sibling_csv)

    if not Path(PICOSCOPE_EXE).is_file():
        raise FileNotFoundError(
            f"PicoScope executable not found at {PICOSCOPE_EXE!r} - "
            "update PICOSCOPE_EXE, or save the trace as CSV manually."
        )

    tmp_dir = Path(tempfile.mkdtemp(prefix='psdata_to_csv_'))
    in_dir = tmp_dir / 'in'
    out_dir = tmp_dir / 'out'
    in_dir.mkdir()
    out_dir.mkdir()
    shutil.copy2(psdata_path, in_dir)

    print(f"Converting '{psdata_path.name}' to CSV with PicoScope 7 ...")
    # Note: BatchConvert fails on folder paths with a trailing backslash;
    # str(Path) never produces one.
    result = subprocess.run(
        [PICOSCOPE_EXE, 'BatchConvert', str(in_dir), str(out_dir), '.csv'],
        capture_output=True, text=True,
    )
    # A single-waveform file becomes '<stem>.csv' directly in out_dir; a file
    # with multiple waveform buffers becomes a '<stem>' subfolder holding
    # '<stem>_1.csv' ... '<stem>_N.csv', so search recursively and sort by the
    # numeric buffer suffix (plain name-sorting would put _10 before _2).
    def buffer_index(p):
        suffix = p.stem.rsplit('_', 1)[-1]
        return int(suffix) if suffix.isdigit() else 0

    csv_files = sorted(out_dir.rglob('*.csv'), key=buffer_index)
    if result.returncode != 0 or not csv_files:
        raise RuntimeError(
            f"psdata -> CSV conversion failed (exit code {result.returncode}).\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
    print("Conversion succeeded.")

    if len(csv_files) == 1:
        return str(csv_files[0])

    print(f"The psdata file contains {len(csv_files)} waveform buffers:")
    for i, p in enumerate(csv_files, start=1):
        print(f"  [{i}] {p.name}")
    while True:
        raw_in = input(
            f"Which waveform to use? 1-{len(csv_files)} "
            f"[default {len(csv_files)} - the most recent]: "
        ).strip()
        if raw_in == '':
            choice = len(csv_files)
        else:
            try:
                choice = int(raw_in)
            except ValueError:
                choice = 0
        if 1 <= choice <= len(csv_files):
            break
        print(f"  Please enter a number between 1 and {len(csv_files)}.")
    print(f"Using waveform: {csv_files[choice - 1].name}")
    return str(csv_files[choice - 1])


# %% [Step 1] Load the PicoScope trace (.psdata or .csv) ---------------------
input_path = wait_for_path_from_clipboard(filetype=('csv', 'psdata'))
if input_path.lower().endswith('.psdata'):
    csv_path = psdata_to_csv(input_path)
else:
    csv_path = input_path
print(f"Loading: {csv_path}")

# Rows 1 and 2 of a PicoScope export are the unit / blank header rows.
raw = pd.read_csv(csv_path, skiprows=[1, 2])
raw = raw.loc[:, [TIME_COLUMN, SIGNAL_COLUMN]].dropna()

x = raw[TIME_COLUMN].to_numpy(dtype=float)
y = raw[SIGNAL_COLUMN].to_numpy(dtype=float)
print(f"Loaded {len(x)} samples from column '{SIGNAL_COLUMN}'.")


# %% [Step 1.5] Choose the analysis mode ------------------------------------
def ask_analysis_mode():
    """Return 'fit' (Lorentzian fit) or 'point' (use clicked points as-is)."""
    while True:
        choice = input(
            "Analysis mode - [F]it or [P]oint-selection? [F]: "
        ).strip().lower()
        if choice in ('', 'f', 'fit'):
            return 'fit'
        if choice in ('p', 'point', 'point-selection'):
            return 'point'
        print("  Please enter 'f' (fit) or 'p' (point-selection).")


analysis_mode = ask_analysis_mode()
print(f"Analysis mode: {analysis_mode}")


# %% [Step 2] Plot the full spectrum ----------------------------------------
# Decimate only for display; the full-resolution x/y are kept for cropping.
x_full_plot, y_full_plot = decimate(x, y)
print(f"Plotting {len(x_full_plot)} of {len(x)} points "
      f"(stride {max(1, len(x) // len(x_full_plot))}).")

fig_full, ax_full = plt.subplots()
ax_full.plot(x_full_plot, y_full_plot, lw=0.8, label='Raw data')
ax_full.set_xlabel(f"{TIME_COLUMN} [s]")
ax_full.set_ylabel(f"{SIGNAL_COLUMN}")
ax_full.set_title('Step 3: drag a horizontal window over the region of interest')
ax_full.legend(loc='upper right')
fig_full.show()  # realise the window (interactive mode is off by default)


# %% [Step 3] Drag a horizontal window to select the region ------------------
def select_region(fig, ax):
    """Block until the user drags a horizontal span; return (xmin, xmax)."""
    state = {}

    def on_select(xmin, xmax):
        if xmax - xmin <= 0:  # ignore accidental clicks / zero-width spans
            return
        state['xmin'], state['xmax'] = xmin, xmax
        fig.canvas.stop_event_loop()

    selector = SpanSelector(
        ax, on_select, 'horizontal', useblit=True, interactive=True,
        props=dict(alpha=0.2, facecolor='tab:orange'),
    )
    fig.canvas.draw_idle()
    fig.show()  # realise the window before blocking on events
    fig.canvas.start_event_loop(timeout=0)  # runs until stop_event_loop()
    selector.disconnect_events()
    return state['xmin'], state['xmax']


region_min, region_max = select_region(fig_full, ax_full)
print(f"Selected region: [{region_min:.6g}, {region_max:.6g}] s")

mask = (x >= region_min) & (x <= region_max)
# Crop at full resolution, then decimate the region for re-plotting and fitting
# (the decimated density is still well beyond what the 9-parameter fit needs).
x_fit, y_fit = decimate(x[mask], y[mask])
region_width = region_max - region_min
print(f"Region of interest: {int(mask.sum())} raw points "
      f"-> {len(x_fit)} after decimation.")


# %% [Steps 4-6] Click the mode / sideband positions ------------------------
fig, ax = plt.subplots()
ax.plot(x_fit, y_fit, lw=0.9, label='Selected data')
ax.set_xlabel(f"{TIME_COLUMN} [s]")
ax.set_ylabel(f"{SIGNAL_COLUMN}")
ax.legend(loc='upper right')
fig.show()  # realise the window before blocking on ginput


def click_x(fig, ax, prompt, marker_color):
    """Block until the user clicks once; return the clicked x and draw a marker."""
    ax.set_title(prompt)
    fig.canvas.draw_idle()
    (xc, _yc), = fig.ginput(n=1, timeout=0)
    ax.axvline(xc, color=marker_color, ls='--', lw=1)
    fig.canvas.draw_idle()
    return xc


x0_guess = click_x(fig, ax, 'Step 4: click the ZEROTH-order mode', 'tab:green')
x_sb_guess = click_x(fig, ax, 'Step 5: click ONE sideband of the central line', 'tab:gray')
x1_guess = click_x(fig, ax, 'Step 6: click the FIRST-order mode', 'tab:red')

d_guess = abs(x_sb_guess - x0_guess)
print(f"x0 guess = {x0_guess:.6g} s")
print(f"x1 guess = {x1_guess:.6g} s")
print(f"d  guess = {d_guess:.6g} s  (from sideband click)")


# %% [Step 7] Enter the sideband modulation frequency -----------------------
# def ask_sideband_freq(default_mhz):
#     raw_in = input(
#         f"Step 7: sideband (single-side) frequency in MHz "
#         f"[default {default_mhz}]: "
#     ).strip()
#     if raw_in == '':
#         return float(default_mhz)
#     try:
#         return float(raw_in)
#     except ValueError:
#         print(f"  Could not parse '{raw_in}', using default {default_mhz} MHz.")
#         return float(default_mhz)


f_sb_mhz = DEFAULT_SIDEBAND_FREQ_MHZ  # ask_sideband_freq(DEFAULT_SIDEBAND_FREQ_MHZ)
print(f"Using sideband frequency f_sb = {f_sb_mhz} MHz (per side).")


# %% [Step 8] Build the 6-Lorentzian model and fit --------------------------
def lorentzian(x, A, s, x0):
    """Peak value A, half-width-at-half-maximum s, centre x0."""
    return A / (1.0 + ((x - x0) / s) ** 2)


def six_lorentzian_model(x, A0, s0, x0, A1, s1, x1, d, r, y0):
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


if analysis_mode == 'fit':
    # --- initial guesses ----------------------------------------------------
    y0_init = float(np.min(y_fit))
    A0_init = max(nearest_value(x_fit, y_fit, x0_guess) - y0_init, 1e-9)
    A1_init = max(nearest_value(x_fit, y_fit, x1_guess) - y0_init, 1e-9)
    s_init = max(d_guess / 6.0, region_width / 200.0)

    p0 = [A0_init, s_init, x0_guess,
          A1_init, s_init, x1_guess,
          d_guess, SIDEBAND_AMP_RATIO_GUESS, y0_init]

    # --- bounds (peaks => positive amplitudes; widths/distance positive) ---
    eps = region_width / 1e6
    lower = [0.0,    eps,          region_min,
             0.0,    eps,          region_min,
             eps,    1.0,          -np.inf]
    upper = [np.inf, region_width, region_max,
             np.inf, region_width, region_max,
             region_width, np.inf, np.inf]

    popt, pcov = curve_fit(
        six_lorentzian_model, x_fit, y_fit,
        p0=p0, bounds=(lower, upper), maxfev=50000,
    )
    perr = np.sqrt(np.diag(pcov))
    A0, s0, x0, A1, s1, x1, d, r, y0 = popt
    fit_params, fit_errors = popt, perr

    # --- overlay the fit and its components --------------------------------
    x_dense = np.linspace(region_min, region_max, 2000)
    ax.plot(x_dense, six_lorentzian_model(x_dense, *popt),
            color='k', lw=1.6, label='Fit (6 Lorentzians + offset)')
    for centre, width, amp in [
        (x0, s0, A0), (x0 - d, s0, A0 / r), (x0 + d, s0, A0 / r),
        (x1, s1, A1), (x1 - d, s1, A1 / r), (x1 + d, s1, A1 / r),
    ]:
        ax.plot(x_dense, lorentzian(x_dense, amp, width, centre) + y0,
                color='tab:blue', ls=':', lw=0.8, alpha=0.7)
    ax.set_title('Fit result')

else:  # point-selection: use the clicked positions as-is, no fitting
    x0, x1, d = x0_guess, x1_guess, d_guess
    s0 = s1 = np.nan  # widths are not measured in this mode
    fit_params = fit_errors = None
    ax.set_title('Point-selection mode - no fit')

ax.legend(loc='upper right')
fig.canvas.draw_idle()
# Without a running event loop, draw_idle alone would not repaint the window
# until the final plt.show(); a short pause flushes the fit overlay to screen
# before the (slow) NA simulation runs.
plt.pause(0.1)


# %% [Step 9] Compute and pretty-print the results --------------------------
def report_results(x0, x1, d, s0, s1, f_sb_mhz, fit_params=None, fit_errors=None,
                   na_interp=None):
    """Scale the geometric quantities by the sideband frequency and print them.

    If `na_interp` is given (the mode-spacing -> NA interpolator from the
    cavity-design simulation), the numerical aperture is looked up from the
    mode spacing and printed alongside the other parameters.

    This is the seam for the future coordinate-only mode: call it with values
    obtained from clicks instead of from the fit.
    """
    scale = f_sb_mhz / d  # MHz per x-unit (sidebands sit at +/- f_sb -> +/- d)

    mode_spacing = abs(x1 - x0) / d * f_sb_mhz   # (x1 - x0)/d * f_sb
    linewidth_1 = s1 / d * f_sb_mhz           # s1/d * f_sb (HWHM)
    linewidth_0 = s0 / d * f_sb_mhz           # s0/d * f_sb (HWHM)

    # The simulation interpolator is built on mode spacing in Hz, so convert.
    na = float(na_interp(mode_spacing * 1e6)) if na_interp is not None else None

    width = 64
    bar = '=' * width

    def row(label, value, unit):
        return f"  {label:<34}{value:>18.4f} {unit}"

    print()
    print(bar)
    print("  RESONATOR MODE ANALYSIS".center(width))
    print(bar)
    print(f"  Sideband frequency f_sb (per side){'':>6}{f_sb_mhz:>10.4f} MHz")
    print(f"  Frequency scale  f_sb / d{'':>15}{scale:>10.4g} MHz / x-unit")
    print('-' * width)
    print(row("Mode spacing  (x1 - x0)/d * f_sb", mode_spacing, "MHz"))
    print(row("0th-order linewidth s0/d * f_sb", linewidth_0, "MHz (HWHM)"))
    print(row("1st-order linewidth s1/d * f_sb", linewidth_1, "MHz (HWHM)"))
    if na is not None:
        print('-' * width)
        print(row("Numerical aperture  NA", na, ""))
    print('-' * width)
    print(row("  (0th-order FWHM)", 2 * linewidth_0, "MHz"))
    print(row("  (1st-order FWHM)", 2 * linewidth_1, "MHz"))
    print(bar)

    if fit_params is not None:
        names = ['A0', 's0', 'x0', 'A1', 's1', 'x1', 'd', 'r', 'y0']
        print("  Fit parameters:")
        errs = fit_errors if fit_errors is not None else [float('nan')] * len(names)
        for name, val, err in zip(names, fit_params, errs):
            print(f"    {name:<4}= {val:>14.6g}  +/- {err:.3g}")
        print(bar)
    print()

    return {
        'mode_spacing_MHz': mode_spacing,
        'linewidth_0_HWHM_MHz': linewidth_0,
        'linewidth_1_HWHM_MHz': linewidth_1,
        'NA': na,
    }


# %% [Step 10] Map the mode spacing to numerical aperture (NA) ---------------
# The NA<->mode-spacing relation comes from the cavity-design project, which is
# added to sys.path at runtime (the IDE resolves it via local_config).
from local_config import PATH_CAVITY_DESIGN_PROJECT

if PATH_CAVITY_DESIGN_PROJECT not in sys.path:
    sys.path.append(PATH_CAVITY_DESIGN_PROJECT)
import simple_analysis_scripts.mode_spacing_to_NA as simulation

# mode_spacing_interp: mode spacing [Hz] -> NA
mode_spacing_interp, mode_spacing_over_fsr_interp = \
    simulation.generate_lens_position_dependencies_output(
        short_arm_lengths=4e-4,
        long_arm_length=LONG_ARM_LENGTH,
        mid_arm_length=MID_ARM_LENGTH,
        plot_cavity=SIM_PLOT_CAVITY,
        plot_spectrum=SIM_PLOT_SPECTRUM,
        plot_dependencies=SIM_PLOT_DEPENDENCIES,
    )


# %% [Step 11] Final report (mode spacing, linewidths, NA) -------------------
results = report_results(
    x0, x1, d, s0, s1, f_sb_mhz,
    fit_params=fit_params, fit_errors=fit_errors,
    na_interp=mode_spacing_interp,
)


# %% [Step 12] Record the results next to the original data file -------------
# Appends a one-line record to numerical-results.txt in the folder of the
# original file (the .psdata/.csv the user copied, not the temporary CSV).
na_text = f"{results['NA']:.4f}" if results['NA'] is not None else "N/A"
append_numerical_result_line(
    input_path,
    f"long_arm_length = {LONG_ARM_LENGTH:.4g} m, "
    f"mode_spacing = {results['mode_spacing_MHz']:.4f} MHz, "
    f"NA = {na_text}",
)

# Keep all windows open (and responsive) after the report has been printed.
plt.show(block=True)
