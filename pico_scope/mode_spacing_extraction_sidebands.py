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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector
from utilities.utils import (append_numerical_result_line,
                             get_picoscope_trace_path_from_clipboard)
# All the math (models, fit, scaling, NA mapping) lives in mode_analysis so
# the kalishlot web GUI runs the identical computation on the live stream.
from pico_scope.mode_analysis import (DEFAULT_SIDEBAND_FREQ_MHZ,
                                      LONG_ARM_LENGTH, MID_ARM_LENGTH,
                                      SIDEBAND_AMP_RATIO_GUESS,
                                      SIX_LORENTZIAN_PARAMS, decimate,
                                      fit_six_lorentzians,
                                      get_na_interpolators, lorentzian,
                                      sideband_results, six_lorentzian_model)

# --- configuration the user may want to tweak ------------------------------
TIME_COLUMN = 'Time'              # x-axis column in the PicoScope CSV
SIGNAL_COLUMN = 'Channel D'       # intensity column to analyze

# --- NA mapping (cavity-design project) ------------------------------------
# Plot toggles passed to generate_lens_position_dependencies_output(); the
# cavity / spectrum / dependency plots are shown non-blocking, so the final
# report prints without waiting for the windows to be closed.
SIM_PLOT_CAVITY = False
SIM_PLOT_SPECTRUM = False
SIM_PLOT_DEPENDENCIES = False


# %% [Step 1] Load the PicoScope trace (.psdata or .csv) ---------------------
csv_path, input_path = get_picoscope_trace_path_from_clipboard()
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


# %% [Step 8] Fit the 6-Lorentzian model (shared with the web GUI) ----------
if analysis_mode == 'fit':
    fit_params, fit_errors = fit_six_lorentzians(
        x_fit, y_fit, x0_guess=x0_guess, x1_guess=x1_guess, d_guess=d_guess,
        r_guess=SIDEBAND_AMP_RATIO_GUESS, region=(region_min, region_max))
    A0, s0, x0 = fit_params['A0'], fit_params['s0'], fit_params['x0']
    A1, s1, x1 = fit_params['A1'], fit_params['s1'], fit_params['x1']
    d, r, y0 = fit_params['d'], fit_params['r'], fit_params['y0']

    # --- overlay the fit and its components --------------------------------
    x_dense = np.linspace(region_min, region_max, 2000)
    ax.plot(x_dense, six_lorentzian_model(
                x_dense, *(fit_params[name] for name in SIX_LORENTZIAN_PARAMS)),
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

    results = sideband_results(x0, x1, d, s0, s1, f_sb_mhz, na_interp=na_interp)
    mode_spacing = results['mode_spacing_MHz']
    linewidth_0 = results['linewidth_0_HWHM_MHz']
    linewidth_1 = results['linewidth_1_HWHM_MHz']
    na = results['NA']

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
        print("  Fit parameters:")
        for name in SIX_LORENTZIAN_PARAMS:
            err = fit_errors[name] if fit_errors is not None else float('nan')
            print(f"    {name:<4}= {fit_params[name]:>14.6g}  +/- {err:.3g}")
        print(bar)
    print()

    return results


# %% [Step 10] Map the mode spacing to numerical aperture (NA) ---------------
# The NA<->mode-spacing relation comes from the cavity-design project (path in
# local_config.py); mode_analysis builds and caches the interpolators.
# mode_spacing_interp: mode spacing [Hz] -> NA
mode_spacing_interp, mode_spacing_over_fsr_interp, na_error = \
    get_na_interpolators(
        long_arm=LONG_ARM_LENGTH,
        mid_arm=MID_ARM_LENGTH,
        plot_cavity=SIM_PLOT_CAVITY,
        plot_spectrum=SIM_PLOT_SPECTRUM,
        plot_dependencies=SIM_PLOT_DEPENDENCIES,
    )
if na_error is not None:
    print(f"NA mapping unavailable: {na_error}")


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
