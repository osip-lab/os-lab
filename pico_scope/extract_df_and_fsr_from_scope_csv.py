from pathlib import Path

import pandas as pd
import matplotlib

matplotlib.use('Qt5Agg')
from utilities.utils import (append_numerical_result_line, ask_long_arm_length,
                             wait_for_path_from_clipboard)
import matplotlib.pyplot as plt

# All the math (models, pair fit, df/FSR extraction, NA interpolators) lives in
# pico_scope.mode_analysis so the kalishlot web GUI runs the exact same
# operations live on the streaming scope; the interactive marking window lives
# in pico_scope.mode_marking, shared with pico_scope/mode_map_2d.py.
from pico_scope.mode_analysis import (cavity_fsr_mhz, get_na_interpolators,
                                      pair_positions_results, pair_summary)
from pico_scope.mode_marking import mark_pairs, positions_and_widths
# The marking is saved as a '<data file>.modemarks.json' sidecar next to the
# data and read back here and by pico_scope/mode_map_2d.py - so the marking
# done at the bench, while the measurement is being taken, is the one the 2D
# map reuses instead of asking for it all over again.
from pico_scope.mode_marks_cache import (ask_use_cached_marks, complete_pairs,
                                         load_cached_marks, make_record,
                                         save_marks, trace_csv_path)

# --- the cavity being measured (edit this when the setup changes) ----------
# Element names come from the cavity-design catalog; list them in optical order.
# To see the available names:
#   python -c "from pico_scope.mode_analysis import list_cavity_elements; print(*list_cavity_elements(), sep='\n')"
CAVITY_ELEMENTS = [
    'LASER_OPTIK_MIRROR',
    'EDMUND_4MM_ASPHERIC_16701',
    'COASTLINE_20CM_MIRROR',
]
SHORT_ARM_LENGTHS = (0.5e-4, 2e-4)  # [m] lens-scan span around the collimation point
MID_ARM_LENGTH = 1.5e-2           # [m] only used by 4-element cavities
N_points = 300                    # lens positions simulated across SHORT_ARM_LENGTHS
SHORT_ARM_LENGTH = 0.7e-2   # [m] near mirror -> lens (the physical one, not the simulation's scan)

TIME_COLUMN = 'Time'        # x-axis column in the PicoScope CSV
SIGNAL_COLUMN = 'Channel D'  # intensity column to analyze

# The NA mapping is NOT built here: the cavity-design simulation is run at the end of the
# script, once every pair has been fitted, so that the measured mode spacing can be handed
# to it and come back marked on its dependency plot.
# %% Load the PicoScope trace (.psdata or .csv; psdata is converted on the fly)
# The path that is copied is the one the sidecar and the results line belong
# to, so the .psdata is kept as it is here and converted only when the trace
# actually has to be drawn - a file that is already marked never is.
input_path = wait_for_path_from_clipboard(filetype=('csv', 'psdata'))


# %% Mark the mode pairs -----------------------------------------------------
# Drag the 0th-order mode and the first-order mode that follows it, for
# consecutive longitudinal mode numbers; two pairs are the minimum (their
# spacing is the FSR). See mode_marking for the keys the window understands.
#
# The long arm changes between measurements, so it is asked for rather than
# configured - but only for a file being marked now: a file that was marked
# before was measured with the long arm its sidecar holds, and typing it again
# could only disagree with the marks. It feeds both the FSR and the NA
# simulation, so it is settled before either is built.
cached = load_cached_marks(input_path, min_pairs=2, signal_column=SIGNAL_COLUMN)
if cached is not None and ask_use_cached_marks(cached, input_path):
    print("  using the cached marks")
    marks = cached['marks']
    long_arm_length = cached['long_arm_m']  # [m]
    print(f"Long arm length: {long_arm_length * 100:.4g} cm (from the sidecar)")
else:
    csv_path = trace_csv_path(input_path)
    df = pd.read_csv(csv_path, skiprows=[1, 2])
    df = df.loc[:, [TIME_COLUMN, SIGNAL_COLUMN]].dropna()
    data_numpy = df.to_numpy()

    x = data_numpy[:, 0]  # Time column
    y = data_numpy[:, 1]  # intensity column

    raw_marks = mark_pairs(x, y, title=Path(input_path).name)
    marks = complete_pairs(raw_marks)
    if len(marks) != len(raw_marks):
        print(f"Ignoring {len(raw_marks) - len(marks)} incomplete pair(s).")
    long_arm_length = ask_long_arm_length()  # [m], prompted in cm
    if marks:
        # next to the data, for this script's next run and for the 2D map
        save_marks(input_path, make_record(input_path, csv_path, marks,
                                           long_arm_length,
                                           signal_column=SIGNAL_COLUMN))

lorentzian_positions, lorentzian_widths = positions_and_widths(marks)
print("Marked pairs:", lorentzian_positions)

L = long_arm_length + MID_ARM_LENGTH + SHORT_ARM_LENGTH  # Cavity length in meters, sets the FSR via FSR = c / (2 * L)
FSR_MHZ = cavity_fsr_mhz(long_arm=long_arm_length, mid_arm=MID_ARM_LENGTH,
                         short_arm=SHORT_ARM_LENGTH)

# %%
if len(lorentzian_positions) < 2:
    print("Not enough data to calculate FSR and df.")
else:
    # First pass, without the NA mapping: it gives the measured mode spacing (the mean df
    # over the pairs), which the simulation needs before it runs in order to mark it on the
    # dependency plot.
    measured_mode_spacing_MHz = pair_summary(
        pair_positions_results(lorentzian_positions, fsr_mhz=FSR_MHZ))['df_MHz_mean']
    print(f"Measured mode spacing: {measured_mode_spacing_MHz:.4f} MHz")

    # Built by the cavity-design project (path in local_config.py). Slow - it runs the whole
    # lens-position simulation - so it happens once, here, rather than per fitted pair.
    mode_spacing_interp, mode_spacing_over_fsr_interp, na_error = get_na_interpolators(
        elements=CAVITY_ELEMENTS, long_arm=long_arm_length, mid_arm=MID_ARM_LENGTH,
        short_arm_lengths=SHORT_ARM_LENGTHS, N_points=N_points,
        measured_mode_spacing_MHz=measured_mode_spacing_MHz,
        plot_system=True)  # always: the plot is what carries the measurement marker
    if mode_spacing_over_fsr_interp is None:
        raise RuntimeError(f'cavity-design NA simulation unavailable: {na_error}')

    # Second pass, now with an NA per pair.
    # widths= adds each peak's measured FWHM [MHz] to the table (empty for a
    # clicked, unfitted position).
    rows = pair_positions_results(lorentzian_positions, fsr_mhz=FSR_MHZ,
                                  na_over_fsr_interp=mode_spacing_over_fsr_interp,
                                  widths=lorentzian_widths)
    results_df = pd.DataFrame(rows)
    print(results_df)
    summary = pair_summary(rows)

    # Record the extraction next to the original data file (one line per run).
    na_text = (f"{summary['NA_mean']:.4f}" if summary["NA_mean"] is not None
               else "unavailable (df/FSR outside the simulated range)")
    df_mhz_text = (f"{summary['df_MHz_mean']:.4f} MHz"
                   if summary["df_MHz_mean"] is not None else "unavailable")
    # The measured linewidths (FWHM), one per mode of the pair - "unavailable"
    # when the positions were clicked (mode 2) rather than fitted.
    linewidth_text = ", ".join(
        f"linewidth_{i} = " + (f"{summary[f'fwhm_{i}_MHz_mean']:.4f} MHz"
                               if summary[f"fwhm_{i}_MHz_mean"] is not None
                               else "unavailable")
        for i in (0, 1))
    results_text = (f"long_arm_length = {long_arm_length:.4g} m, "
                    f"n_mode_pairs = {summary['n_pairs']}, "
                    f"mode_spacing = {df_mhz_text}, "
                    f"df_over_fsr = {summary['df_over_fsr_mean']:.4f}, "
                    f"{linewidth_text}, "
                    f"NA = {na_text}")
    if summary["df_over_fsr_std"] is not None:
        results_text += f" (std over pairs: df_over_fsr {summary['df_over_fsr_std']:.4f}"
        if summary["df_MHz_std"] is not None:
            results_text += f", mode_spacing {summary['df_MHz_std']:.4f} MHz"
        for i in (0, 1):
            if summary[f"fwhm_{i}_MHz_std"] is not None:
                results_text += f", linewidth_{i} {summary[f'fwhm_{i}_MHz_std']:.4f} MHz"
        if summary["NA_std"] is not None:
            results_text += f", NA {summary['NA_std']:.4f}"
        results_text += ")"
    append_numerical_result_line(input_path, results_text)

# The simulation shows its system plot non-blocking (so the report above prints without waiting
# for the window). Without this the interpreter would reach the end of the script and tear the Qt
# event loop down with the window still on screen - it would flash open and vanish.
plt.show(block=True)
