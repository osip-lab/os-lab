import pandas as pd
import matplotlib

matplotlib.use('Qt5Agg')
from utilities.utils import (append_numerical_result_line, ask_long_arm_length,
                             get_picoscope_trace_path_from_clipboard)
import matplotlib.pyplot as plt

# All the math (models, pair fit, df/FSR extraction, NA interpolators) lives in
# pico_scope.mode_analysis so the kalishlot web GUI runs the exact same
# operations live on the streaming scope; the interactive marking window lives
# in pico_scope.mode_marking, shared with pico_scope/mode_map_2d.py.
from pico_scope.mode_analysis import (cavity_fsr_mhz, get_na_interpolators,
                                      pair_positions_results, pair_summary)
from pico_scope.mode_marking import mark_pairs, positions_and_widths

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
LONG_ARM_LENGTH = 36e-2         # [m] lens -> far mirror; only the DEFAULT - the
                                  # value actually used is asked for on every run
MID_ARM_LENGTH = 1.5e-2           # [m] only used by 4-element cavities
N_points = 300                    # lens positions simulated across SHORT_ARM_LENGTHS
SHORT_ARM_LENGTH = 0.7e-2   # [m] near mirror -> lens (the physical one, not the simulation's scan)

# The long arm changes between measurements, so it is asked for on every run;
# LONG_ARM_LENGTH above is only the default. It feeds both the FSR and the NA
# simulation, so it has to be answered before either is built.
long_arm_length = ask_long_arm_length(LONG_ARM_LENGTH)  # [m], prompted in cm

L = long_arm_length + MID_ARM_LENGTH + SHORT_ARM_LENGTH  # Cavity length in meters, sets the FSR via FSR = c / (2 * L)
FSR_MHZ = cavity_fsr_mhz(long_arm=long_arm_length, mid_arm=MID_ARM_LENGTH,
                         short_arm=SHORT_ARM_LENGTH)

# The NA mapping is NOT built here: the cavity-design simulation is run at the end of the
# script, once every pair has been fitted, so that the measured mode spacing can be handed
# to it and come back marked on its dependency plot.
# %% Load the PicoScope trace (.psdata or .csv; psdata is converted on the fly)
specific_file_path, input_path = get_picoscope_trace_path_from_clipboard()

df = pd.read_csv(specific_file_path, skiprows=[1, 2])
df = df.loc[:, ['Time', 'Channel D']]
data_numpy = df.to_numpy()

x = data_numpy[:, 0]  # Time column
y = data_numpy[:, 1]  # Channel B column


# %% Mark the mode pairs -----------------------------------------------------
# Drag the 0th-order mode and the first-order mode that follows it, for
# consecutive longitudinal mode numbers; two pairs are the minimum (their
# spacing is the FSR). See mode_marking for the keys the window understands.
marks = mark_pairs(x, y)
complete = [pair for pair in marks if len(pair) == 2]
if len(complete) != len(marks):
    print(f"Ignoring {len(marks) - len(complete)} incomplete pair(s).")
lorentzian_positions, lorentzian_widths = positions_and_widths(complete)
print("Marked pairs:", lorentzian_positions)

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
