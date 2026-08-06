import pandas as pd
import matplotlib

matplotlib.use('Qt5Agg')
from utilities.utils import (append_numerical_result_line, ask_long_arm_length,
                             get_picoscope_trace_path_from_clipboard)
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib.widgets import SpanSelector
import itertools

# All the math (models, pair fit, df/FSR extraction, NA interpolators) lives
# in pico_scope.mode_analysis so the kalishlot web GUI runs the exact same
# operations live on the streaming scope.
from pico_scope.mode_analysis import (DOUBLE_LORENTZIAN_PARAMS,
                                      area_lorentzian as lorentzian,
                                      cavity_fsr_mhz, double_lorentzian,
                                      fit_lorentzian_pair,
                                      get_na_interpolators,
                                      mark_mode_spacing_on_dependencies,
                                      pair_positions_results, pair_summary)

# --- the cavity being measured (edit this when the setup changes) ----------
# Element names come from the cavity-design catalog; list them in optical order.
# To see the available names:
#   python -c "from pico_scope.mode_analysis import list_cavity_elements; print(*list_cavity_elements(), sep='\n')"
CAVITY_ELEMENTS = [
    'LASER_OPTIK_MIRROR',
    'EDMUND_4p5MM_ASPHERIC_83580',
    'Thorlabs 200mm Plano Convex Lens - LA-1708-B',
    'COASTLINE_20CM_MIRROR',
]
SHORT_ARM_LENGTHS = (0.5e-4, 2e-4)  # [m] lens-scan span around the collimation point
LONG_ARM_LENGTH = 36e-2         # [m] lens -> far mirror; only the DEFAULT - the
                                  # value actually used is asked for on every run
MID_ARM_LENGTH = 1.5e-2           # [m] only used by 4-element cavities
N_points = 200                    # lens positions simulated across SHORT_ARM_LENGTHS
SHORT_ARM_LENGTH = 0.7e-2   # [m] near mirror -> lens (the physical one, not the simulation's scan)
PLOT_CAVITY=True
# The dependency plot is not optional - it is where the measured mode spacing is marked once
# the pairs have been fitted.

# The long arm changes between measurements, so it is asked for on every run;
# LONG_ARM_LENGTH above is only the default. It feeds both the FSR and the NA
# simulation, so it has to be answered before either is built.
long_arm_length = ask_long_arm_length(LONG_ARM_LENGTH)  # [m], prompted in cm

L = long_arm_length + MID_ARM_LENGTH + SHORT_ARM_LENGTH  # Cavity length in meters, sets the FSR via FSR = c / (2 * L)
FSR_MHZ = cavity_fsr_mhz(long_arm=long_arm_length, mid_arm=MID_ARM_LENGTH,
                         short_arm=SHORT_ARM_LENGTH)

# Built by the cavity-design project (path in local_config.py); cached, so a
# second analysis in the same session is instant.
mode_spacing_interp, mode_spacing_over_fsr_interp, na_error = get_na_interpolators(
    elements=CAVITY_ELEMENTS, long_arm=long_arm_length, mid_arm=MID_ARM_LENGTH, N_points=N_points, plot_dependencies=True, plot_cavity=PLOT_CAVITY)
if mode_spacing_over_fsr_interp is None:
    raise RuntimeError(f'cavity-design NA simulation unavailable: {na_error}')
# %% Load the PicoScope trace (.psdata or .csv; psdata is converted on the fly)
specific_file_path, input_path = get_picoscope_trace_path_from_clipboard()

df = pd.read_csv(specific_file_path, skiprows=[1, 2])
df = df.loc[:, ['Time', 'Channel D']]
data_numpy = df.to_numpy()

x = data_numpy[:, 0]  # Time column
y = data_numpy[:, 1]  # Channel B column


lorentzian_positions = [[]]
fit_colors = itertools.cycle(["r", "g", "b", "m", "c", "y"])
current_color = next(fit_colors)

fig, ax = plt.subplots()
ax.plot(x, y, label="Raw Data")
fit_lines = []
position_lines = []
mode = "single"  # Can be 'single', 'position', or 'double'
double_span_stage = 0


def add_position(x0):
    if not lorentzian_positions:
        lorentzian_positions.append([x0])
    elif len(lorentzian_positions[-1]) < 2:
        lorentzian_positions[-1].append(x0)
    else:
        lorentzian_positions.append([x0])
    print(lorentzian_positions)

    if len(lorentzian_positions[-1]) == 2:
        global current_color
        print("Changing color")
        current_color = next(fit_colors)


def print_latest_pair_na():
    # Needs two consecutive completed pairs: the FSR is the spacing between their first peaks.
    pairs = [pos for pos in lorentzian_positions if len(pos) == 2]
    if len(pairs) < 2:
        return
    fsr = np.abs(pairs[-1][0] - pairs[-2][0])
    df_pair = np.abs(pairs[-1][1] - pairs[-1][0])
    df_over_fsr = df_pair / fsr
    na = float(mode_spacing_over_fsr_interp(df_over_fsr))
    df_mhz = df_over_fsr * FSR_MHZ
    print(f"df/FSR = {df_over_fsr:.4f}, NA = {na:.4f}, "
          f"df = {df_mhz:.2f} MHz (FSR = {FSR_MHZ:.1f} MHz for L = {L:.4g} m)")


def onselect(xmin, xmax):
    global current_color, mode, double_span_stage

    indices = np.where((x >= xmin) & (x <= xmax))[0]
    if len(indices) < 5:
        return

    x_range = x[indices]
    y_range = y[indices]

    if mode == "single":
        x0_init = x_range[np.argmax(y_range)]
        y0_init = np.min(y_range)
        gamma_init = (xmax - xmin) / 4
        A_init = np.pi * gamma_init * np.max(y_range)
        p0 = [x0_init, gamma_init, A_init, y0_init]

        try:
            popt, _ = curve_fit(lorentzian, x_range, y_range, p0=p0)
            x0_fitted = popt[0]

            fit_x = np.linspace(xmin, xmax, 200)
            fit_y = lorentzian(fit_x, *popt)
            fit_line, = ax.plot(fit_x, fit_y, color=current_color, linestyle="--")
            fit_lines.append(fit_line)

            add_position(x0_fitted)

        except Exception as e:
            print("Single fit failed:", e)

    elif mode == "double":
        if double_span_stage == 0:
            double_fit_data['xmin'] = xmin
            double_fit_data['xmax'] = xmax
            double_fit_data['x_range'] = x_range
            double_fit_data['y_range'] = y_range
            double_span_stage = 1
            ax.set_title(
                "Select second span between estimated center of the first lorentzian and the "
                "estimated center of the second lorentzian")
        elif double_span_stage == 1:
            x01_init = xmin
            x02_init = xmax

            x_range = double_fit_data['x_range']
            y_range = double_fit_data['y_range']
            xmin = double_fit_data['xmin']
            xmax = double_fit_data['xmax']

            try:
                popt, _ = fit_lorentzian_pair(x_range, y_range,
                                              x1_guess=x01_init,
                                              x2_guess=x02_init,
                                              region=(xmin, xmax))
                x01_fitted, x02_fitted = popt['x01'], popt['x02']
                lorentzian_positions.append([x01_fitted, x02_fitted])

                fit_x = np.linspace(xmin, xmax, 300)
                fit_y = double_lorentzian(
                    fit_x, *(popt[name] for name in DOUBLE_LORENTZIAN_PARAMS))
                fit_line, = ax.plot(fit_x, fit_y, color=current_color, linestyle="--")
                fit_lines.append(fit_line)

                print_latest_pair_na()

                current_color = next(fit_colors)
                double_span_stage = 0
            except Exception as e:
                print("Double fit failed:", e)
                double_span_stage = 0

    ax.set_title(f"Mode: {mode}")
    plt.draw()


def onclick(event):
    if event.inaxes != ax:
        return

    if event.button != 1 or plt.get_current_fig_manager().toolbar.mode != '':
        return

    if mode == "position":
        x_clicked = event.xdata
        add_position(x_clicked)
        line = ax.axvline(x_clicked, color=current_color, linestyle=":")
        position_lines.append(line)
        ax.set_title(f"Mode: {mode}")
        plt.draw()


def on_key(event):
    global current_color, mode, double_span_stage
    if event.key == "d" and lorentzian_positions[-1]:
        lorentzian_positions[-1].pop()
        if not lorentzian_positions[-1] and len(lorentzian_positions) > 1:
            lorentzian_positions.pop()
        if mode == "position" and position_lines:
            position_lines.pop().remove()
        elif fit_lines:
            fit_lines.pop().remove()
        plt.draw()
    elif event.key in ["1", "2", "3"]:
        if event.key == "1":
            mode = "single"
            ax.set_title("Mode: single-lorentzian fit")
        elif event.key == "2":
            mode = "position"
            ax.set_title("Mode: manual position selection")
        elif event.key == "3":
            mode = "double"
            double_span_stage = 0
            ax.set_title("Mode: sum of 2 lorentzians, choose a range in which to fit")
        plt.draw()
    elif event.key == "z":
        toolbar = plt.get_current_fig_manager().toolbar
        toolbar.zoom()


# Temporary storage for double fit region
double_fit_data = {}

span = SpanSelector(ax, onselect, "horizontal", useblit=True, interactive=True)
fig.canvas.mpl_connect("key_press_event", on_key)
fig.canvas.mpl_connect("button_press_event", onclick)

plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show(block=True)
# Remove all elements that are empty lists from lorentzian_positions:
# %%
print("before cleaning:", lorentzian_positions)
lorentzian_positions = [pos for pos in lorentzian_positions if pos]
print("after cleaning:", lorentzian_positions)
if len(lorentzian_positions) < 2:
    print("Not enough data to calculate FSR and df.")
else:
    rows = pair_positions_results(lorentzian_positions, fsr_mhz=FSR_MHZ,
                                  na_over_fsr_interp=mode_spacing_over_fsr_interp)
    results_df = pd.DataFrame(rows)
    print(results_df)

    summary = pair_summary(rows)

    # Mark the measured mode spacing (the mean df over the pairs) on both panels of the
    # dependency plot: at the spacing itself on the right, at the small arm length that
    # would produce it on the left.
    if summary["df_MHz_mean"] is not None:
        mark_error = mark_mode_spacing_on_dependencies(summary["df_MHz_mean"])
        if mark_error is not None:
            print(f"Could not mark the dependency plots: {mark_error}")

    # Record the extraction next to the original data file (one line per run).
    na_text = (f"{summary['NA_mean']:.4f}" if summary["NA_mean"] is not None
               else "unavailable (df/FSR outside the simulated range)")
    df_mhz_text = (f"{summary['df_MHz_mean']:.4f} MHz"
                   if summary["df_MHz_mean"] is not None else "unavailable")
    results_text = (f"long_arm_length = {long_arm_length:.4g} m, "
                    f"n_mode_pairs = {summary['n_pairs']}, "
                    f"mode_spacing = {df_mhz_text}, "
                    f"df_over_fsr = {summary['df_over_fsr_mean']:.4f}, "
                    f"NA = {na_text}")
    if summary["df_over_fsr_std"] is not None:
        results_text += f" (std over pairs: df_over_fsr {summary['df_over_fsr_std']:.4f}"
        if summary["df_MHz_std"] is not None:
            results_text += f", mode_spacing {summary['df_MHz_std']:.4f} MHz"
        if summary["NA_std"] is not None:
            results_text += f", NA {summary['NA_std']:.4f}"
        results_text += ")"
    append_numerical_result_line(input_path, results_text)
