import pandas as pd
import matplotlib
from utilities.video_tools.utils import wait_for_path_from_clipboard
from local_config import PATH_CAVITY_DESIGN_PROJECT
import sys



import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib.widgets import SpanSelector
import itertools

sys.path.append(PATH_CAVITY_DESIGN_PROJECT)
import plots_generations_scripts.df_over_fsr_to_NA_ratio_output as simulation
df_over_FSR_interp, NAs, df_over_FSR, Ls = simulation.generate_lens_position_dependencies_output(plot_cavity=True,
                                                                                                 plot_spectrum=True,
                                                                                                 plot_dependencies=True)
matplotlib.use('Qt5Agg')
# %%
specific_file_path = wait_for_path_from_clipboard(filetype='csv')

df = pd.read_csv(specific_file_path, skiprows=[1, 2])
df = df.loc[:, ['Time', 'Channel B']]
data_numpy = df.to_numpy()

x = data_numpy[:, 0]  # Time column
y = data_numpy[:, 1]  # Channel B column


def lorentzian(x, x0, gamma, A, y0):
    return A * gamma / (np.pi * ((x - x0) ** 2 + gamma ** 2)) + y0


def double_lorentzian(x, x01, gamma1, A1, x02, gamma2, A2, y0):
    return (lorentzian(x, x01, gamma1, A1, 0) + lorentzian(x, x02, gamma2, A2, 0) + y0)


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

            y0_init = np.min(y_range)
            gamma1_init = gamma2_init = (xmax - xmin) / 6
            A1_init = np.pi * gamma1_init * y_range[np.argmin(np.abs(x_range - xmin))]
            A2_init = np.pi * gamma2_init * y_range[np.argmin(np.abs(x_range - xmax))]

            p0 = [x01_init, gamma1_init, A1_init, x02_init, gamma2_init, A2_init, y0_init]

            try:
                popt, _ = curve_fit(double_lorentzian, x_range, y_range, p0=p0)
                x01_fitted, x02_fitted = popt[0], popt[3]
                lorentzian_positions.append([x01_fitted, x02_fitted])

                fit_x = np.linspace(xmin, xmax, 300)
                fit_y = double_lorentzian(fit_x, *popt)
                fit_line, = ax.plot(fit_x, fit_y, color=current_color, linestyle="--")
                fit_lines.append(fit_line)

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
    results = {"fsr": [], "df": [], "df_over_fsr": []}
    for i in range(len(lorentzian_positions) - 1):
        if i == 0:
            fsr = lorentzian_positions[i + 1][0] - lorentzian_positions[i][0]
        elif i == len(lorentzian_positions) - 2:
            fsr = lorentzian_positions[i][0] - lorentzian_positions[i - 1][0]
        else:
            fsr = (lorentzian_positions[i + 1][0] - lorentzian_positions[i - 1][0]) / 2
        df = np.abs(lorentzian_positions[i][1] - lorentzian_positions[i][0])
        df_over_fsr = df / fsr

        results["fsr"].append(fsr)
        results["df"].append(df)
        results["df_over_fsr"].append(df_over_fsr)

    results_df = pd.DataFrame(results)
    results_df["NA"] = df_over_FSR_interp(results_df["df_over_fsr"])

    print(results_df)
