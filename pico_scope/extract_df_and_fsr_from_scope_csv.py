import pandas as pd
import matplotlib

from utilities.video_tools.utils import wait_for_video_path_from_clipboard

matplotlib.use('Qt5Agg')  # Or 'TkAgg' if Qt5Agg doesn't work
from local_config import PATH_DROPBOX
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib.widgets import SpanSelector
import itertools


specific_file_path = wait_for_video_path_from_clipboard(filetype='csv')

df = pd.read_csv(specific_file_path, skiprows=[1, 2])
df = df.loc[:, ['Time', 'Channel B']]
data_numpy = df.to_numpy()

x = data_numpy[:, 0]  # Time column
y = data_numpy[:, 1]  # Channel B column
# CONSTANT_TITLE = "press: 1=single, 2=position, 3=double, d=delete last fit, enter=next group, z=zoom"
# def generate_title(i, mode):


# Lorentzian Function
def lorentzian(x, x0, gamma, A, y0):
    return A * gamma / (np.pi * ((x - x0) ** 2 + gamma ** 2)) + y0


def double_lorentzian(x, x01, gamma1, A1, x02, gamma2, A2, y0):
    return (lorentzian(x, x01, gamma1, A1, 0) + lorentzian(x, x02, gamma2, A2, 0) + y0)


# List to store only x0 results
lorentzian_positions = [[]]
fit_colors = itertools.cycle(["r", "g", "b", "m", "c", "y"])
current_color = next(fit_colors)

fig, ax = plt.subplots()
ax.plot(x, y, label="Raw Data")
fit_lines = []
position_lines = []
mode = "single"  # Can be 'single', 'position', or 'double'
temp_clicks = []
waiting_for_double_span = False
double_span_stage = 0  # 0 = wait for data span, 1 = wait for guess span


# SpanSelector callback for single or double fit
def onselect(xmin, xmax):
    global current_color, mode, temp_clicks, waiting_for_double_span, double_span_stage

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

        popt, _ = curve_fit(lorentzian, x_range, y_range, p0=p0)
        x0_fitted = popt[0]
        lorentzian_positions[-1].append([x0_fitted])

        fit_x = np.linspace(xmin, xmax, 200)
        fit_y = lorentzian(fit_x, *popt)
        fit_line, = ax.plot(fit_x, fit_y, color=current_color, linestyle="--")
        fit_lines.append(fit_line)

    elif mode == "double":
        if double_span_stage == 0:
            double_fit_data['xmin'] = xmin
            double_fit_data['xmax'] = xmax
            double_fit_data['x_range'] = x_range
            double_fit_data['y_range'] = y_range
            double_span_stage = 1
            print("Select second span to choose initial x0 positions")
        elif double_span_stage == 1:
            x01_init = xmin
            x02_init = xmax

            x_range = double_fit_data['x_range']
            y_range = double_fit_data['y_range']
            xmin = double_fit_data['xmin']
            xmax = double_fit_data['xmax']

            y0_init = np.min(y_range)
            gamma1_init = gamma2_init = (xmax - xmin) / 6
            A1_init = A2_init = np.max(y_range)

            p0 = [x01_init, gamma1_init, A1_init, x02_init, gamma2_init, A2_init, y0_init]

            try:
                popt, _ = curve_fit(double_lorentzian, x_range, y_range, p0=p0)
                x01_fitted, x02_fitted = popt[0], popt[3]
                print(f" onselect, try, before appending: {lorentzian_positions}")
                lorentzian_positions.append([x01_fitted, x02_fitted])
                print(f" onselect, try, after appending: {lorentzian_positions}")

                fit_x = np.linspace(xmin, xmax, 300)
                fit_y = double_lorentzian(fit_x, *popt)
                fit_line, = ax.plot(fit_x, fit_y, color=current_color, linestyle="--")
                fit_lines.append(fit_line)

                current_color = next(fit_colors)
                double_span_stage = 0
            except Exception as e:
                print("Fit failed:", e)
                double_span_stage = 0

    ax.set_title(f"Mode: {mode}")
    print(f"onselect, final {lorentzian_positions}")
    plt.draw()


# Mouse click for position mode

def onclick(event):
    if event.inaxes != ax:
        return

    if event.button != 1 or plt.get_current_fig_manager().toolbar.mode != '':
        return

    if mode == "position":
        x_clicked = event.xdata
        lorentzian_positions[-1].append([x_clicked])
        line = ax.axvline(x_clicked, color=current_color, linestyle=":")
        position_lines.append(line)
        ax.set_title(f"Mode: {mode}")
        plt.draw()


# Key events
def on_key(event):
    global current_color, mode, waiting_for_double_span, double_span_stage
    if event.key == "d" and lorentzian_positions[-1]:
        lorentzian_positions[-1].pop()
        if mode == "position" and position_lines:
            position_lines.pop().remove()
        elif fit_lines:
            fit_lines.pop().remove()
        plt.draw()
    elif event.key == "enter":
        if lorentzian_positions[-1]:
            lorentzian_positions.append([])
            current_color = next(fit_colors)
    elif event.key in ["1", "2", "3"]:
        if event.key == "1":
            mode = "single"
            ax.set_title("Mode: single-lorentzian fit")
            print("Mode set to: single-lorentzian fit")
        elif event.key == "2":
            mode = "position"
            ax.set_title("Mode: manual position selection")
            print("Mode set to: manual position selection")
        elif event.key == "3":
            mode = "double"
            double_span_stage = 0
            ax.set_title("Mode: sum of 2 lorentzians")
            print("Mode set to: sum of 2 lorentzians — select data span first")
        plt.draw()
    elif event.key == "z":
        toolbar = plt.get_current_fig_manager().toolbar
        if toolbar.mode == 'zoom rect':
            toolbar.zoom()
        else:
            toolbar.zoom()


# Temporary storage for double fit region
double_fit_data = {}

span = SpanSelector(ax, onselect, "horizontal", useblit=True, interactive=True)
fig.canvas.mpl_connect("key_press_event", on_key)
fig.canvas.mpl_connect("button_press_event", onclick)

plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()
# Remove all elements that are empty lists from lorentzian_positions:


# %%
lorentzian_positions = [[a[0][0], a[1][0]] for a in lorentzian_positions]
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
    print(results_df)
