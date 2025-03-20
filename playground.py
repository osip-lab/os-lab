import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')  # Or 'TkAgg' if Qt5Agg doesn't work
import matplotlib.pyplot as plt


df = pd.read_csv(r"C:\Users\OsipLab\Weizmann Institute Dropbox\Michael Kali\Lab's Dropbox\Laser Phase Plate\Experiments\Results\20250320\New folder\20250320-0009.csv", skiprows=[1, 2])

data_numpy = df.to_numpy()
# %%

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib.widgets import SpanSelector
import itertools

# Example Data
x = data_numpy[:, 0]  # Time column
y = data_numpy[:, 2]  # Channel B column


# Lorentzian Function
def lorentzian(x, x0, gamma, A, y0):
    return A * gamma / (np.pi * ((x - x0) ** 2 + gamma ** 2)) + y0


# List to store fit results
lorentzian_params = [[]]
fit_colors = itertools.cycle(["r", "g", "b", "m", "c", "y"])  # Cycle through colors
current_color = next(fit_colors)

# Plot Data
fig, ax = plt.subplots()
ax.plot(x, y, label="Raw Data")
fit_lines = []  # Store fit line objects


# Selection Callback
def onselect(xmin, xmax):
    global current_color

    # Get indices for selected range
    indices = np.where((x >= xmin) & (x <= xmax))[0]
    if len(indices) < 5:  # Ignore very small selections
        return

    # Compute improved initial guesses
    x_range = x[indices]
    y_range = y[indices]

    x0_init = x_range[np.argmax(y_range)]  # Peak position
    y0_init = np.min(y_range)  # Minimum baseline
    gamma_init = (xmax - xmin) / 4  # Width estimate
    A_init = np.pi * gamma_init * np.max(y_range)  # Amplitude estimate

    # Fit Lorentzian
    p0 = [x0_init, gamma_init, A_init, y0_init]  # Improved initial guess
    popt, _ = curve_fit(lorentzian, x_range, y_range, p0=p0)

    lorentzian_params[-1].append(popt)

    # Plot Fit
    fit_x = np.linspace(xmin, xmax, 200)
    fit_y = lorentzian(fit_x, *popt)
    fit_line, = ax.plot(fit_x, fit_y, color=current_color, linestyle="--", label=f"Fit {len(lorentzian_params[-1])}")
    fit_lines.append(fit_line)

    plt.draw()


# Delete Last Fit
def on_key(event):
    global current_color

    if event.key == "d" and lorentzian_params[-1]:
        # Remove last fit
        lorentzian_params[-1].pop()
        if fit_lines:
            fit_lines.pop().remove()
        plt.draw()

    elif event.key == "enter":
        # Start a new fit set
        if lorentzian_params[-1]:  # Only add a new list if the last one isn't empty
            lorentzian_params.append([])
            current_color = next(fit_colors)  # Change color


# Span Selector for Selection
span = SpanSelector(ax, onselect, "horizontal", useblit=True, interactive=True)
fig.canvas.mpl_connect("key_press_event", on_key)

plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()

# Print results after closing
print("Lorentzian Fit Results:", lorentzian_params)
# %%
for i in range(len(lorentzian_params) - 1):
    fsr = lorentzian_params[i + 1][0][0] -lorentzian_params[i][0][0]
    print(f"FSR {i}: {fsr}")
    df = lorentzian_params[i][1][0] - lorentzian_params[i][0][0]
    print(f"df {i}: {df}")
    df_over_fst = df/fsr
    print(f"df/FST {i}: {df_over_fst}")
