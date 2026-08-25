"""Interactive marking of 0th-order / 1st-order mode pairs on a scope trace.

This is the picker that used to live inline in
pico_scope/extract_df_and_fsr_from_scope_csv.py, extracted so the 2D-map
script (pico_scope/mode_map_2d.py) marks its files with the exact same
procedure - and so the state it mutates while the window is open belongs to
one marker object rather than to the module, which lets several files be
marked one after another in a single run.

The interaction is unchanged:

    drag a span          fit (or, in mode 2, click) the next peak
    1 / 2 / 3            single-Lorentzian fit / manual position / pair fit
    d                    undo the last marking (a mode-3 pair fit, which
                         draws one curve over two peaks, goes as a whole)
    z                    toggle the toolbar's zoom
    Enter                finish (same as closing the window)
    close the window     finish

Peaks are marked in pairs: the 0th-order mode first, then the first-order
mode that follows it, repeated for consecutive longitudinal mode numbers.

Finishing with nothing marked returns an empty list, which is how a caller
that marks many files in a row (pico_scope/mode_map_2d.py) is told to skip
this one.
"""

import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import SpanSelector
from scipy.optimize import curve_fit

# repo root on sys.path so `pico_scope.*` resolves even when this file is run
# directly (Python only puts the script's own folder on sys.path, not the repo
# root the absolute imports assume).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pico_scope.mode_analysis import (DOUBLE_LORENTZIAN_PARAMS,  # noqa: E402
                                      area_lorentzian, double_lorentzian,
                                      fit_lorentzian_pair)

# Kept as the figure's suptitle rather than the axes title, which the handlers
# below overwrite with the current mode - this way the instructions stay on
# screen while you click.
SELECTION_INSTRUCTIONS = ("Drag the zeroth lorentzian range and the first order lorentzian "
                          "in each FSR sequentially. 'd' undoes the last "
                          "marking, Enter finishes.")


def peak_record(x0, gamma=None, area=None, y0=None):
    """One marked peak.

    `gamma` is the fitted HWHM, `height` the fitted peak height above the
    baseline (area_lorentzian is area-parametrized: height = A / (pi gamma)),
    `y0` the fitted offset. All three are None for a position that was clicked
    (mode 2) rather than fitted - the callers that need them say so.
    """
    height = None
    if area is not None and gamma:
        height = float(area) / (np.pi * abs(float(gamma)))
    return {'x0': float(x0),
            'gamma': None if gamma is None else float(gamma),
            'height': height,
            'y0': None if y0 is None else float(y0)}


def positions_and_widths(marks):
    """(positions, widths) as pair_positions_results() wants them.

    positions: [[x01, x02], ...], widths: [[gamma1, gamma2], ...] - both in
    marking order, with None for the widths of clicked (unfitted) positions.
    """
    positions = [[peak['x0'] for peak in pair] for pair in marks]
    widths = [[peak['gamma'] for peak in pair] for pair in marks]
    return positions, widths


class _PairMarker:
    """The window's state while the user marks one trace."""

    def __init__(self, x, y, title=None, instructions=SELECTION_INSTRUCTIONS):
        self.x = np.asarray(x, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self.pairs = [[]]  # list of pairs, each a list of up to 2 peak records
        # One entry per marking action, in order, so that 'd' can undo an
        # action whole. The colours are an index into a list rather than an
        # itertools.cycle(), because an undo has to step back through them.
        self.undo_stack = []
        self.fit_colors = ["r", "g", "b", "m", "c", "y"]
        self.color_index = 0
        self.mode = "single"  # 'single', 'position' or 'double'
        self.double_span_stage = 0
        self.double_fit_data = {}

        self.fig, self.ax = plt.subplots()
        self.fig.suptitle(instructions)
        self.ax.plot(self.x, self.y, label="Raw Data")
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_title(title if title else f"Mode: {self.mode}")
        self.ax.legend()
        self.span = SpanSelector(self.ax, self.onselect, "horizontal",
                                 useblit=True, interactive=True)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.fig.canvas.mpl_connect("button_press_event", self.onclick)

    # ------------------------------------------------------------- bookkeeping
    @property
    def current_color(self):
        return self.fit_colors[self.color_index % len(self.fit_colors)]

    def add_peaks(self, peaks, artists, new_pair=False):
        """Record one marking action: the peak(s) it found and what it drew.

        Kept together because an action is what 'd' undoes: the pair fit (mode
        3) produces two peaks under a single curve, so undoing peak by peak
        would leave the plot showing something the marks no longer say.
        `new_pair` is for that same mode - a pair fit always starts a pair of
        its own, rather than completing one that is half marked.
        """
        colors_advanced = 0
        for index, peak in enumerate(peaks):
            starts_pair = new_pair and index == 0
            if self.pairs and len(self.pairs[-1]) < 2 and not starts_pair:
                self.pairs[-1].append(peak)
            else:
                self.pairs.append([peak])
            if len(self.pairs[-1]) == 2:
                print("Changing color")
                self.color_index += 1
                colors_advanced += 1
        print(positions_and_widths(self.pairs)[0])
        self.undo_stack.append({'n_peaks': len(peaks),
                                'artists': list(artists),
                                'colors_advanced': colors_advanced})

    def undo_last_action(self):
        """Take back the last marking action - its peaks and its artists."""
        if not self.undo_stack:
            return
        action = self.undo_stack.pop()
        for _ in range(action['n_peaks']):
            if self.pairs and self.pairs[-1]:
                self.pairs[-1].pop()
            if self.pairs and not self.pairs[-1] and len(self.pairs) > 1:
                self.pairs.pop()
        for artist in action['artists']:
            artist.remove()
        self.color_index -= action['colors_advanced']
        print(positions_and_widths(self.pairs)[0])

    def print_latest_pair(self):
        """df / FSR of the two most recent complete pairs, as a live check.

        Needs two consecutive completed pairs: the FSR is the spacing between
        their first peaks. Printed in x-units - scaling it to MHz belongs to
        the caller, which is the one that knows the cavity.
        """
        pairs = [pair for pair in self.pairs if len(pair) == 2]
        if len(pairs) < 2:
            return
        fsr = np.abs(pairs[-1][0]['x0'] - pairs[-2][0]['x0'])
        df_pair = np.abs(pairs[-1][1]['x0'] - pairs[-1][0]['x0'])
        print(f"df/FSR = {df_pair / fsr:.4f}")

    # ---------------------------------------------------------------- handlers
    def onselect(self, xmin, xmax):
        indices = np.where((self.x >= xmin) & (self.x <= xmax))[0]
        if len(indices) < 5:
            return

        x_range = self.x[indices]
        y_range = self.y[indices]

        if self.mode == "single":
            x0_init = x_range[np.argmax(y_range)]
            y0_init = np.min(y_range)
            gamma_init = (xmax - xmin) / 4
            A_init = np.pi * gamma_init * np.max(y_range)
            p0 = [x0_init, gamma_init, A_init, y0_init]

            try:
                popt, _ = curve_fit(area_lorentzian, x_range, y_range, p0=p0)
                x0_fitted, gamma_fitted, A_fitted, y0_fitted = popt

                fit_x = np.linspace(xmin, xmax, 200)
                fit_y = area_lorentzian(fit_x, *popt)
                fit_line, = self.ax.plot(fit_x, fit_y, color=self.current_color,
                                         linestyle="--")

                self.add_peaks([peak_record(x0_fitted, gamma_fitted,
                                            A_fitted, y0_fitted)], [fit_line])

            except Exception as e:
                print("Single fit failed:", e)

        elif self.mode == "double":
            if self.double_span_stage == 0:
                self.double_fit_data = {'xmin': xmin, 'xmax': xmax,
                                        'x_range': x_range, 'y_range': y_range}
                self.double_span_stage = 1
                self.ax.set_title(
                    "Select second span between estimated center of the first lorentzian and the "
                    "estimated center of the second lorentzian")
                plt.draw()
                return

            x01_init, x02_init = xmin, xmax
            x_range = self.double_fit_data['x_range']
            y_range = self.double_fit_data['y_range']
            xmin = self.double_fit_data['xmin']
            xmax = self.double_fit_data['xmax']

            try:
                popt, _ = fit_lorentzian_pair(x_range, y_range,
                                              x1_guess=x01_init,
                                              x2_guess=x02_init,
                                              region=(xmin, xmax))
                fit_x = np.linspace(xmin, xmax, 300)
                fit_y = double_lorentzian(
                    fit_x, *(popt[name] for name in DOUBLE_LORENTZIAN_PARAMS))
                fit_line, = self.ax.plot(fit_x, fit_y, color=self.current_color,
                                         linestyle="--")

                self.add_peaks(
                    [peak_record(popt['x01'], popt['gamma1'], popt['A1'], popt['y0']),
                     peak_record(popt['x02'], popt['gamma2'], popt['A2'], popt['y0'])],
                    [fit_line], new_pair=True)

                self.print_latest_pair()
                self.double_span_stage = 0
            except Exception as e:
                print("Double fit failed:", e)
                self.double_span_stage = 0

        self.ax.set_title(f"Mode: {self.mode}")
        plt.draw()

    def onclick(self, event):
        if event.inaxes != self.ax:
            return

        if event.button != 1 or plt.get_current_fig_manager().toolbar.mode != '':
            return

        if self.mode == "position":
            x_clicked = event.xdata
            line = self.ax.axvline(x_clicked, color=self.current_color,
                                   linestyle=":")
            self.add_peaks([peak_record(x_clicked)], [line])
            self.ax.set_title(f"Mode: {self.mode}")
            plt.draw()

    def on_key(self, event):
        if event.key == "d":
            self.undo_last_action()
            plt.draw()
        elif event.key in ["1", "2", "3"]:
            if event.key == "1":
                self.mode = "single"
                self.ax.set_title("Mode: single-lorentzian fit")
            elif event.key == "2":
                self.mode = "position"
                self.ax.set_title("Mode: manual position selection")
            elif event.key == "3":
                self.mode = "double"
                self.double_span_stage = 0
                self.ax.set_title("Mode: sum of 2 lorentzians, choose a range in which to fit")
            plt.draw()
        elif event.key == "z":
            toolbar = plt.get_current_fig_manager().toolbar
            toolbar.zoom()
        elif event.key in ("enter", "return"):
            # confirm without reaching for the window's close button; closing
            # is what run()'s blocking show() waits for either way
            plt.close(self.fig)

    # -------------------------------------------------------------------- run
    def run(self):
        plt.show(block=True)
        return [pair for pair in self.pairs if pair]  # drop the empty entries


def mark_pairs(x, y, title=None, instructions=SELECTION_INSTRUCTIONS):
    """Open the marking window on (x, y) and block until it is closed.

    Returns one entry per marked pair, each a list of the pair's peak records
    (see peak_record()) in marking order: the 0th-order mode first, then the
    first-order mode. A trailing entry may hold a single peak, if the window
    was closed halfway through a pair, and the list is empty when the window
    was finished without anything marked at all.
    """
    return _PairMarker(x, y, title=title, instructions=instructions).run()


# ------------------------------------------------------------------ self-test
def _self_test():
    """Drive the handlers with synthetic spans - no window, no clicking."""
    from pico_scope.mode_analysis import (pair_positions_results, pair_summary)

    fsr, df, gamma, height, offset = 1e-3, 2.4e-4, 6e-6, 0.7, 0.05
    x = np.linspace(-3e-4, 2.6e-3, 60000)
    y = np.full_like(x, offset)
    for k in range(3):
        for shift, amplitude in ((0.0, height), (df, 0.6 * height)):
            y += area_lorentzian(x, k * fsr + shift, gamma,
                                 amplitude * np.pi * gamma, 0.0)

    marker = _PairMarker(x, y)
    # the trace plus the span selector's two edge handles - what ax.lines
    # holds before anything is marked, and what it must hold again once
    # everything has been undone
    bare_lines = len(marker.ax.lines)

    # mode 1: one dragged span per peak, two peaks per pair
    for k in range(2):
        for shift in (0.0, df):
            centre = k * fsr + shift
            marker.onselect(centre - 8 * gamma, centre + 8 * gamma)
    assert len(marker.pairs) == 2, marker.pairs
    for k, pair in enumerate(marker.pairs):
        assert abs(pair[0]['x0'] - k * fsr) < 1e-8, pair[0]
        assert abs(pair[1]['x0'] - (k * fsr + df)) < 1e-8, pair[1]
        assert abs(pair[0]['gamma'] - gamma) < 1e-8, pair[0]
        assert abs(pair[0]['height'] - height) < 1e-2, pair[0]
        assert abs(pair[0]['y0'] - offset) < 1e-2, pair[0]

    # mode 3: the region first, then the two centre guesses
    color_index_before_double = marker.color_index
    marker.mode = 'double'
    marker.onselect(2 * fsr - 10 * gamma, 2 * fsr + df + 10 * gamma)
    assert marker.double_span_stage == 1
    assert 'Select second span' in marker.ax.get_title(), marker.ax.get_title()
    marker.onselect(2 * fsr - gamma, 2 * fsr + df + gamma)
    assert marker.double_span_stage == 0 and len(marker.pairs) == 3, marker.pairs
    assert abs(marker.pairs[2][0]['x0'] - 2 * fsr) < 1e-8, marker.pairs[2]
    assert abs(marker.pairs[2][1]['x0'] - (2 * fsr + df)) < 1e-8, marker.pairs[2]
    assert abs(marker.pairs[2][0]['height'] - height) < 1e-2, marker.pairs[2]

    # 'd' undoes one marking action - and the mode-3 pair fit is one action,
    # two peaks under a single curve, so it goes as a whole and leaves the
    # plot saying exactly what the marks say
    marker.mode = 'single'
    key_d = type('Event', (), {'key': 'd'})()
    curves = len(marker.ax.lines)  # bare_lines plus one line per fit
    marker.on_key(key_d)
    assert len(marker.pairs) == 2, marker.pairs
    assert len(marker.ax.lines) == curves - 1, len(marker.ax.lines)
    assert marker.color_index == color_index_before_double, marker.color_index

    # and what the pairs script makes of the result
    positions, widths = positions_and_widths(marker.pairs)
    rows = pair_positions_results(positions, fsr_mhz=417.0, widths=widths)
    summary = pair_summary(rows)
    assert abs(summary['df_over_fsr_mean'] - df / fsr) < 1e-5, summary
    assert abs(summary['fwhm_0_MHz_mean'] - 2 * gamma / fsr * 417.0) < 1e-2, summary

    # the single-peak modes still undo one peak (and its curve) per press,
    # dropping the pair a peak leaves empty
    marker.on_key(key_d)
    assert [len(pair) for pair in marker.pairs] == [2, 1], marker.pairs
    marker.on_key(key_d)
    assert [len(pair) for pair in marker.pairs] == [2], marker.pairs
    assert len(marker.ax.lines) == curves - 3, len(marker.ax.lines)

    # undoing everything gets the window back to how it started, colours
    # included, and 'd' on an unmarked window does nothing
    while marker.undo_stack:
        marker.on_key(key_d)
    marker.on_key(key_d)
    assert marker.pairs == [[]], marker.pairs
    assert marker.color_index == 0, marker.color_index
    assert len(marker.ax.lines) == bare_lines, len(marker.ax.lines)

    # Enter finishes the marking - it closes the window run() blocks on
    marker.on_key(type('Event', (), {'key': 'enter'})())
    assert not plt.fignum_exists(marker.fig.number)

    # and a window finished with nothing marked yields the empty list that
    # tells mode_map_2d to skip the file
    assert _PairMarker(x, y).run() == []

    print(f"mode_marking self-test passed: df/FSR = "
          f"{summary['df_over_fsr_mean']:.4f} (expected {df / fsr:.4f}), "
          f"FWHM0 = {summary['fwhm_0_MHz_mean']:.4f} MHz")


if __name__ == '__main__':
    matplotlib.use('Agg')  # the self-test never opens a window
    _self_test()
