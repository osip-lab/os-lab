"""2D map of resonator transmission spectra versus a swept cavity parameter.

Give it a dictionary {parameter value: PicoScope file}, mark every file the way
pico_scope/extract_df_and_fsr_from_scope_csv.py has you mark one, and it draws
all of them as the rows of a single 2D map:

    x      frequency detuning from the 0th-order mode [MHz]
    y      the dictionary key (the swept parameter - see Y_AXIS_LABEL)
    color   transmitted intensity, 1.0 = the row's own reference peak
            (its 0th- or its 1st-order mode - see NORMALIZE_TO)

The rows cannot simply be stacked: every trace was scanned at a different pace
(so a scope-second is a different number of MHz in each), sometimes upward and
sometimes downward in frequency, at a different laser power, with the 0th order
wherever it happened to fall in the record. Marking the same 0th/1st-order pair
over consecutive longitudinal mode numbers fixes all of that, because it gives
both the FSR in scope-seconds and (from the cavity length) the FSR in MHz:

    1. MHz per scope-second = FSR [MHz] / FSR [s]  -> every row is stretched or
       contracted so that 1 MHz is the same length on the x axis. The FSR in
       seconds is the one of the pair the row is drawn around (step 4), not the
       average over the record: a scan whose pace drifts over the record has a
       different one in every FSR, and only its own gets that window right.
    2. The 0th order of the reference pair becomes x = 0 -> the 0th orders of
       all rows line up in one vertical ridge.
    3. The row is flipped when needed so the 1st order is to the RIGHT of the
       0th order, whichever way the scan ran.
    4. The row is trimmed to one FSR, from 3 0th-order FWHMs before the 0th
       order to 3 FWHMs before the next one (TRIM_WIDTHS_BEFORE_ZEROTH).
    5. The row is normalized: (y - baseline) / reference peak height, the
       reference being the 0th- or the 1st-order mode (NORMALIZE_TO).

Usage
-----
    edit MEASUREMENTS (and Y_AXIS_LABEL) below, then

    python pico_scope/mode_map_2d.py                       draw the map
    python pico_scope/mode_map_2d.py --normalize-to first  ... with 1 = the
                                                           1st-order peak
    python pico_scope/mode_map_2d.py --self-test           synthetic check,
                                                           no GUI/files

Marking is slow, so every file's marks are cached in a
'<data file>.modemarks.json' sidecar next to the data - the same sidecar
pico_scope/extract_df_and_fsr_from_scope_csv.py writes, so a file already
marked there while the measurement was being taken needs no marking here at
all. When a file already has one, the run stops to ask whether to use that
marking ('y'), to mark the file again ('n', which then replaces the
sidecar), or to use it and every remaining cached marking without asking
again ('a'); set REMARK_ALL, or list keys in REMARK_KEYS, to skip the
question and mark those files again. A marking with fewer than the two
consecutive pairs a map row needs for its FSR is marked again without asking.

Finishing a marking window without marking anything asks again which waveform
buffer of that file to use - the usual reason for an unmarkable trace being
that the wrong buffer was picked. Another buffer marks the file with that one
instead; -1 skips the file, which is then left out of the map (and any sidecar
it already had is left as it was).
"""

# %% [Step 0] Imports and configuration -------------------------------------
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use('Agg' if '--self-test' in sys.argv else 'Qt5Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# repo root on sys.path so `pico_scope.*` / `utilities.*` resolve even when
# this file is run directly (Python only puts the script's own folder on
# sys.path, not the repo root the absolute imports assume).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pico_scope.mode_analysis import cavity_fsr_mhz  # noqa: E402
from pico_scope.mode_marking import (SELECTION_INSTRUCTIONS,  # noqa: E402
                                     mark_pairs)
# The sidecar itself lives in pico_scope.mode_marks_cache, shared with
# pico_scope/extract_df_and_fsr_from_scope_csv.py - a file marked there while
# the measurement was being taken is reused here rather than marked again.
from pico_scope.mode_marks_cache import (ask_use_cached_marks,  # noqa: E402
                                         complete_pairs, load_cached_marks,
                                         make_record, resolve_csv, save_marks,
                                         trace_csv_path)
from utilities.utils import (ask_long_arm_length,  # noqa: E402
                             choose_buffer_csv, psdata_buffer_csvs)

# --- the measurements to map (this is the dictionary to edit) --------------
# {y-axis value: PicoScope trace}. .psdata files are converted to CSV on the
# fly (and you are asked which waveform buffer to use); .csv files are read as
# they are. The keys need not be evenly spaced - the map keeps their spacing.
MEASUREMENTS = {
    37: r"C:\Users\michaeka\Weizmann Institute Dropbox\Michael Kali\Labs Dropbox\Laser Phase Plate\Daily measurements and notes\2026-08-27\37cm\without EOM 2-0013.psdata",
    # 38: r"C:\Users\michaeka\Weizmann Institute Dropbox\Michael Kali\Labs Dropbox\Laser Phase Plate\Daily measurements and notes\2026-08-27\38cm\without EOM 2-0012.psdata",
    # 39: r"C:\Users\michaeka\Weizmann Institute Dropbox\Michael Kali\Labs Dropbox\Laser Phase Plate\Daily measurements and notes\2026-08-27\39cm\without EOM 2-0010.psdata",
    40: r"C:\Users\michaeka\Weizmann Institute Dropbox\Michael Kali\Labs Dropbox\Laser Phase Plate\Daily measurements and notes\2026-08-27\40cm\without EOM 2-0008.psdata",
    # 41: r"C:\Users\michaeka\Weizmann Institute Dropbox\Michael Kali\Labs Dropbox\Laser Phase Plate\Daily measurements and notes\2026-08-27\41cm\without EOM-0007.psdata",
    42: r"C:\Users\michaeka\Weizmann Institute Dropbox\Michael Kali\Labs Dropbox\Laser Phase Plate\Daily measurements and notes\2026-08-27\42cm\without EOM-0006.psdata",
    43: r"C:\Users\michaeka\Weizmann Institute Dropbox\Michael Kali\Labs Dropbox\Laser Phase Plate\Daily measurements and notes\2026-08-27\43cm\without EOM-0005.psdata",
    44: r"C:\Users\michaeka\Weizmann Institute Dropbox\Michael Kali\Labs Dropbox\Laser Phase Plate\Daily measurements and notes\2026-08-27\44cm\without EOM-0005.psdata",
    45: r"C:\Users\michaeka\Weizmann Institute Dropbox\Michael Kali\Labs Dropbox\Laser Phase Plate\Daily measurements and notes\2026-08-27\45cm\without EOM-0003.psdata",
    46: r"C:\Users\michaeka\Weizmann Institute Dropbox\Michael Kali\Labs Dropbox\Laser Phase Plate\Daily measurements and notes\2026-08-27\46cm\without EOM-0003.psdata",
    47: r"C:\Users\michaeka\Weizmann Institute Dropbox\Michael Kali\Labs Dropbox\Laser Phase Plate\Daily measurements and notes\2026-08-27\47cm\without EOM-0002.psdata",
    # 48: r"C:\Users\michaeka\Weizmann Institute Dropbox\Michael Kali\Labs Dropbox\Laser Phase Plate\Daily measurements and notes\2026-08-27\48cm\without EOM.psdata",
    49: r"C:\Users\michaeka\Weizmann Institute Dropbox\Michael Kali\Labs Dropbox\Laser Phase Plate\Daily measurements and notes\2026-08-27\49cm\without EOM-0001.psdata",
    50: r"C:\Users\michaeka\Weizmann Institute Dropbox\Michael Kali\Labs Dropbox\Laser Phase Plate\Daily measurements and notes\2026-08-27\50cm max linear stage\without EOM-0003.psdata",
}
Y_AXIS_LABEL = 'Long arm length'  # sometimes 'Short arm length'

# --- the cavity being measured (edit this when the setup changes) ----------
# Only the FSR is needed here (it is the frequency ruler), so no element list:
# FSR = c / 2L with L = long arm + mid arm + short arm. The long arm is asked
# for once per file, as in the sibling scripts, so it is not configured here.
MID_ARM_LENGTH = 1.5e-2           # [m] only used by 4-element cavities
SHORT_ARM_LENGTH = 0.7e-2         # [m] near mirror -> lens

# --- analysis / plotting knobs ---------------------------------------------
TIME_COLUMN = 'Time'              # x-axis column in the PicoScope CSV
SIGNAL_COLUMN = 'Channel D'       # intensity column to analyze
TRIM_WIDTHS_BEFORE_ZEROTH = 3.0   # margin before the 0th order, in its FWHMs
# Which mode's peak height is 1.0 in every row: 'zeroth' or 'first'. The 0th
# order is the natural ruler, but when the map is about the 1st order - a 0th
# order that saturates the detector, or rows whose 0th order is barely there -
# normalizing to the 1st order is what makes the rows comparable. Note that
# with 'first' the 0th order rises well above 1, so COLOR_LIMITS below has to
# be widened (or set to None) for it not to saturate the color scale.
NORMALIZE_TO = 'zeroth'
N_X = 2000                        # columns of the common frequency grid
CMAP = 'viridis'
COLOR_LIMITS = (0.0, 1.0)         # (vmin, vmax); None for autoscale
OVERLAY_FIRST_ORDER = True        # mark the fitted 0th->1st spacing per row
SAVE_OUTPUTS = True               # write the figure and the map arrays to disk
REMARK_ALL = False                # mark every file again, without asking
REMARK_KEYS = ()                  # ... or only these keys, e.g. (36.0,)

# The marker's own instructions plus the way out of a file that cannot be
# marked - only this script, which walks a whole dictionary, has a next file.
MARKING_INSTRUCTIONS = (SELECTION_INSTRUCTIONS +
                        "  Finish with nothing marked to pick another "
                        "waveform buffer, or to skip this file.")

# The accepted values of NORMALIZE_TO, and how each one is named to the reader
NORMALIZATION_REFERENCES = {'zeroth': '0th-order', 'first': '1st-order'}


def check_normalize_to(value):
    """Return `value` if it names a normalization reference, else complain."""
    if value not in NORMALIZATION_REFERENCES:
        raise ValueError(f"normalize_to must be one of "
                         f"{sorted(NORMALIZATION_REFERENCES)}, not {value!r}")
    return value


def normalize_to_from_argv(argv=None):
    """The `--normalize-to zeroth|first` override, or NORMALIZE_TO if absent."""
    argv = sys.argv if argv is None else argv
    if '--normalize-to' not in argv:
        return NORMALIZE_TO
    index = argv.index('--normalize-to') + 1
    if index >= len(argv):
        raise ValueError("--normalize-to needs a value: "
                         f"{' or '.join(sorted(NORMALIZATION_REFERENCES))}")
    return check_normalize_to(argv[index])


# %% [Step 1] Loading, marking and caching one file --------------------------
def csv_candidates(path):
    """Every CSV that could be marked for `path`, in waveform-buffer order.

    One entry unless `path` is a .psdata holding several waveform buffers -
    and those are the alternatives offered when the buffer that was picked
    turns out to be the wrong one (see analyse_file).
    """
    path = Path(path)
    if path.suffix.lower() == '.psdata':
        return [Path(csv) for csv in psdata_buffer_csvs(path)]
    return [path]


def load_trace(csv_path):
    """(time, intensity) arrays from a PicoScope CSV export."""
    # Rows 1 and 2 of a PicoScope export are the unit / blank header rows.
    raw = pd.read_csv(csv_path, skiprows=[1, 2])
    raw = raw.loc[:, [TIME_COLUMN, SIGNAL_COLUMN]].dropna()
    return (raw[TIME_COLUMN].to_numpy(dtype=float),
            raw[SIGNAL_COLUMN].to_numpy(dtype=float))


def analyse_file(key, data_path, remark=False, accept_all=None):
    """Return the marking record for one measurement, marking it if needed.

    The record is the '<data file>.modemarks.json' sidecar (see
    pico_scope.mode_marks_cache), so a file already marked at the bench with
    pico_scope/extract_df_and_fsr_from_scope_csv.py needs no marking here. A
    map row needs two consecutive pairs for its FSR, which is more than that
    script needs, so a cached marking with fewer is marked again.

    `accept_all` is the one-item mutable flag main() shares across the whole
    dictionary (see ask_use_cached_marks): once set, every remaining cached
    file is used without asking.

    Finishing the marking window without anything marked means the trace on
    screen is not the one to analyse, so the waveform-buffer question is asked
    again: another buffer marks this file with that buffer instead, -1 gives
    up on the file. Giving up returns None - the file is then left out of the
    map and its sidecar, if it has one, is left untouched.
    """
    data_path = Path(data_path)
    if not data_path.is_file():
        raise FileNotFoundError(f"{Y_AXIS_LABEL} = {key:g}: no such file: {data_path}")

    if not remark:
        cached = load_cached_marks(data_path, min_pairs=2,
                                   signal_column=SIGNAL_COLUMN)
        if cached is not None and ask_use_cached_marks(cached, data_path,
                                                        accept_all=accept_all):
            print("  using the cached marks")
            return cached

    candidates = csv_candidates(data_path)
    csv_path = choose_buffer_csv(candidates)
    while True:
        x, y = load_trace(csv_path)
        print(f"  loaded {len(x)} samples from '{SIGNAL_COLUMN}'")

        # the folder is part of the title too: the file names repeat across
        # measurements, so the name alone does not say which trace this is
        marks = mark_pairs(x, y, title=f"{Y_AXIS_LABEL} = {key:g}   |   "
                                       f"{data_path.parent.name}/{data_path.name}",
                           instructions=MARKING_INSTRUCTIONS)
        if marks:
            break
        # nothing was marked: usually the wrong waveform buffer is on screen,
        # so offer the others again rather than losing the file over it
        print(f"  nothing was marked on '{Path(csv_path).name}'")
        csv_path = choose_buffer_csv(candidates, allow_skip=True)
        if csv_path is None:
            return None

    marks = complete_pairs(marks)
    if len(marks) < 2:
        # something was marked, so this is a half-finished marking rather than
        # a skip, and silently dropping the file would hide the mistake
        raise ValueError(
            f"{Y_AXIS_LABEL} = {key:g}: only {len(marks)} complete pair(s) were "
            "marked; at least two consecutive ones are needed for the FSR. "
            "Finish the marking window with nothing marked to pick another "
            "waveform buffer, or to skip the file.")

    # asked only now, so that skipping a file costs no answer
    long_arm = ask_long_arm_length()

    record = make_record(data_path, csv_path, marks, long_arm,
                         signal_column=SIGNAL_COLUMN, key=key)
    save_marks(data_path, record)
    return record


# %% [Step 2] One trace -> one row of the map --------------------------------
def ran_away(peak):
    """Whether this marked peak's fit landed on something that is not a peak.

    mode_marking fits an area, so a fit that ran away - onto a trough, or onto
    nothing at all - comes back with a negative area and with it a negative
    height (and usually a nonsense width beside it). Taken at face value such a
    fit divides the row by a negative number, turning that row of the map
    upside down, so it counts here as no fit at all and the same mode's fits in
    the other marked FSRs stand in for it.
    """
    return peak['height'] is not None and peak['height'] <= 0


def build_row(x, y, marks, fsr_mhz, trim_widths=TRIM_WIDTHS_BEFORE_ZEROTH,
              normalize_to=None):
    """Turn a marked trace into a frequency-aligned, normalized map row.

    `marks` are mode_marking's pairs (0th order first, then the 1st order that
    follows it) over consecutive longitudinal mode numbers; `fsr_mhz` is the
    cavity's free spectral range, which is what turns the scope's time axis
    into a frequency axis. `normalize_to` ('zeroth' or 'first', NORMALIZE_TO by
    default) picks the mode whose peak height becomes 1.0.

    Everything is measured on the reference pair - the one whose FSR the
    trimmed row covers - so that the numbers describe what the row shows even
    when the scan pace drifts over the record: 'fsr_s' is that pair's own FSR
    and 'df_mhz' its own 0th->1st spacing. The averages over all the marked
    pairs come along as 'fsr_s_mean' and 'df_mhz_mean' for comparison.

    Returns {'f': [MHz from the 0th order], 'i': [normalized intensity],
             'df_mhz', 'df_mhz_mean', 'fwhm_0_mhz', 'fwhm_1_mhz', 'fsr_mhz',
             'fsr_s', 'fsr_s_mean', 'mhz_per_s', 'flipped', 'normalized_to'}.
    """
    normalize_to = check_normalize_to(NORMALIZE_TO if normalize_to is None
                                      else normalize_to)
    pairs = sorted((pair for pair in marks if len(pair) == 2),
                   key=lambda pair: pair[0]['x0'])
    if len(pairs) < 2:
        raise ValueError('need at least two marked pairs to get an FSR')

    t_zero = np.array([pair[0]['x0'] for pair in pairs], dtype=float)
    intervals = np.diff(t_zero)  # one FSR each, in scope-seconds
    if np.any(intervals == 0):
        raise ValueError('two pairs share the same 0th-order position')

    # Which way the scan ran: +1 when the 1st order sits at a later time than
    # its 0th order. Unanimous in practice, so a majority vote is only a guard
    # against a single mismarked pair.
    votes = np.sign([pair[1]['x0'] - pair[0]['x0'] for pair in pairs])
    direction = 1.0 if votes.sum() >= 0 else -1.0

    # The reference is the pair with a neighbouring 0th order on the +f side -
    # the trimmed window runs one FSR in that direction. With direction = +1
    # (frequency rising with time) that is the earliest pair, otherwise the
    # latest one.
    reference_index = 0 if direction > 0 else len(pairs) - 1
    reference_pair = pairs[reference_index]
    zeroth, first = reference_pair

    # The pace is the reference pair's OWN FSR - the one the trimmed window
    # covers - not the average over the whole record. The scan is not always
    # linear: in a record whose pace drifts by several percent per FSR, an
    # averaged pace puts this window's modes at the wrong detuning (and the
    # 0th->1st spacing drawn over the map then misses the ridge it describes).
    # The neighbour on the +f side is the next 0th order in time when the scan
    # runs upward, the previous one when it runs downward.
    fsr_s = float(intervals[reference_index if direction > 0
                            else reference_index - 1])
    fsr_s_mean = float(np.mean(intervals))
    mhz_per_s = fsr_mhz / abs(fsr_s)  # the scan pace, in MHz per scope-second
    drift = abs(fsr_s - fsr_s_mean) / abs(fsr_s_mean)
    if drift > 0.02:
        print(f"  the scan pace drifts: this window's FSR is {fsr_s:.4g} s "
              f"against {fsr_s_mean:.4g} s averaged over the record "
              f"({100 * drift:.1f}% apart) - the window's own pace is used")

    # The width is signed only by accident (a Lorentzian is symmetric in it),
    # so it is the magnitude that means anything.
    def widths(index):
        return [abs(pair[index]['gamma']) for pair in pairs
                if pair[index]['gamma'] is not None and not ran_away(pair[index])]

    gammas = widths(0)
    gamma = (abs(zeroth['gamma'])
             if zeroth['gamma'] is not None and not ran_away(zeroth)
             else (float(np.median(gammas)) if gammas else None))
    if gamma is None:
        fwhm_0_mhz = fsr_mhz / 200.0
        print("  no usable 0th-order width (the positions were clicked rather "
              "than fitted, or the fits did not land on a peak) - trimming "
              f"with a placeholder FWHM of {fwhm_0_mhz:.4g} MHz")
    else:
        fwhm_0_mhz = 2.0 * gamma * mhz_per_s
    gammas_1 = widths(1)
    fwhm_1_mhz = (2.0 * float(np.median(gammas_1)) * mhz_per_s
                  if gammas_1 else float('nan'))

    # Time -> frequency: 0th order at 0, 1st order always at positive f.
    f = direction * (np.asarray(x, dtype=float) - zeroth['x0']) * mhz_per_s

    margin = trim_widths * fwhm_0_mhz
    keep = (f >= -margin) & (f <= fsr_mhz - margin)
    if not keep.any():
        raise ValueError('the trimming window holds no samples')
    order = np.argsort(f[keep])  # also performs the flip when direction = -1
    f_row = f[keep][order]
    i_row = np.asarray(y, dtype=float)[keep][order]

    # Normalization only rescales the row - nothing moves on the frequency
    # axis - so the reference is free to be either mode of the reference pair.
    order_index = 0 if normalize_to == 'zeroth' else 1
    reference = reference_pair[order_index]
    height, baseline = reference['height'], reference['y0']
    if not height or ran_away(reference):
        # This one fit is unusable, but the same mode was marked in the other
        # FSRs of the same record and is the same height there, so those stand
        # in rather than the row losing its common scale.
        stand_ins = [pair[order_index] for pair in pairs
                     if pair[order_index]['height'] is not None
                     and pair[order_index]['height'] > 0
                     and pair[order_index]['y0'] is not None]
        height = (float(np.median([peak['height'] for peak in stand_ins]))
                  if stand_ins else None)
        baseline = (float(np.median([peak['y0'] for peak in stand_ins]))
                    if stand_ins else None)
        if height:
            fit = ('no fitted height' if reference['height'] is None else
                   f"a fitted height of {reference['height']:.4g}")
            print(f"  the {NORMALIZATION_REFERENCES[normalize_to]} mode of this "
                  f"row's own FSR has {fit} - normalizing by the median over "
                  f"the {len(stand_ins)} marked FSR(s) that did fit "
                  f"({height:.4g}) instead. Mark this file again when you can.")
    if height and baseline is not None:
        i_row = (i_row - baseline) / height
    else:
        span = float(np.ptp(i_row)) or 1.0
        print(f"  no usable {NORMALIZATION_REFERENCES[normalize_to]} height "
              "anywhere in this file - normalizing this row by its own min/max "
              "instead")
        i_row = (i_row - float(np.min(i_row))) / span

    # The spacing of the reference pair, so that it describes the very FSR the
    # row shows; the average over all the marked pairs is carried along beside
    # it as a cross-check (print_table prints both), being the steadier number
    # when the scan is linear and the marking noisy.
    df_mhz = abs(first['x0'] - zeroth['x0']) * mhz_per_s
    df_mhz_mean = float(np.mean([abs(pair[1]['x0'] - pair[0]['x0'])
                                 for pair in pairs])) * mhz_per_s
    return {'f': f_row, 'i': i_row,
            'df_mhz': df_mhz,
            'df_mhz_mean': df_mhz_mean,
            'fwhm_0_mhz': fwhm_0_mhz,
            'fwhm_1_mhz': fwhm_1_mhz,
            'fsr_mhz': fsr_mhz,
            'fsr_s': fsr_s,
            'fsr_s_mean': fsr_s_mean,
            'mhz_per_s': mhz_per_s,
            'flipped': direction < 0,
            'normalized_to': normalize_to,
            'n_pairs': len(pairs)}


# %% [Step 3] Rows -> a common grid ------------------------------------------
def common_grid(rows, n_x=N_X):
    """Cell centres and edges of the frequency grid covering every row.

    Rows can span different ranges: a different long arm gives a different FSR,
    and a different linewidth a different trimming margin.
    """
    f_min = min(float(row['f'][0]) for row in rows)
    f_max = max(float(row['f'][-1]) for row in rows)
    edges = np.linspace(f_min, f_max, n_x + 1)
    return 0.5 * (edges[:-1] + edges[1:]), edges


def resample_row(row, centres, edges):
    """The row's intensity on the common grid: the mean of the samples in each
    cell, interpolated in the cells a sparse row leaves empty, NaN outside the
    row's own range (so the map shows blank there rather than a made-up
    color)."""
    f, i = row['f'], row['i']
    counts, _ = np.histogram(f, bins=edges)
    sums, _ = np.histogram(f, bins=edges, weights=i)
    with np.errstate(invalid='ignore'):
        values = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
    inside = (centres >= f[0]) & (centres <= f[-1])
    gaps = inside & (counts == 0)
    if gaps.any():
        values[gaps] = np.interp(centres[gaps], f, i)
    values[~inside] = np.nan
    return values


def cell_edges(values):
    """Edges of unevenly spaced cells centred on `values` (sorted): midpoints
    between neighbours, the outer edges half a gap beyond the ends."""
    values = np.asarray(values, dtype=float)
    if len(values) == 1:
        half = 0.5 if values[0] == 0 else abs(values[0]) * 0.05
        return np.array([values[0] - half, values[0] + half])
    middles = 0.5 * (values[:-1] + values[1:])
    return np.concatenate(([values[0] - (middles[0] - values[0])],
                           middles,
                           [values[-1] + (values[-1] - middles[-1])]))


def build_map(keys, rows, n_x=N_X):
    """Sort the rows by key and resample them onto one grid.

    Returns (keys, rows, centres, x_edges, y_edges, Z), all sorted by key, with
    Z masked wherever nothing was measured.
    """
    order = np.argsort(np.asarray(keys, dtype=float))
    keys = np.asarray(keys, dtype=float)[order]
    rows = [rows[index] for index in order]
    centres, x_edges = common_grid(rows, n_x=n_x)
    z = np.vstack([resample_row(row, centres, x_edges) for row in rows])
    return keys, rows, centres, x_edges, cell_edges(keys), np.ma.masked_invalid(z)


# %% [Step 4] The map --------------------------------------------------------
def plot_map(keys, rows, x_edges, y_edges, z, y_label=Y_AXIS_LABEL):
    fig, ax = plt.subplots(figsize=(9, 6))
    vmin, vmax = COLOR_LIMITS if COLOR_LIMITS else (None, None)
    mesh = ax.pcolormesh(x_edges, y_edges, z, cmap=CMAP, shading='flat',
                         vmin=vmin, vmax=vmax)
    reference = NORMALIZATION_REFERENCES[rows[0]['normalized_to']]
    fig.colorbar(mesh, ax=ax, label=f'Transmission (1 = {reference} peak)')

    ax.axvline(0.0, color='w', ls='--', lw=0.8, alpha=0.6)
    if OVERLAY_FIRST_ORDER:
        ax.plot([row['df_mhz'] for row in rows], keys, 'o--', color='w',
                lw=0.8, ms=3, alpha=0.7, label='fitted 0th->1st spacing')
        legend = ax.legend(loc='upper right', framealpha=0.75)
        legend.get_frame().set_facecolor('0.15')
        for text in legend.get_texts():
            text.set_color('w')

    ax.set_xlabel('Frequency detuning from the 0th-order mode [MHz]')
    ax.set_ylabel(y_label)
    ax.set_title(f"Transmission spectra vs {y_label.lower()} "
                 f"({len(keys)} measurements)")
    fig.tight_layout()
    return fig


def plot_trends(keys, rows, y_label=Y_AXIS_LABEL):
    """The map's two readings as numbers, against the swept parameter.

    Upper axis: the width of the 0th- and the 1st-order mode, together on one
    axis so they can be compared directly. Lower axis: the spacing between
    them. Both are fitted quantities - what the map shows as ridges, this
    shows as curves.
    """
    fig, (ax_width, ax_spacing) = plt.subplots(2, 1, sharex=True,
                                               figsize=(8, 7))

    ax_width.plot(keys, [row['fwhm_0_mhz'] for row in rows], 'o-',
                  color='tab:blue', label='0th order')
    fwhm_1 = [row['fwhm_1_mhz'] for row in rows]
    # nan when the first-order peaks were clicked rather than fitted: no line
    # to draw, and a legend entry for it would only be misleading
    if np.any(np.isfinite(fwhm_1)):
        # dashed: when the two modes happen to have the same width the lines
        # sit on top of each other, and a solid one would simply hide the other
        ax_width.plot(keys, fwhm_1, 's--', color='tab:orange',
                      label='1st order')
    ax_width.set_ylabel('Mode width, FWHM [MHz]')
    ax_width.legend()
    ax_width.grid(alpha=0.3)
    ax_width.set_title(f"Mode width and spacing vs {y_label.lower()}")

    ax_spacing.plot(keys, [row['df_mhz'] for row in rows], 'o-',
                    color='tab:green')
    ax_spacing.set_ylabel('0th -> 1st mode spacing [MHz]')
    ax_spacing.set_xlabel(y_label)
    ax_spacing.grid(alpha=0.3)

    fig.tight_layout()
    return fig


def print_table(keys, rows, y_label=Y_AXIS_LABEL):
    # 'df' is the row's own spacing, the one the map draws; '<df>' the average
    # over every marked pair. They part company when the scan pace drifts, and
    # seeing both is what says so.
    header = (f"{y_label:>18} | {'df [MHz]':>10} | {'<df> [MHz]':>10} | "
              f"{'FWHM0 [MHz]':>11} | "
              f"{'FWHM1 [MHz]':>11} | {'FSR [MHz]':>10} | {'MHz/s':>10} | scan")
    print()
    print(header)
    print('-' * len(header))
    for key, row in zip(keys, rows):
        print(f"{key:>18g} | {row['df_mhz']:>10.4f} | "
              f"{row['df_mhz_mean']:>10.4f} | {row['fwhm_0_mhz']:>11.4f} | "
              f"{row['fwhm_1_mhz']:>11.4f} | {row['fsr_mhz']:>10.2f} | "
              f"{row['mhz_per_s']:>10.4g} | "
              f"{'down (flipped)' if row['flipped'] else 'up'}")
    print()


def save_outputs(folder, map_fig, trend_fig, keys, rows, centres, z):
    """Write both figures and the map arrays next to the measurements."""
    folder = Path(folder)
    stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    png_path = folder / f"mode_map_2d_{stamp}.png"
    trend_path = folder / f"mode_trends_{stamp}.png"
    npz_path = folder / f"mode_map_2d_{stamp}.npz"
    map_fig.savefig(png_path, dpi=200)
    trend_fig.savefig(trend_path, dpi=200)
    np.savez(npz_path, keys=keys, f_mhz=centres, intensity=z.filled(np.nan),
             df_mhz=[row['df_mhz'] for row in rows],
             df_mhz_mean=[row['df_mhz_mean'] for row in rows],
             fwhm_0_mhz=[row['fwhm_0_mhz'] for row in rows],
             fwhm_1_mhz=[row['fwhm_1_mhz'] for row in rows],
             fsr_mhz=[row['fsr_mhz'] for row in rows],
             normalized_to=rows[0]['normalized_to'],
             y_axis_label=Y_AXIS_LABEL)
    print(f"Saved {png_path}")
    print(f"Saved {trend_path}")
    print(f"Saved {npz_path}")


# %% [Step 5] The run --------------------------------------------------------
def main(measurements=None, y_label=None, normalize_to=None):
    """Mark every measurement (or reuse its cache) and draw the map."""
    measurements = MEASUREMENTS if measurements is None else measurements
    y_label = Y_AXIS_LABEL if y_label is None else y_label
    normalize_to = check_normalize_to(NORMALIZE_TO if normalize_to is None
                                      else normalize_to)
    print(f"Rows normalized to their {NORMALIZATION_REFERENCES[normalize_to]} "
          "peak")
    if not measurements:
        raise ValueError(
            'MEASUREMENTS is empty - fill in {parameter value: file path} at '
            'the top of pico_scope/mode_map_2d.py.')

    keys, rows = [], []
    accept_all = [False]  # shared across files: 'a' at the cache prompt sets it
    for index, key in enumerate(sorted(measurements), start=1):
        path = measurements[key]
        # the full path, not just the name: many measurements share a file
        # name and only the folder says which one is on screen
        print(f"\n=== [{index}/{len(measurements)}] {y_label} = {key:g} ===")
        print(f"  {path}")
        record = analyse_file(
            key, path, remark=REMARK_ALL or key in REMARK_KEYS,
            accept_all=accept_all)
        if record is None:
            print(f"  {y_label} = {key:g} skipped")
            continue

        x, y = load_trace(resolve_csv(record, path))
        fsr_mhz = cavity_fsr_mhz(long_arm=record['long_arm_m'],
                                 mid_arm=MID_ARM_LENGTH,
                                 short_arm=SHORT_ARM_LENGTH)
        row = build_row(x, y, record['marks'], fsr_mhz,
                        normalize_to=normalize_to)
        print(f"  {row['n_pairs']} pairs, FSR = {fsr_mhz:.2f} MHz "
              f"({row['fsr_s']:.4g} s), df = {row['df_mhz']:.4f} MHz "
              f"(averaged over the pairs: {row['df_mhz_mean']:.4f} MHz), "
              f"0th-order FWHM = {row['fwhm_0_mhz']:.4f} MHz"
              f"{', scan flipped' if row['flipped'] else ''}")
        keys.append(float(key))
        rows.append(row)

    if not rows:
        raise ValueError('every measurement was skipped - there is nothing to '
                         'map. Mark at least two pairs in one file.')
    if len(rows) < len(measurements):
        print(f"\n{len(measurements) - len(rows)} of {len(measurements)} "
              "measurements were skipped")

    keys, rows, centres, x_edges, y_edges, z = build_map(keys, rows)
    print_table(keys, rows, y_label=y_label)
    fig = plot_map(keys, rows, x_edges, y_edges, z, y_label=y_label)
    trend_fig = plot_trends(keys, rows, y_label=y_label)
    if SAVE_OUTPUTS:
        # the date folder holding every measurement's own subfolder, not the
        # first measurement's own subfolder (e.g. .../2026-08-27, not
        # .../2026-08-27/37cm)
        first_path = next(path for path in measurements.values()
                          if path is not None)
        save_outputs(Path(first_path).parent.parent,
                     fig, trend_fig, keys, rows, centres, z)
    plt.show(block=True)
    return keys, rows, centres, z


# %% [Step 6] Self-test (synthetic data; no scope, no GUI, no files) ---------
def _synthetic_trace(fsr_mhz, df_mhz, fwhm_mhz, mhz_per_s, descending,
                     amplitude, offset, n_fsr=3, n_samples=40000,
                     first_amplitude=1.0, pace_drift=1.0):
    """A fake scan and its exact marks, as the marker would have fitted them.

    Returns (x, y, marks). The scan covers `n_fsr` free spectral ranges, runs
    up or down in frequency, and carries a 0th/1st-order Lorentzian pair per
    FSR - so the row builder gets everything it needs and the right answers are
    known exactly. `first_amplitude` scales the 1st order relative to the 0th,
    which is what tells the two normalization references apart. `pace_drift`
    is how much longer each FSR takes than the one before it (1.0 = a perfectly
    linear scan, `mhz_per_s` then being the pace everywhere); it is the drift
    real records show, and what tells a per-window pace from an averaged one.
    """
    from pico_scope.mode_analysis import area_lorentzian
    from pico_scope.mode_marking import peak_record

    f_lin = np.linspace(-0.3 * fsr_mhz, (n_fsr - 0.5) * fsr_mhz, n_samples)

    def pace_at(f_mhz):
        """The scan pace [MHz/s] where the scan is at `f_mhz`."""
        return mhz_per_s / pace_drift ** (np.asarray(f_mhz) / fsr_mhz)

    # time = the integral of 1 / pace over the frequencies already scanned,
    # which for pace_drift = 1 is the plain linear (f - f[0]) / mhz_per_s
    seconds_per_mhz = 1.0 / pace_at(f_lin)
    x = np.concatenate(([0.0], np.cumsum(
        0.5 * (seconds_per_mhz[1:] + seconds_per_mhz[:-1]) * np.diff(f_lin))))
    f_of_t = f_lin[::-1] if descending else f_lin

    gamma_mhz = fwhm_mhz / 2.0
    first_amp = amplitude * first_amplitude
    centres_mhz = [(k * fsr_mhz + shift, amp)
                   for k in range(n_fsr)
                   for shift, amp in ((0.0, amplitude), (df_mhz, first_amp))]
    y = np.full(n_samples, float(offset))
    for centre, amp in centres_mhz:
        y += area_lorentzian(f_of_t, centre, gamma_mhz,
                             amp * np.pi * gamma_mhz, 0.0)

    # frequency -> time, on whichever monotonic direction this scan ran
    f_sorted, t_sorted = (f_of_t[::-1], x[::-1]) if descending else (f_of_t, x)

    def time_of(centre_mhz):
        return float(np.interp(centre_mhz, f_sorted, t_sorted))

    def width_at(centre_mhz):
        """The peak's width in scope-seconds: MHz-wide, so pace-dependent."""
        return gamma_mhz / float(pace_at(centre_mhz))

    marks = []
    for k in range(n_fsr):
        zeroth_mhz, first_mhz = k * fsr_mhz, k * fsr_mhz + df_mhz
        gamma_0, gamma_1 = width_at(zeroth_mhz), width_at(first_mhz)
        pair = [peak_record(time_of(zeroth_mhz), gamma_0,
                            amplitude * np.pi * gamma_0, offset),
                peak_record(time_of(first_mhz), gamma_1,
                            first_amp * np.pi * gamma_1, offset)]
        marks.append(pair)
    if descending:
        marks.reverse()  # the marker returns them in scan (time) order
    return x, y, marks


def _self_test():
    fsr_mhz, df_mhz, fwhm_mhz = 400.0, 90.0, 4.0
    settings = [  # (mhz_per_s, descending, amplitude, offset)
        (1.0e5, False, 0.8, 0.02),
        (2.7e5, True, 0.15, -0.30),   # slower scan, downward, weak and offset
    ]
    rows = []
    for mhz_per_s, descending, amplitude, offset in settings:
        x, y, marks = _synthetic_trace(fsr_mhz, df_mhz, fwhm_mhz, mhz_per_s,
                                       descending, amplitude, offset)
        row = build_row(x, y, marks, fsr_mhz)
        rows.append(row)

        assert row['flipped'] == descending, row['flipped']
        assert abs(row['df_mhz'] - df_mhz) < 1e-6 * fsr_mhz, row['df_mhz']
        assert abs(row['fwhm_0_mhz'] - fwhm_mhz) < 1e-6 * fsr_mhz, row['fwhm_0_mhz']
        assert abs(row['fwhm_1_mhz'] - fwhm_mhz) < 1e-6 * fsr_mhz, row['fwhm_1_mhz']
        assert abs(row['mhz_per_s'] - mhz_per_s) < 1e-6 * mhz_per_s, row['mhz_per_s']

        # the window: one FSR, starting 3 FWHMs before the 0th order
        margin = TRIM_WIDTHS_BEFORE_ZEROTH * fwhm_mhz
        sample_mhz = float(np.median(np.diff(row['f'])))
        assert abs(row['f'][0] + margin) < 2 * sample_mhz, row['f'][0]
        assert abs(row['f'][-1] - (fsr_mhz - margin)) < 2 * sample_mhz, row['f'][-1]
        assert np.all(np.diff(row['f']) > 0), 'the row is not in frequency order'

        # the 0th order sits at f = 0 and is normalized to 1
        near_zero = np.abs(row['f']) < 0.1 * fsr_mhz
        peak_f = row['f'][near_zero][np.argmax(row['i'][near_zero])]
        assert abs(peak_f) < 2 * sample_mhz, peak_f
        assert abs(row['i'].max() - 1.0) < 0.02, row['i'].max()
        # ... and the 1st order is to the right of it, at df
        right = row['f'] > 0.5 * df_mhz
        first_f = row['f'][right][np.argmax(row['i'][right])]
        assert abs(first_f - df_mhz) < 2 * sample_mhz, first_f

    # both scans, however they ran, describe the same spectrum
    assert abs(rows[0]['df_mhz'] - rows[1]['df_mhz']) < 1e-6 * fsr_mhz

    # --- a scan whose pace drifts -----------------------------------------
    # Each FSR takes 8% longer than the one before, as the 38 cm record does.
    # The row covers the first FSR alone, so its numbers have to be that FSR's
    # own: with a pace averaged over the record the row would be stretched and
    # the spacing drawn over the map would miss the ridge it describes.
    drift = 1.08
    x, y, marks = _synthetic_trace(fsr_mhz, df_mhz, fwhm_mhz, 1.0e5, False,
                                   0.6, 0.0, pace_drift=drift)
    drifting = build_row(x, y, marks, fsr_mhz)
    intervals = np.diff([pair[0]['x0'] for pair in marks])
    assert abs(drifting['fsr_s'] - intervals[0]) < 1e-9, drifting['fsr_s']
    assert abs(drifting['fsr_s_mean'] - np.mean(intervals)) < 1e-9
    assert drifting['fsr_s'] < 0.97 * drifting['fsr_s_mean'], 'the pace did not drift'

    # the spacing the map draws sits on the row's own 1st-order ridge ...
    sample_mhz = float(np.median(np.diff(drifting['f'])))
    right = drifting['f'] > 0.5 * df_mhz
    ridge_f = drifting['f'][right][np.argmax(drifting['i'][right])]
    assert abs(ridge_f - drifting['df_mhz']) < 2 * sample_mhz,         (ridge_f, drifting['df_mhz'])
    # ... which the average over the three FSRs, being ~8% wider per FSR, does
    # not: that is the mismatch this reference-pair spacing exists to avoid.
    assert drifting['df_mhz_mean'] - ridge_f > 0.01 * fsr_mhz,         drifting['df_mhz_mean']
    # the row is still normalized and still starts at its 0th order
    assert abs(drifting['i'].max() - 1.0) < 0.02, drifting['i'].max()
    assert abs(drifting['f'][0] + TRIM_WIDTHS_BEFORE_ZEROTH
               * drifting['fwhm_0_mhz']) < 2 * sample_mhz

    # --- a fit that ran away ------------------------------------------------
    # The 0th order of the row's own FSR fitted as a trough (a negative
    # height), as one 43 cm record did. Normalizing by it would divide the row
    # by a negative number and turn it upside down; the same mode's fits in the
    # other marked FSRs have to stand in, giving back exactly the sound row.
    x, y, marks = _synthetic_trace(fsr_mhz, df_mhz, fwhm_mhz, 1.0e5, False,
                                   0.6, 0.05)
    healthy = build_row(x, y, marks, fsr_mhz)
    runaway = [[dict(peak) for peak in pair] for pair in marks]
    runaway[0][0].update(height=-4.6, gamma=-runaway[0][0]['gamma'])
    assert ran_away(runaway[0][0]) and not ran_away(marks[0][0])
    guarded = build_row(x, y, runaway, fsr_mhz)
    assert abs(guarded['i'].max() - 1.0) < 0.02, guarded['i'].max()
    assert np.allclose(guarded['f'], healthy['f']), 'the trimming window moved'
    assert np.allclose(guarded['i'], healthy['i']), 'the row was not restored'
    assert abs(guarded['fwhm_0_mhz'] - fwhm_mhz) < 1e-6 * fsr_mhz

    # a third measurement, at a longer cavity (a smaller FSR): its row is
    # shorter than the others, so the map has to leave the rest of it blank
    x, y, marks = _synthetic_trace(0.75 * fsr_mhz, df_mhz, fwhm_mhz, 1.4e5,
                                   False, 0.5, 0.0)
    rows.append(build_row(x, y, marks, 0.75 * fsr_mhz))

    # the map itself: unevenly spaced keys, sorted, with their own cell edges
    keys, rows, centres, x_edges, y_edges, z = build_map([12.0, 20.0, 12.5],
                                                         rows)
    assert np.allclose(keys, [12.0, 12.5, 20.0]), keys
    assert z.shape == (3, N_X), z.shape
    assert len(x_edges) == N_X + 1 and len(y_edges) == 4
    assert np.allclose(y_edges, [11.75, 12.25, 16.25, 23.75]), y_edges
    assert not np.ma.is_masked(z[:, N_X // 2]), 'the middle of the map is empty'
    # the short row (key 12.5, the 0.75 FSR one) stops before the right edge
    assert np.ma.is_masked(z[1, -1]) and not np.ma.is_masked(z[0, -1]), \
        'the short row was not masked where it has no data'

    edges = cell_edges([1.0])
    assert edges[0] < 1.0 < edges[1], edges

    # --- the normalization reference ---------------------------------------
    # One trace whose 1st order is 40% of its 0th: the two references give the
    # same row scaled by a known factor, and nothing else about it changes.
    ratio = 0.4
    x, y, marks = _synthetic_trace(fsr_mhz, df_mhz, fwhm_mhz, 1.0e5, False,
                                   0.6, 0.05, first_amplitude=ratio)
    to_zeroth = build_row(x, y, marks, fsr_mhz, normalize_to='zeroth')
    to_first = build_row(x, y, marks, fsr_mhz, normalize_to='first')
    assert to_zeroth['normalized_to'] == 'zeroth'
    assert to_first['normalized_to'] == 'first'
    assert np.allclose(to_zeroth['f'], to_first['f']), 'normalizing moved the row'
    first_window = to_zeroth['f'] > 0.5 * df_mhz
    assert abs(to_zeroth['i'].max() - 1.0) < 0.02, to_zeroth['i'].max()
    assert abs(to_zeroth['i'][first_window].max() - ratio) < 0.02
    assert abs(to_first['i'][first_window].max() - 1.0) < 0.02
    assert abs(to_first['i'].max() - 1.0 / ratio) < 0.05, to_first['i'].max()
    try:
        build_row(x, y, marks, fsr_mhz, normalize_to='second')
    except ValueError:
        pass
    else:
        raise AssertionError('an unknown normalization reference was accepted')
    assert normalize_to_from_argv(['x', '--normalize-to', 'first']) == 'first'
    assert normalize_to_from_argv(['x']) == NORMALIZE_TO

    # the trends figure: two width lines on one axis, one spacing line below
    figure = plot_trends(keys, rows, y_label='Long arm length')
    width_axis, spacing_axis = figure.axes
    assert len(width_axis.lines) == 2, len(width_axis.lines)
    assert len(spacing_axis.lines) == 1, len(spacing_axis.lines)
    assert np.allclose(spacing_axis.lines[0].get_xdata(), keys)
    assert np.allclose(spacing_axis.lines[0].get_ydata(),
                       [row['df_mhz'] for row in rows])
    plt.close(figure)
    print('mode_map_2d self-test passed')


if __name__ == '__main__':
    if '--self-test' in sys.argv:
        _self_test()
    else:
        main(normalize_to=normalize_to_from_argv())
