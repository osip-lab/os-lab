"""2D map of resonator transmission spectra versus a swept cavity parameter.

Give it a dictionary {parameter value: PicoScope file}, mark every file the way
pico_scope/extract_df_and_fsr_from_scope_csv.py has you mark one, and it draws
all of them as the rows of a single 2D map:

    x      frequency detuning from the 0th-order mode [MHz]
    y      the dictionary key (the swept parameter - see Y_AXIS_LABEL)
    color   transmitted intensity, 1.0 = the row's own 0th-order peak

The rows cannot simply be stacked: every trace was scanned at a different pace
(so a scope-second is a different number of MHz in each), sometimes upward and
sometimes downward in frequency, at a different laser power, with the 0th order
wherever it happened to fall in the record. Marking the same 0th/1st-order pair
over consecutive longitudinal mode numbers fixes all of that, because it gives
both the FSR in scope-seconds and (from the cavity length) the FSR in MHz:

    1. MHz per scope-second = FSR [MHz] / FSR [s]  -> every row is stretched or
       contracted so that 1 MHz is the same length on the x axis.
    2. The 0th order of the reference pair becomes x = 0 -> the 0th orders of
       all rows line up in one vertical ridge.
    3. The row is flipped when needed so the 1st order is to the RIGHT of the
       0th order, whichever way the scan ran.
    4. The row is trimmed to one FSR, from 3 0th-order FWHMs before the 0th
       order to 3 FWHMs before the next one (TRIM_WIDTHS_BEFORE_ZEROTH).
    5. The row is normalized: (y - baseline) / 0th-order peak height.

Usage
-----
    edit MEASUREMENTS (and Y_AXIS_LABEL) below, then

    python pico_scope/mode_map_2d.py              draw the map
    python pico_scope/mode_map_2d.py --self-test  synthetic check, no GUI/files

Marking is slow, so every file's marks are cached in a
'<data file>.modemarks.json' sidecar next to the data and reused on the next
run; set REMARK_ALL, or list keys in REMARK_KEYS, to mark them again.
"""

# %% [Step 0] Imports and configuration -------------------------------------
import json
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
from pico_scope.mode_marking import mark_pairs  # noqa: E402
from utilities.utils import ask_long_arm_length, psdata_to_csv  # noqa: E402

# --- the measurements to map (this is the dictionary to edit) --------------
# {y-axis value: PicoScope trace}. .psdata files are converted to CSV on the
# fly (and you are asked which waveform buffer to use); .csv files are read as
# they are. The keys need not be evenly spaced - the map keeps their spacing.
MEASUREMENTS = {
    44: r"C:\Users\michaeka\Weizmann Institute Dropbox\Michael Kali\Labs Dropbox\Laser Phase Plate\Daily measurements and notes\2026-08-23\44cm\04 43 44cm\without EOM-0003.psdata",
    45: r"C:\Users\michaeka\Weizmann Institute Dropbox\Michael Kali\Labs Dropbox\Laser Phase Plate\Daily measurements and notes\2026-08-23\45cm\04 43 45cm\without EOM-0002.psdata",
    42: r"C:\Users\michaeka\Weizmann Institute Dropbox\Michael Kali\Labs Dropbox\Laser Phase Plate\Daily measurements and notes\2026-08-23\42cm\04 43 42cm\without EOM-0002.psdata",
}
Y_AXIS_LABEL = 'Long arm length'  # sometimes 'Short arm length'

# --- the cavity being measured (edit this when the setup changes) ----------
# Only the FSR is needed here (it is the frequency ruler), so no element list:
# FSR = c / 2L with L = long arm + mid arm + short arm. The long arm is asked
# for once per file, as in the sibling scripts; the value below is the default
# offered for the first file.
LONG_ARM_LENGTH = 36e-2           # [m] lens -> far mirror (default only)
MID_ARM_LENGTH = 1.5e-2           # [m] only used by 4-element cavities
SHORT_ARM_LENGTH = 0.7e-2         # [m] near mirror -> lens

# --- analysis / plotting knobs ---------------------------------------------
TIME_COLUMN = 'Time'              # x-axis column in the PicoScope CSV
SIGNAL_COLUMN = 'Channel D'       # intensity column to analyze
TRIM_WIDTHS_BEFORE_ZEROTH = 3.0   # margin before the 0th order, in its FWHMs
N_X = 2000                        # columns of the common frequency grid
CMAP = 'viridis'
COLOR_LIMITS = (0.0, 1.0)         # (vmin, vmax); None for autoscale
OVERLAY_FIRST_ORDER = True        # mark the fitted 0th->1st spacing per row
SAVE_OUTPUTS = True               # write the figure and the map arrays to disk
REMARK_ALL = False                # ignore every cached marking
REMARK_KEYS = ()                  # ... or only these keys, e.g. (36.0,)

CACHE_SUFFIX = '.modemarks.json'
CACHE_VERSION = 1


# %% [Step 1] Loading, marking and caching one file --------------------------
def trace_csv_path(path):
    """The readable CSV for `path`; .psdata is converted (and its waveform
    buffer chosen) by the same helper the other scope scripts use."""
    path = Path(path)
    if path.suffix.lower() == '.psdata':
        return psdata_to_csv(path)
    return str(path)


def load_trace(csv_path):
    """(time, intensity) arrays from a PicoScope CSV export."""
    # Rows 1 and 2 of a PicoScope export are the unit / blank header rows.
    raw = pd.read_csv(csv_path, skiprows=[1, 2])
    raw = raw.loc[:, [TIME_COLUMN, SIGNAL_COLUMN]].dropna()
    return (raw[TIME_COLUMN].to_numpy(dtype=float),
            raw[SIGNAL_COLUMN].to_numpy(dtype=float))


def cache_file(data_path):
    """The marks sidecar for a data file: '<stem>.modemarks.json' beside it."""
    data_path = Path(data_path)
    return data_path.with_name(data_path.stem + CACHE_SUFFIX)


def load_cached_marks(data_path):
    """The cached marking of `data_path`, or None if there is no usable one.

    A cache made before the data file was last written is ignored - it belongs
    to a different measurement that happened to have the same name.
    """
    path = cache_file(data_path)
    if not path.is_file():
        return None
    try:
        record = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError) as error:
        print(f"  ignoring unreadable {path.name}: {error}")
        return None
    if record.get('version') != CACHE_VERSION:
        return None
    if record.get('source_mtime') != Path(data_path).stat().st_mtime:
        print(f"  {Path(data_path).name} changed since {path.name} was written "
              "- marking it again.")
        return None
    return record


def save_marks(data_path, record):
    path = cache_file(data_path)
    path.write_text(json.dumps(record, indent=1), encoding='utf-8')
    print(f"  marks cached in {path.name}")


def resolve_csv(record, data_path):
    """The CSV a cached marking was made on, reconverting it if it is gone.

    The psdata -> CSV cache expires (PSDATA_CSV_CACHE_MAX_AGE_DAYS), while the
    marks sidecar does not. The marks are positions on one waveform buffer, so
    a reconversion is only usable if it yields the same buffer file.
    """
    csv_path = Path(record['csv_path'])
    if csv_path.is_file():
        return str(csv_path)
    print(f"  the CSV this marking was made on is gone ({csv_path.name}) "
          "- converting again")
    new_path = Path(trace_csv_path(data_path))
    if new_path.name != csv_path.name:
        raise RuntimeError(
            f"the marks in {cache_file(data_path).name} were made on waveform "
            f"buffer '{csv_path.name}', but '{new_path.name}' was chosen - "
            "rerun and pick the same buffer, or delete the sidecar to mark the "
            "file again.")
    record['csv_path'] = str(new_path)
    save_marks(data_path, record)
    return str(new_path)


def analyse_file(key, data_path, default_long_arm, remark=False):
    """Return the marking record for one measurement, marking it if needed.

    Keys of the record: csv_path (which waveform buffer was used), long_arm_m,
    marks (as mode_marking.mark_pairs returns them), source_mtime.
    """
    data_path = Path(data_path)
    if not data_path.is_file():
        raise FileNotFoundError(f"{Y_AXIS_LABEL} = {key:g}: no such file: {data_path}")

    if not remark:
        cached = load_cached_marks(data_path)
        if cached is not None:
            print(f"  using cached marks ({len(cached['marks'])} pairs, "
                  f"long arm {cached['long_arm_m'] * 100:.4g} cm)")
            return cached

    csv_path = trace_csv_path(data_path)
    x, y = load_trace(csv_path)
    print(f"  loaded {len(x)} samples from '{SIGNAL_COLUMN}'")

    long_arm = ask_long_arm_length(default_long_arm)
    marks = mark_pairs(x, y, title=f"{Y_AXIS_LABEL} = {key:g}")
    marks = [pair for pair in marks if len(pair) == 2]
    if len(marks) < 2:
        raise ValueError(
            f"{Y_AXIS_LABEL} = {key:g}: only {len(marks)} complete pair(s) were "
            "marked; at least two consecutive ones are needed for the FSR.")

    record = {'version': CACHE_VERSION,
              'key': float(key),
              'source_name': data_path.name,
              'source_mtime': data_path.stat().st_mtime,
              'csv_path': str(csv_path),
              'long_arm_m': float(long_arm),
              'marks': marks}
    save_marks(data_path, record)
    return record


# %% [Step 2] One trace -> one row of the map --------------------------------
def build_row(x, y, marks, fsr_mhz, trim_widths=TRIM_WIDTHS_BEFORE_ZEROTH):
    """Turn a marked trace into a frequency-aligned, normalized map row.

    `marks` are mode_marking's pairs (0th order first, then the 1st order that
    follows it) over consecutive longitudinal mode numbers; `fsr_mhz` is the
    cavity's free spectral range, which is what turns the scope's time axis
    into a frequency axis.

    Returns {'f': [MHz from the 0th order], 'i': [normalized intensity],
             'df_mhz', 'fwhm_0_mhz', 'fwhm_1_mhz', 'fsr_mhz', 'fsr_s',
             'mhz_per_s', 'flipped'}.
    """
    pairs = sorted((pair for pair in marks if len(pair) == 2),
                   key=lambda pair: pair[0]['x0'])
    if len(pairs) < 2:
        raise ValueError('need at least two marked pairs to get an FSR')

    t_zero = np.array([pair[0]['x0'] for pair in pairs], dtype=float)
    fsr_s = float(np.mean(np.diff(t_zero)))  # FSR in scope-seconds
    if fsr_s == 0:
        raise ValueError('two pairs share the same 0th-order position')
    mhz_per_s = fsr_mhz / abs(fsr_s)  # the scan pace, in MHz per scope-second

    # Which way the scan ran: +1 when the 1st order sits at a later time than
    # its 0th order. Unanimous in practice, so a majority vote is only a guard
    # against a single mismarked pair.
    votes = np.sign([pair[1]['x0'] - pair[0]['x0'] for pair in pairs])
    direction = 1.0 if votes.sum() >= 0 else -1.0

    # The reference is the pair with a neighbouring 0th order on the +f side -
    # the trimmed window runs one FSR in that direction. With direction = +1
    # (frequency rising with time) that is the earliest pair, otherwise the
    # latest one.
    reference = pairs[0] if direction > 0 else pairs[-1]
    zeroth, first = reference

    gammas = [pair[0]['gamma'] for pair in pairs if pair[0]['gamma'] is not None]
    gamma = zeroth['gamma'] if zeroth['gamma'] is not None else (
        float(np.median(gammas)) if gammas else None)
    if gamma is None:
        fwhm_0_mhz = fsr_mhz / 200.0
        print("  no fitted 0th-order width (positions were clicked, not fitted)"
              f" - trimming with a placeholder FWHM of {fwhm_0_mhz:.4g} MHz")
    else:
        fwhm_0_mhz = 2.0 * abs(gamma) * mhz_per_s
    gammas_1 = [pair[1]['gamma'] for pair in pairs if pair[1]['gamma'] is not None]
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

    if zeroth['height'] and zeroth['y0'] is not None:
        i_row = (i_row - zeroth['y0']) / zeroth['height']
    else:
        span = float(np.ptp(i_row)) or 1.0
        print("  no fitted 0th-order height - normalizing this row by its own "
              "min/max instead")
        i_row = (i_row - float(np.min(i_row))) / span

    df_mhz = float(np.mean([abs(pair[1]['x0'] - pair[0]['x0'])
                            for pair in pairs])) * mhz_per_s
    return {'f': f_row, 'i': i_row,
            'df_mhz': df_mhz,
            'fwhm_0_mhz': fwhm_0_mhz,
            'fwhm_1_mhz': fwhm_1_mhz,
            'fsr_mhz': fsr_mhz,
            'fsr_s': fsr_s,
            'mhz_per_s': mhz_per_s,
            'flipped': direction < 0,
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
    fig.colorbar(mesh, ax=ax, label='Transmission (1 = 0th-order peak)')

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


def print_table(keys, rows, y_label=Y_AXIS_LABEL):
    header = (f"{y_label:>18} | {'df [MHz]':>10} | {'FWHM0 [MHz]':>11} | "
              f"{'FWHM1 [MHz]':>11} | {'FSR [MHz]':>10} | {'MHz/s':>10} | scan")
    print()
    print(header)
    print('-' * len(header))
    for key, row in zip(keys, rows):
        print(f"{key:>18g} | {row['df_mhz']:>10.4f} | {row['fwhm_0_mhz']:>11.4f} | "
              f"{row['fwhm_1_mhz']:>11.4f} | {row['fsr_mhz']:>10.2f} | "
              f"{row['mhz_per_s']:>10.4g} | "
              f"{'down (flipped)' if row['flipped'] else 'up'}")
    print()


def save_outputs(folder, fig, keys, rows, centres, z):
    """Write the figure and the map arrays next to the measurements."""
    folder = Path(folder)
    stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    png_path = folder / f"mode_map_2d_{stamp}.png"
    npz_path = folder / f"mode_map_2d_{stamp}.npz"
    fig.savefig(png_path, dpi=200)
    np.savez(npz_path, keys=keys, f_mhz=centres, intensity=z.filled(np.nan),
             df_mhz=[row['df_mhz'] for row in rows],
             fwhm_0_mhz=[row['fwhm_0_mhz'] for row in rows],
             fwhm_1_mhz=[row['fwhm_1_mhz'] for row in rows],
             fsr_mhz=[row['fsr_mhz'] for row in rows],
             y_axis_label=Y_AXIS_LABEL)
    print(f"Saved {png_path}")
    print(f"Saved {npz_path}")


# %% [Step 5] The run --------------------------------------------------------
def main(measurements=None, y_label=None):
    """Mark every measurement (or reuse its cache) and draw the map."""
    measurements = MEASUREMENTS if measurements is None else measurements
    y_label = Y_AXIS_LABEL if y_label is None else y_label
    if not measurements:
        raise ValueError(
            'MEASUREMENTS is empty - fill in {parameter value: file path} at '
            'the top of pico_scope/mode_map_2d.py.')

    keys, rows = [], []
    default_long_arm = LONG_ARM_LENGTH
    for index, key in enumerate(sorted(measurements), start=1):
        path = measurements[key]
        print(f"\n=== [{index}/{len(measurements)}] {y_label} = {key:g}: "
              f"{Path(path).name} ===")
        record = analyse_file(
            key, path, default_long_arm,
            remark=REMARK_ALL or key in REMARK_KEYS)
        default_long_arm = record['long_arm_m']

        x, y = load_trace(resolve_csv(record, path))
        fsr_mhz = cavity_fsr_mhz(long_arm=record['long_arm_m'],
                                 mid_arm=MID_ARM_LENGTH,
                                 short_arm=SHORT_ARM_LENGTH)
        row = build_row(x, y, record['marks'], fsr_mhz)
        print(f"  {row['n_pairs']} pairs, FSR = {fsr_mhz:.2f} MHz "
              f"({row['fsr_s']:.4g} s), df = {row['df_mhz']:.4f} MHz, "
              f"0th-order FWHM = {row['fwhm_0_mhz']:.4f} MHz"
              f"{', scan flipped' if row['flipped'] else ''}")
        keys.append(float(key))
        rows.append(row)

    keys, rows, centres, x_edges, y_edges, z = build_map(keys, rows)
    print_table(keys, rows, y_label=y_label)
    fig = plot_map(keys, rows, x_edges, y_edges, z, y_label=y_label)
    if SAVE_OUTPUTS:
        save_outputs(Path(next(iter(measurements.values()))).parent, fig,
                     keys, rows, centres, z)
    plt.show(block=True)
    return keys, rows, centres, z


# %% [Step 6] Self-test (synthetic data; no scope, no GUI, no files) ---------
def _synthetic_trace(fsr_mhz, df_mhz, fwhm_mhz, mhz_per_s, descending,
                     amplitude, offset, n_fsr=3, n_samples=40000):
    """A fake scan and its exact marks, as the marker would have fitted them.

    Returns (x, y, marks). The scan covers `n_fsr` free spectral ranges, runs
    up or down in frequency, and carries a 0th/1st-order Lorentzian pair per
    FSR - so the row builder gets everything it needs and the right answers are
    known exactly.
    """
    from pico_scope.mode_analysis import area_lorentzian
    from pico_scope.mode_marking import peak_record

    f_lin = np.linspace(-0.3 * fsr_mhz, (n_fsr - 0.5) * fsr_mhz, n_samples)
    x = np.linspace(0.0, (f_lin[-1] - f_lin[0]) / mhz_per_s, n_samples)
    f_of_t = f_lin[::-1] if descending else f_lin

    gamma_mhz = fwhm_mhz / 2.0
    gamma_s = gamma_mhz / mhz_per_s
    centres_mhz = [k * fsr_mhz + shift
                   for k in range(n_fsr) for shift in (0.0, df_mhz)]
    y = np.full(n_samples, float(offset))
    for centre in centres_mhz:
        y += area_lorentzian(f_of_t, centre, gamma_mhz,
                             amplitude * np.pi * gamma_mhz, 0.0)

    # frequency -> time, on whichever monotonic direction this scan ran
    f_sorted, t_sorted = (f_of_t[::-1], x[::-1]) if descending else (f_of_t, x)

    def time_of(centre_mhz):
        return float(np.interp(centre_mhz, f_sorted, t_sorted))

    marks = []
    for k in range(n_fsr):
        pair = [peak_record(time_of(k * fsr_mhz), gamma_s,
                            amplitude * np.pi * gamma_s, offset),
                peak_record(time_of(k * fsr_mhz + df_mhz), gamma_s,
                            amplitude * np.pi * gamma_s, offset)]
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
    print('mode_map_2d self-test passed')


if __name__ == '__main__':
    if '--self-test' in sys.argv:
        _self_test()
    else:
        main()
