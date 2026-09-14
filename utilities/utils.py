# general functions for the project
import hashlib
import shutil
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import pyperclip
from scipy.optimize import curve_fit
from send2trash import send2trash
from typing import Optional, Union, Sequence
from local_config import PATH_OBSIDIAN_ATTACHMENTS_FOLDER

def wait_for_path_from_clipboard(filetype: Optional[Union[str, Sequence[str]]] = None, poll_interval=0.5, verbose=True,
                                 instructions_message="Waiting for a file path to be copied to clipboard..."):
    if instructions_message is not None:
        print("\n" + instructions_message + '\n')

    I = 0
    while True:
        clipboard = pyperclip.paste().strip().strip('"')  # Strip whitespace and quotes

        if os.path.isfile(clipboard) or os.path.isdir(clipboard):
            # `filetype` may be a single extension/keyword or a sequence of
            # acceptable extensions (e.g. ('csv', 'psdata')). The keyword
            # branches below only apply to the single-string form.
            filetype_is_str = isinstance(filetype, str)
            filetype_lower = filetype.lower() if filetype_is_str else None

            if filetype_lower in ['video', 'media']:
                # Try to open as a video
                cap = cv2.VideoCapture(clipboard)
                if cap.isOpened():
                    cap.release()
                    if verbose:
                        print(f"OK: Detected valid video path: {clipboard}")
                    return clipboard
                cap.release()

            if filetype_lower in ['image', 'media']:
                # Try to read as an image
                img = cv2.imread(clipboard)
                if img is not None:
                    if verbose:
                        print(f"OK: Detected valid image path: {clipboard}")
                    return clipboard

            if filetype_lower == 'excel':
                if clipboard.endswith('.xlsx') or clipboard.endswith('.xls'):
                    if verbose:
                        print(f"OK: Detected valid CSV path: {clipboard}")
                    return clipboard

            if filetype_lower in ['table', 'tabular']:
                if clipboard.endswith('.csv') or clipboard.endswith('.xlsx') or clipboard.endswith('.xls'):
                    if verbose:
                        print(f"OK: Detected valid table path: {clipboard}")
                    return clipboard

            if filetype_lower in ['folder', 'directory', 'dir']:
                if os.path.isdir(clipboard):
                    if verbose:
                        print(f"OK: Detected valid directory path: {clipboard}")
                    return clipboard

            if filetype is not None:
                # In sequence form, entries can mix directory keywords with
                # plain extensions (e.g. ('avi', 'directory')).
                tokens = [filetype_lower] if filetype_is_str else [ft.lower() for ft in filetype]
                extensions = [t for t in tokens if t not in ('folder', 'directory', 'dir')]
                wants_directory = len(extensions) != len(tokens)

                if wants_directory and os.path.isdir(clipboard):
                    if verbose:
                        print(f"OK: Detected valid directory path: {clipboard}")
                    return clipboard

                if any(clipboard.lower().endswith(f'.{ext}') for ext in extensions):
                    if verbose:
                        print(f"OK: Detected valid path: {clipboard}")
                    return clipboard

            if filetype is None:
                # No specific filetype validation
                if verbose:
                    print(f"OK: Detected path: {clipboard}")
                return clipboard

        if verbose:
            number_of_dots = I % 3 + 1
            dots = '.' * number_of_dots
            print(f"Waiting for path to be copied{dots}", end="\r")
            I += 1
        time.sleep(poll_interval)

# BatchConvert first shipped in PicoScope 7.1.32. Older versions do not know
# the argument: they start up normally instead, and on a computer with a scope
# attached they silently connect to it and begin measuring - so the version
# must be checked *before* launching the exe.
PICOSCOPE_MIN_BATCHCONVERT_VERSION = (7, 1, 32)
PSDATA_CONVERT_TIMEOUT_S = 120

# Converted CSVs are kept in a per-source-file folder under the system temp
# directory so that re-analysing the same .psdata (a common thing to do - the
# buffer choice or the fit is often redone) reuses the earlier export instead
# of launching PicoScope again. Entries are pruned by age; the cache lives in
# temp precisely because losing it is harmless.
PSDATA_CSV_CACHE_DIR = Path(tempfile.gettempdir()) / 'psdata_to_csv_cache'
PSDATA_CSV_CACHE_MAX_AGE_DAYS = 14


def _windows_exe_version(path):
    """Return the file version of a Windows exe as a tuple of 4 ints, or None."""
    import ctypes

    path = str(path)
    size = ctypes.windll.version.GetFileVersionInfoSizeW(path, None)
    if not size:
        return None
    data = ctypes.create_string_buffer(size)
    if not ctypes.windll.version.GetFileVersionInfoW(path, 0, size, data):
        return None
    buf = ctypes.c_void_p()
    length = ctypes.c_uint()
    if not ctypes.windll.version.VerQueryValueW(
            data, '\\', ctypes.byref(buf), ctypes.byref(length)):
        return None
    # VS_FIXEDFILEINFO: dwFileVersionMS / dwFileVersionLS are the 3rd and 4th
    # DWORDs of the structure.
    fixed = ctypes.cast(buf, ctypes.POINTER(ctypes.c_uint32 * 4)).contents
    ms, ls = fixed[2], fixed[3]
    return (ms >> 16, ms & 0xFFFF, ls >> 16, ls & 0xFFFF)


def _psdata_cache_dir(psdata_path):
    """Return the cache folder for one .psdata file.

    The key is a hash of the full path (so that same-named files in different
    folders - 'trace.psdata' in every measurement folder - never collide) plus
    size and mtime (so that an overwritten or re-saved file simply misses the
    cache instead of silently handing back the previous export). The stem is
    kept as a readable prefix, since this folder is meant to be browsable.

    The waveform buffer index is deliberately *not* part of the key: one
    conversion produces every buffer at once, so a cache entry holds them all
    and the choice is made afterwards from the CSVs inside it.
    """
    stat = psdata_path.stat()
    path_hash = hashlib.sha1(
        str(psdata_path.resolve()).lower().encode('utf-8')).hexdigest()[:10]
    return (PSDATA_CSV_CACHE_DIR /
            f"{psdata_path.stem}_{path_hash}_{stat.st_size}_{int(stat.st_mtime)}")


def _sorted_buffer_csvs(out_dir):
    """List the CSVs in a BatchConvert output folder, in buffer order.

    A single-waveform file becomes '<stem>.csv' directly in out_dir; a file
    with multiple waveform buffers becomes a '<stem>' subfolder holding
    '<stem>_1.csv' ... '<stem>_N.csv', so search recursively and sort by the
    numeric buffer suffix (plain name-sorting would put _10 before _2).
    """
    def buffer_index(p):
        suffix = p.stem.rsplit('_', 1)[-1]
        return int(suffix) if suffix.isdigit() else 0

    if not out_dir.is_dir():
        return []
    return sorted(out_dir.rglob('*.csv'), key=buffer_index)


def choose_buffer_csv(csv_files, allow_skip=False):
    """Return the CSV to analyse, asking the user when the file has several.

    `allow_skip` is for a caller that is asking a second time - because the
    buffer it got the first time turned out to be the wrong one - and can go on
    without this file at all: it adds -1 (skip, returns None) to the choices,
    and it makes the question be asked even when there is only one buffer,
    where it means 'that one again' versus 'give up on this file'.
    """
    if len(csv_files) == 1 and not allow_skip:
        return str(csv_files[0])

    plural = '' if len(csv_files) == 1 else 's'
    print(f"The psdata file contains {len(csv_files)} waveform buffer{plural}:")
    for i, p in enumerate(csv_files, start=1):
        print(f"  [{i}] {p.name}")
    skip_hint = ', -1 to skip this file' if allow_skip else ''
    while True:
        raw_in = input(
            f"Which waveform to use? 1-{len(csv_files)}{skip_hint} "
            f"[default {len(csv_files)} - the most recent]: "
        ).strip()
        if raw_in == '':
            choice = len(csv_files)
        else:
            try:
                choice = int(raw_in)
            except ValueError:
                choice = 0
        if allow_skip and choice == -1:
            print("Skipping this file.")
            return None
        if 1 <= choice <= len(csv_files):
            break
        print(f"  Please enter a number between 1 and {len(csv_files)}"
              f"{skip_hint}.")
    print(f"Using waveform: {csv_files[choice - 1].name}")
    return str(csv_files[choice - 1])


def _prune_psdata_csv_cache(max_age_days=PSDATA_CSV_CACHE_MAX_AGE_DAYS):
    """Delete conversion-cache entries older than `max_age_days`.

    Age comes from the entry's own mtime, i.e. when it was converted; reuse
    does not refresh it, so every entry expires a fixed time after creation
    however often it is read.
    """
    if not PSDATA_CSV_CACHE_DIR.is_dir():
        return
    cutoff = time.time() - max_age_days * 24 * 3600
    for entry in PSDATA_CSV_CACHE_DIR.iterdir():
        try:
            if entry.is_dir() and entry.stat().st_mtime < cutoff:
                shutil.rmtree(entry, ignore_errors=True)
        except OSError:
            pass  # a cache we cannot tidy is no reason to fail the analysis


def psdata_buffer_csvs(psdata_path):
    """Convert a PicoScope .psdata file to CSV; return one CSV per buffer.

    Uses PicoScope 7's command-line `BatchConvert` mode, which produces a CSV
    identical to the GUI's "Save as CSV". BatchConvert operates on folders,
    so the single file is copied to a temporary folder and converted there.
    Requires PICOSCOPE_EXE (the path of PicoScope.exe) in local_config.py,
    and PicoScope >= 7.1.32 (checked before launching).

    If an up-to-date CSV with the same name already sits next to the .psdata
    file (e.g. from an earlier manual export or an earlier run), it is used
    directly and no conversion is performed. Failing that, an earlier
    conversion of the same file cached under PSDATA_CSV_CACHE_DIR is reused -
    which also means a cached file can be re-analysed without PicoScope
    installed.

    The list is in buffer order (the most recent capture last) and holds a
    single entry when the file has one waveform buffer. psdata_to_csv() picks
    one of them; a caller that may have to come back and pick a different one
    (pico_scope/mode_map_2d.py) keeps the list and re-runs choose_buffer_csv()
    on it, which costs no second conversion.
    """
    psdata_path = Path(psdata_path)
    sibling_csv = psdata_path.with_suffix('.csv')
    if sibling_csv.is_file() and sibling_csv.stat().st_mtime >= psdata_path.stat().st_mtime:
        print(f"Using existing up-to-date CSV: {sibling_csv}")
        return [sibling_csv]

    _prune_psdata_csv_cache()
    cache_dir = _psdata_cache_dir(psdata_path)
    out_dir = cache_dir / 'out'
    cached_csvs = _sorted_buffer_csvs(out_dir)
    if cached_csvs:
        print(f"Using cached conversion of '{psdata_path.name}' "
              f"({len(cached_csvs)} waveform buffer(s)) from {out_dir}")
        return cached_csvs

    try:
        from local_config import PICOSCOPE_EXE
    except ImportError:
        raise FileNotFoundError(
            "PICOSCOPE_EXE is not defined in local_config.py - add it there "
            "(see local_config_template.py), or save the trace as CSV manually."
        )
    if not Path(PICOSCOPE_EXE).is_file():
        raise FileNotFoundError(
            f"PicoScope executable not found at {PICOSCOPE_EXE!r} - "
            "update PICOSCOPE_EXE in local_config.py, or save the trace as CSV manually."
        )

    version = _windows_exe_version(PICOSCOPE_EXE)
    if version is not None and version[:3] < PICOSCOPE_MIN_BATCHCONVERT_VERSION:
        min_ver = '.'.join(map(str, PICOSCOPE_MIN_BATCHCONVERT_VERSION))
        raise RuntimeError(
            f"PicoScope 7 at {PICOSCOPE_EXE!r} is version "
            f"{'.'.join(map(str, version[:3]))}, but command-line conversion "
            f"(BatchConvert) was only added in {min_ver}. Older versions ignore "
            "the arguments and just open the PicoScope GUI (grabbing any "
            "attached scope). Update PicoScope 7 from "
            "https://www.picotech.com/downloads, or save the trace as CSV "
            "manually."
        )

    in_dir = cache_dir / 'in'
    in_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(psdata_path, in_dir)

    print(f"Converting '{psdata_path.name}' to CSV with PicoScope 7 ...")
    # Note: BatchConvert fails on folder paths with a trailing backslash;
    # str(Path) never produces one.
    try:
        result = subprocess.run(
            [PICOSCOPE_EXE, 'BatchConvert', str(in_dir), str(out_dir), '.csv'],
            capture_output=True, text=True, timeout=PSDATA_CONVERT_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        # subprocess.run kills the child before raising, so the stray
        # PicoScope window is closed rather than left grabbing the scope.
        # Drop the half-written cache entry so the next run retries cleanly.
        shutil.rmtree(cache_dir, ignore_errors=True)
        raise RuntimeError(
            f"PicoScope did not finish converting within "
            f"{PSDATA_CONVERT_TIMEOUT_S} s and was closed. This usually means "
            "it opened as a normal GUI session instead of converting (e.g. it "
            "connected to an attached scope, or popped a device-selection "
            "dialog). Close any open PicoScope window and retry, or save the "
            "trace as CSV manually."
        )

    csv_files = _sorted_buffer_csvs(out_dir)
    if result.returncode != 0 or not csv_files:
        # A partial output must not be left behind: the next run would take it
        # for a good cache entry and never retry the conversion.
        shutil.rmtree(cache_dir, ignore_errors=True)
        raise RuntimeError(
            f"psdata -> CSV conversion failed (exit code {result.returncode}).\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
    print("Conversion succeeded.")
    # The copy BatchConvert worked from has done its job; only the CSVs are
    # worth keeping in the cache.
    shutil.rmtree(in_dir, ignore_errors=True)

    return csv_files


def psdata_to_csv(psdata_path):
    """Convert a PicoScope .psdata file to CSV; return the CSV path.

    psdata_buffer_csvs() does the conversion; when the file holds several
    waveform buffers the user is asked which one to use (default: the last,
    i.e. most recent, capture).
    """
    return choose_buffer_csv(psdata_buffer_csvs(psdata_path))


def get_picoscope_trace_path_from_clipboard():
    """Wait for a .csv/.psdata path on the clipboard; return a readable CSV path.

    .psdata files are converted via psdata_to_csv(); .csv files are returned
    as-is. Returns (csv_path, original_path) - `original_path` is the file the
    user actually copied, which is where analysis records should be logged.
    """
    input_path = wait_for_path_from_clipboard(filetype=('csv', 'psdata'))
    if input_path.lower().endswith('.psdata'):
        return psdata_to_csv(input_path), input_path
    return input_path, input_path


def ask_long_arm_length(verbose=True):
    """Ask for the cavity's long arm length in cm; return it in METRES.

    It is the one geometry number that changes between measurements, and a
    stale value silently biases the NA (the measured mode spacing itself is
    unaffected), so the analysis scripts prompt for it on every run instead of
    reading a config constant. For the same reason there is no default: the
    length is always typed, so a reflex Enter cannot slip the previous
    measurement's geometry into the results. Values outside 1 - 1000 cm are
    rejected: they are almost always a length typed in metres.
    """
    while True:
        raw_in = input("Long arm length in CENTIMETRES: ").strip()
        if raw_in == '':
            print("  The long arm length is required - type it in centimetres.")
            continue
        try:
            value_cm = float(raw_in)
        except ValueError:
            print(f"  Could not parse '{raw_in}' as a number.")
            continue
        if not 1.0 <= value_cm <= 1000.0:
            print(f"  {value_cm:g} cm is outside the plausible range "
                  f"1 - 1000 cm - the value is in centimetres, not metres.")
            continue
        break
    if verbose:
        print(f"Long arm length: {value_cm:.4g} cm ({value_cm / 100:.4g} m)")
    return value_cm / 100


def convert_path_to_obsidian_embedding_converter(verbose=True):
    """Rewrite a Windows path on the clipboard as an Obsidian embed link.

    Drops everything before "Labs Dropbox" and renames it to "Dropbox Files"
    (e.g. "C:\\Users\\me\\...\\OS Labs Dropbox\\a\\b.png" ->
    "Dropbox Files\\a\\b.png"), then converts all "\\" to "/". If
    "Labs Dropbox" is not found, that first step is skipped and only the
    slash conversion runs. Either way, the result is then wrapped as an
    Obsidian image embed at 500px: "![[<path>|500]]". Writes the result back
    to the clipboard and returns it.
    """
    text = pyperclip.paste().strip()
    # "Copy as path" in Windows Explorer wraps the path in double quotes
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1]
    marker = "Labs Dropbox"
    index = text.find(marker)
    if index != -1:
        text = "Dropbox Files" + text[index + len(marker):]
    text = text.replace("\\", "/")
    text = "![[" + text + "|500]]"
    pyperclip.copy(text)
    if verbose:
        print(text)
    return text


def append_numerical_result_line(data_file_path, results_text,
                                 results_filename='numerical-results.txt'):
    """Append a one-line analysis record next to the data file it came from.

    Writes (creates if needed) `numerical-results.txt` in the folder of
    `data_file_path` and appends a timestamped line of the form:

        2026-07-12 19:30:00 | <data file name> | <results_text>

    Repeated analyses of the same file simply add more rows - the point is a
    lightweight, always-there log of past fits, not a formal results store.
    Returns the path of the results file.
    """
    folder = os.path.dirname(os.path.abspath(data_file_path))
    results_path = os.path.join(folder, results_filename)
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line = f"{timestamp} | {os.path.basename(data_file_path)} | {results_text}\n"
    with open(results_path, 'a', encoding='utf-8') as f:
        f.write(line)
    print(f"Result recorded in: {results_path}")
    return results_path


def get_obsidian_save_path(filename: Optional[str] = None, overwrite: bool = False) -> str:
    attachment_path = Path(PATH_OBSIDIAN_ATTACHMENTS_FOLDER)

    if filename is not None:
        attachments_path = attachment_path / filename
        if attachments_path.exists() and not overwrite:
            raise FileExistsError(f"{attachments_path} already exists")

    return str(attachment_path)

def delete_redundant_avi_files(directory):
    """
    Deletes .avi files in the specified directory if there's a .mp4 file
    with the same name (excluding extension) and the .mp4 file is larger than 10 KB.
    """
    for filename in os.listdir(directory):
        if filename.endswith('.avi'):
            avi_path = os.path.join(directory, filename)
            base_name = os.path.splitext(filename)[0]
            mp4_path = os.path.join(directory, base_name + '.mp4')

            if os.path.isfile(mp4_path) and os.path.getsize(mp4_path) > 10 * 1024:
                print(f"Deleting: {avi_path}")
                send2trash(avi_path)

def save_fig_safe(filepath, **kwargs):
    """
    Saves a matplotlib figure to `filepath`. If the file already exists,
    saves to `filepath` with a numeric suffix before the extension, like _1, _2, etc.

    Example:
        save_fig_safe("output/plot.png")
        → saves to "output/plot.png" or "output/plot_1.png" if already exists

    kwargs are passed to plt.savefig (e.g., dpi=300, bbox_inches='tight')
    """
    base, ext = os.path.splitext(filepath)
    candidate = filepath
    i = 1

    while os.path.exists(candidate):
        candidate = f"{base}_{i}{ext}"
        i += 1

    plt.savefig(candidate, **kwargs)
    print(f"OK: Saved figure to: {candidate}")


# The two functions below are duplicated verbatim in the cavity-design project, in
# cavity_design/_utils.py - the two projects are independent, so keep the copies in sync by hand.
def copy_figure_as_png_to_clipboard(fig=None, dpi=200):
    """Copy a matplotlib figure to the system clipboard as a PNG.

    `fig` defaults to the current figure. The figure is re-rendered at `dpi` rather than grabbed
    off the screen, so the copy does not depend on how large the window happens to be. Both
    clipboard flavours are set: the real PNG bytes ('image/png') and a bitmap, which is what
    Word / Obsidian actually reach for when pasting.

    Needs a Qt backend (matplotlib.use('Qt5Agg')) - there is no Qt clipboard without a Qt app.
    """
    import io
    import matplotlib.pyplot as plt
    from matplotlib.backends.qt_compat import QtCore, QtGui, QtWidgets

    if fig is None:
        fig = plt.gcf()
    application = QtWidgets.QApplication.instance()
    if application is None:
        raise RuntimeError(
            "copying a figure to the clipboard needs a running Qt application - select the Qt "
            "backend with matplotlib.use('Qt5Agg') before creating the figure")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    png_bytes = buf.getvalue()

    mime = QtCore.QMimeData()
    mime.setData("image/png", QtCore.QByteArray(png_bytes))
    mime.setImageData(QtGui.QImage.fromData(png_bytes, "PNG"))
    application.clipboard().setMimeData(mime)
    return fig


def enable_copy_to_clipboard(fig=None, dpi=200):
    """Bind Ctrl+C on `fig` (default: the current figure) to copy_figure_as_png_to_clipboard().

    The binding follows rcParams['keymap.copy'] - the same setting matplotlib's own copy tool
    uses, which the classic toolbar never consults, so it clashes with nothing. Returns the
    connection id, should you want fig.canvas.mpl_disconnect(cid).

    The plot window needs the keyboard focus; with the focus on the console, Ctrl+C interrupts
    the script as usual.
    """
    import matplotlib.pyplot as plt

    if fig is None:
        fig = plt.gcf()

    def on_key(event):
        if event.key in plt.rcParams["keymap.copy"]:
            copy_figure_as_png_to_clipboard(fig, dpi=dpi)
            print(f"Copied figure {fig.get_label() or fig.number} to the clipboard.")

    return fig.canvas.mpl_connect("key_press_event", on_key)


# --------------------------------------------------- 2D Gaussian beam fitting
# A beam's own fit, independent of any camera package: basler_cam/gaussian_fit.py
# and basler_cam/mode_position_capture_gui.py each carry a copy of an older
# routine that this one replaces for the offline analysis scripts. Three things
# are different, and each of them was a way of reading the old result wrong:
#
#   1. The widths come back named after the beam's own axes - 'minor' and
#      'major' - not after the image's. The old routine fitted sigma_x and
#      sigma_y along a rotated frame but kept the image's names for them, and
#      since its angle was bounded to +/- 45 deg, 'w_x' silently meant "the
#      principal width whose axis is nearest the image horizontal". Either of
#      the two could be the larger, so neither name said what it measured.
#   2. The angle is unbounded, and the initial guess comes from the image's own
#      second moments rather than from zero. The old +/- 45 deg bound left a
#      seam there: starting round and unturned, the optimizer had no gradient
#      to tell it which way to go, and a beam tilted within a degree or two of
#      45 deg could settle for a circle instead - a true 90 x 24 px spot at
#      44 deg came back 46 x 46. It took a clean frame to show (sensor noise
#      happens to break the symmetry and rescue the fit), which is exactly what
#      makes it the kind of thing to design out rather than watch for.
#   3. The rotation is written as an explicit change of axes, so the angle is
#      the direction of the major axis, measured from +x toward +y - the same
#      convention matplotlib's Ellipse takes. The old a/b/c form put the
#      sigma_x axis at MINUS theta, and the two overlays in this repo that drew
#      it disagreed about the sign.
#
# Distances are in pixels throughout; the caller multiplies by its own pixel
# size. As everywhere here, w is the 1/e^2 intensity radius, twice sigma.

def gaussian_2d(xy, amplitude, x_0, y_0, sigma_u, sigma_v, angle, offset):
    """A rotated 2D Gaussian, evaluated on the grid `xy` = (x, y) and raveled.

    `angle` [rad] is the direction of the sigma_u axis, measured from +x toward
    +y - that is, toward increasing ROW index on an image shown with
    imshow(origin='upper'). Rotating the coordinates explicitly, rather than
    folding the rotation into the usual a/b/c quadratic coefficients, is what
    keeps that sign readable: the very same angle, in degrees, is what
    matplotlib's Ellipse wants for an overlay that lies along the data.
    """
    x, y = xy[0], xy[1]
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    dx, dy = x - x_0, y - y_0
    u = dx * cos_a + dy * sin_a     # along the sigma_u axis
    v = -dx * sin_a + dy * cos_a    # across it
    return np.ravel(offset + amplitude
                    * np.exp(-0.5 * ((u / sigma_u) ** 2 + (v / sigma_v) ** 2)))


def rebin_image(image, factor):
    """Average `factor` x `factor` blocks of pixels (the edge remainder is dropped)."""
    if factor <= 1:
        return np.asarray(image, dtype=float)
    height, width = np.shape(image)
    h_crop, w_crop = (height // factor) * factor, (width // factor) * factor
    cropped = np.asarray(image, dtype=float)[:h_crop, :w_crop]
    return cropped.reshape(h_crop // factor, factor,
                           w_crop // factor, factor).mean(axis=(1, 3))


# The brightest pixels the moment guess is taken over, as a fraction of the
# peak above background. exp(-2) is the 1/e^2 level - the beam's own edge - so
# the guess is computed over the spot and not over the sensor around it.
_MOMENT_GUESS_LEVEL = np.exp(-2.0)
# Moments of a Gaussian truncated at that level are narrower than the Gaussian
# itself; this puts the sigma guess back on scale. It only has to be roughly
# right - the fit refines it - but starting on scale is what keeps a strongly
# elliptical beam from converging onto a round local minimum.
_MOMENT_GUESS_WIDENING = 1.52


def _moment_guess(image):
    """(x_0, y_0, sigma_major, sigma_minor, angle) from the image's own moments.

    The starting point for the fit. Taking the orientation from the data - and
    not from zero, as the older routine did - is what lets a steeply tilted
    beam be fitted at all: from a round, unrotated start the optimizer has no
    gradient telling it which way to turn, and settles for a circle.
    """
    image = np.asarray(image, dtype=float)
    background = np.percentile(image, 15)
    weights = np.clip(image - background, 0.0, None)
    peak = weights.max()
    if peak <= 0:  # a blank frame has no beam and no orientation
        height, width = image.shape
        return width / 2.0, height / 2.0, max(width, height) / 4.0, \
            max(width, height) / 4.0, 0.0
    weights = np.where(weights >= _MOMENT_GUESS_LEVEL * peak, weights, 0.0)

    yy, xx = np.mgrid[:image.shape[0], :image.shape[1]]
    total = weights.sum()
    x_0 = float((weights * xx).sum() / total)
    y_0 = float((weights * yy).sum() / total)
    dx, dy = xx - x_0, yy - y_0
    c_xx = float((weights * dx * dx).sum() / total)
    c_yy = float((weights * dy * dy).sum() / total)
    c_xy = float((weights * dx * dy).sum() / total)

    # eigh returns ascending eigenvalues, so the second eigenvector is the major axis
    values, vectors = np.linalg.eigh(np.array([[c_xx, c_xy], [c_xy, c_yy]]))
    values = np.clip(values, 1e-6, None)
    sigma_minor, sigma_major = (np.sqrt(values) * _MOMENT_GUESS_WIDENING)
    major = vectors[:, 1]
    angle = float(np.arctan2(major[1], major[0]))
    return x_0, y_0, float(sigma_major), float(sigma_minor), angle


def _canonical_axes(sigma_u, sigma_v, angle):
    """Sort a fitted (sigma_u, sigma_v, angle) into (major, minor, major angle).

    The fit is free to describe one ellipse two ways - swapping the two sigmas
    and turning by 90 deg - so the answer is only unambiguous once the larger
    axis has been named the major one and the reported angle made to be its
    direction. The result is wrapped into [-90, +90) deg, the half turn an
    ellipse's orientation actually lives in.
    """
    if sigma_v > sigma_u:
        sigma_u, sigma_v = sigma_v, sigma_u
        angle += np.pi / 2
    angle_deg = (np.degrees(angle) + 90.0) % 180.0 - 90.0
    return float(sigma_u), float(sigma_v), float(angle_deg)


def fit_gaussian_beam(image, rebinning=1, manual_guess=None):
    """Fit a rotated 2D Gaussian to `image`; return (success, parameters).

    The beam is reported by its own principal axes, in full-resolution pixels:

        x_0, y_0            centre
        sigma_major/_minor  the two principal standard deviations
        w_major, w_minor    the 1/e^2 intensity radii, twice those sigmas
        major_axis_deg      direction of the MAJOR axis, from +x toward +y
                            (increasing row index); in [-90, +90)
        minor_axis_deg      the same for the minor axis, 90 deg away
        ellipticity         w_minor / w_major, 1.0 for a round spot
        amplitude, offset   peak above background, and the background
        angle_rad           major_axis_deg in radians, for the model function
        time                seconds the fit took

    `major_axis_deg` is ready to hand to matplotlib's Ellipse alongside
    width=2*w_major, height=2*w_minor - no sign to flip.

    With `rebinning` > 1 the fit runs on blocks of that many pixels averaged
    together (much faster, and harmless for a beam far wider than a block);
    every reported length is converted back to full-resolution pixels, the
    centre included.

    `manual_guess` is an optional {'x_0', 'y_0', 'sigma'} in full-resolution
    pixels - a circle the user dragged - which replaces the centre and width
    of the automatic guess. The orientation still comes from the image.

    `success` is False when the optimizer did not converge; `parameters` then
    holds the initial guess, so a caller can still draw something.
    """
    image = np.asarray(image, dtype=float)
    binned = rebin_image(image, rebinning)
    height, width = binned.shape
    yy, xx = np.mgrid[:height, :width]
    xx, yy = xx.astype(float), yy.astype(float)

    # A rebinned pixel at index i covers full-resolution pixels
    # [i * rebinning, (i + 1) * rebinning), so its centre sits at
    # i * rebinning + (rebinning - 1) / 2.
    bin_offset = (rebinning - 1) / 2.0

    x_0, y_0, sigma_major, sigma_minor, angle = _moment_guess(binned)
    if manual_guess is not None:
        x_0 = (manual_guess['x_0'] - bin_offset) / rebinning
        y_0 = (manual_guess['y_0'] - bin_offset) / rebinning
        # The drag gives one radius - the beam's width across its widest way -
        # so it replaces the scale, while the shape and the orientation stay
        # with the moments. Starting from a perfect circle instead, as an
        # earlier version of this did, leaves the angle with no gradient to
        # follow, and an elliptical beam then sometimes converges onto a
        # wrongly turned minimum.
        aspect = sigma_minor / sigma_major if sigma_major else 1.0
        sigma_major = max(manual_guess['sigma'] / rebinning, 1.0)
        sigma_minor = max(sigma_major * aspect, 1e-2)

    offset = float(np.percentile(binned, 15))
    amplitude = max(float(binned.max()) - offset, 1e-9)
    initial = (amplitude, x_0, y_0, sigma_major, sigma_minor, angle, offset)

    span = float(binned.max() - binned.min()) or 1.0
    # Bounds taken from the data, never from a fixed full-scale value: an
    # amplitude capped at some bit depth is a cap the caller then has to scale
    # its frames around. The angle spans a full turn - twice the half turn an
    # ellipse needs - so no orientation sits against a wall.
    lower = (0.0, -width, -height, 1e-3, 1e-3, -np.pi, binned.min() - span)
    upper = (10 * span, 2 * width, 2 * height, 10 * width, 10 * height,
             np.pi, binned.max() + span)

    started = time.time()
    success = True
    try:
        fitted, _ = curve_fit(gaussian_2d, np.array((xx, yy)), binned.ravel(),
                              p0=initial, bounds=(lower, upper),
                              ftol=1e-3, xtol=1e-3, maxfev=20000)
    except (RuntimeError, ValueError):
        success = False
        fitted = initial
    elapsed = time.time() - started

    amplitude, x_0, y_0, sigma_u, sigma_v, angle, offset = fitted
    sigma_major, sigma_minor, major_axis_deg = _canonical_axes(sigma_u, sigma_v, angle)
    sigma_major *= rebinning
    sigma_minor *= rebinning
    return success, {
        'amplitude': float(amplitude), 'offset': float(offset),
        'x_0': float(x_0) * rebinning + bin_offset,
        'y_0': float(y_0) * rebinning + bin_offset,
        'sigma_major': sigma_major, 'sigma_minor': sigma_minor,
        'w_major': 2 * sigma_major, 'w_minor': 2 * sigma_minor,
        'major_axis_deg': major_axis_deg,
        'minor_axis_deg': (major_axis_deg + 90.0 + 90.0) % 180.0 - 90.0,
        'angle_rad': np.radians(major_axis_deg),
        'ellipticity': sigma_minor / sigma_major if sigma_major else float('nan'),
        'time': elapsed,
    }


def gaussian_beam_image(parameters, shape):
    """Evaluate a fit_gaussian_beam() result over a full-resolution `shape` grid.

    For overlaying the fitted model on the frame it came from - a cross-section
    through it, or a residual. `parameters` is the dict the fit returns, whose
    lengths are already in full-resolution pixels.
    """
    height, width = shape
    yy, xx = np.mgrid[:height, :width]
    model = gaussian_2d(np.array((xx.astype(float), yy.astype(float))),
                        parameters['amplitude'], parameters['x_0'],
                        parameters['y_0'], parameters['sigma_major'],
                        parameters['sigma_minor'], parameters['angle_rad'],
                        parameters['offset'])
    return model.reshape(height, width)


# ------------------------------------------------------------------ self-test
def _fit_gaussian_beam_self_test():
    """Synthetic beams through fit_gaussian_beam - no camera, no files.

        python -m utilities.utils
    """
    size = 241
    xx, yy = np.meshgrid(np.arange(size, dtype=float), np.arange(size, dtype=float))

    def beam(sigma_major, sigma_minor, major_deg, x_0=118.4, y_0=125.7, noise=5.0,
             seed=0):
        grid = np.array((xx, yy))
        image = gaussian_2d(grid, 800.0, x_0, y_0, sigma_major, sigma_minor,
                            np.radians(major_deg), 20.0).reshape(size, size)
        return image + np.random.default_rng(seed).normal(0.0, noise, image.shape)

    # --- every orientation, the old routine's +/- 45 deg seam included -------
    for major_deg in range(-89, 90, 7):
        success, pars = fit_gaussian_beam(beam(40.0, 11.0, major_deg), rebinning=2)
        assert success, major_deg
        wanted = (major_deg + 90) % 180 - 90
        turned = abs(((pars['major_axis_deg'] - wanted + 90) % 180) - 90)
        assert turned < 3.0, (major_deg, pars['major_axis_deg'])
        assert abs(pars['w_major'] - 80.0) < 4.0, (major_deg, pars['w_major'])
        assert abs(pars['w_minor'] - 22.0) < 4.0, (major_deg, pars['w_minor'])
        # the major axis is the wider one, by construction of the report
        assert pars['w_major'] >= pars['w_minor']
    print('fit ok: every tilt from -89 to +89 deg recovered, widths and angle')

    # --- the centre survives rebinning, offsets and all --------------------
    for rebinning in (1, 2, 4, 8):
        _, pars = fit_gaussian_beam(beam(40.0, 11.0, 35.0), rebinning=rebinning)
        assert abs(pars['x_0'] - 118.4) < 0.5, (rebinning, pars['x_0'])
        assert abs(pars['y_0'] - 125.7) < 0.5, (rebinning, pars['y_0'])
    print('centre ok: unmoved by rebinning 1, 2, 4 and 8')

    # --- a dragged guess sets the scale, not the shape ---------------------
    _, pars = fit_gaussian_beam(beam(40.0, 11.0, -50.0), rebinning=2,
                                manual_guess={'x_0': 120.0, 'y_0': 124.0,
                                              'sigma': 36.0})
    assert abs(((pars['major_axis_deg'] + 50.0 + 90) % 180) - 90) < 3.0, pars
    assert abs(pars['w_major'] - 80.0) < 4.0, pars
    print('manual guess ok: a circle dragged over a tilted beam still fits it')

    # --- a round beam is round, whatever angle comes back ------------------
    _, pars = fit_gaussian_beam(beam(30.0, 30.0, 0.0), rebinning=2)
    assert abs(pars['ellipticity'] - 1.0) < 0.02, pars['ellipticity']
    print(f"round beam ok: ellipticity {pars['ellipticity']:.4f}")

    # --- the angle is the one matplotlib's Ellipse takes, unflipped --------
    from matplotlib.patches import Ellipse
    _, pars = fit_gaussian_beam(beam(40.0, 11.0, -37.0), rebinning=2)
    ellipse = Ellipse((pars['x_0'], pars['y_0']), 2 * pars['w_major'],
                      2 * pars['w_minor'], angle=pars['major_axis_deg'])
    tip = ellipse.get_patch_transform().transform([(1.0, 0.0)])[0]
    drawn = np.degrees(np.arctan2(tip[1] - pars['y_0'], tip[0] - pars['x_0']))
    assert abs((((drawn + 37.0) + 90) % 180) - 90) < 3.0, drawn
    print(f"overlay ok: Ellipse(angle=major_axis_deg) lies at "
          f"{(drawn + 90) % 180 - 90:+.1f} deg, the beam at -37 deg")

    # --- the model can be put back on the frame it came from ---------------
    image = beam(40.0, 11.0, 22.0, noise=5.0)
    _, pars = fit_gaussian_beam(image, rebinning=2)
    model = gaussian_beam_image(pars, image.shape)
    assert model.shape == image.shape
    residual = float(np.sqrt(((model - image) ** 2).mean()))
    assert residual < 7.0, residual   # the noise it was given was 5 counts
    print(f'model ok: rms residual {residual:.2f} counts against 5.0 of noise')

    # --- a blank frame reports something rather than raising ---------------
    success, pars = fit_gaussian_beam(np.full((48, 48), 7.0))
    assert {'w_minor', 'w_major', 'major_axis_deg'} <= set(pars), sorted(pars)
    print('blank frame ok: parameters returned, nothing raised')
    print('fit_gaussian_beam self-test passed')


if __name__ == '__main__':
    _fit_gaussian_beam_self_test()
