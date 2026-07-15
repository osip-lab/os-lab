# general functions for the project
import shutil
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import os
import pyperclip
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
                        print(f"✔ Detected valid video path: {clipboard}")
                    return clipboard
                cap.release()

            if filetype_lower in ['image', 'media']:
                # Try to read as an image
                img = cv2.imread(clipboard)
                if img is not None:
                    if verbose:
                        print(f"✔ Detected valid image path: {clipboard}")
                    return clipboard

            if filetype_lower == 'excel':
                if clipboard.endswith('.xlsx') or clipboard.endswith('.xls'):
                    if verbose:
                        print(f"✔ Detected valid CSV path: {clipboard}")
                    return clipboard

            if filetype_lower in ['table', 'tabular']:
                if clipboard.endswith('.csv') or clipboard.endswith('.xlsx') or clipboard.endswith('.xls'):
                    if verbose:
                        print(f"✔ Detected valid table path: {clipboard}")
                    return clipboard

            if filetype_lower in ['folder', 'directory', 'dir']:
                if os.path.isdir(clipboard):
                    if verbose:
                        print(f"✔ Detected valid directory path: {clipboard}")
                    return clipboard

            if filetype is not None:
                extensions = [filetype_lower] if filetype_is_str else [ft.lower() for ft in filetype]
                if any(clipboard.lower().endswith(f'.{ext}') for ext in extensions):
                    if verbose:
                        print(f"✔ Detected valid path: {clipboard}")
                    return clipboard

            if filetype is None:
                # No specific filetype validation
                if verbose:
                    print(f"✔ Detected path: {clipboard}")
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


def psdata_to_csv(psdata_path):
    """Convert a PicoScope .psdata file to CSV; return the CSV path.

    Uses PicoScope 7's command-line `BatchConvert` mode, which produces a CSV
    identical to the GUI's "Save as CSV". BatchConvert operates on folders,
    so the single file is copied to a temporary folder and converted there.
    Requires PICOSCOPE_EXE (the path of PicoScope.exe) in local_config.py,
    and PicoScope >= 7.1.32 (checked before launching).

    If an up-to-date CSV with the same name already sits next to the .psdata
    file (e.g. from an earlier manual export or an earlier run), it is used
    directly and no conversion is performed. If the .psdata file holds several
    waveform buffers, the user is asked which one to use (default: the last,
    i.e. most recent, capture).
    """
    psdata_path = Path(psdata_path)
    sibling_csv = psdata_path.with_suffix('.csv')
    if sibling_csv.is_file() and sibling_csv.stat().st_mtime >= psdata_path.stat().st_mtime:
        print(f"Using existing up-to-date CSV: {sibling_csv}")
        return str(sibling_csv)

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

    tmp_dir = Path(tempfile.mkdtemp(prefix='psdata_to_csv_'))
    in_dir = tmp_dir / 'in'
    out_dir = tmp_dir / 'out'
    in_dir.mkdir()
    out_dir.mkdir()
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
        raise RuntimeError(
            f"PicoScope did not finish converting within "
            f"{PSDATA_CONVERT_TIMEOUT_S} s and was closed. This usually means "
            "it opened as a normal GUI session instead of converting (e.g. it "
            "connected to an attached scope, or popped a device-selection "
            "dialog). Close any open PicoScope window and retry, or save the "
            "trace as CSV manually."
        )

    # A single-waveform file becomes '<stem>.csv' directly in out_dir; a file
    # with multiple waveform buffers becomes a '<stem>' subfolder holding
    # '<stem>_1.csv' ... '<stem>_N.csv', so search recursively and sort by the
    # numeric buffer suffix (plain name-sorting would put _10 before _2).
    def buffer_index(p):
        suffix = p.stem.rsplit('_', 1)[-1]
        return int(suffix) if suffix.isdigit() else 0

    csv_files = sorted(out_dir.rglob('*.csv'), key=buffer_index)
    if result.returncode != 0 or not csv_files:
        raise RuntimeError(
            f"psdata -> CSV conversion failed (exit code {result.returncode}).\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
    print("Conversion succeeded.")

    if len(csv_files) == 1:
        return str(csv_files[0])

    print(f"The psdata file contains {len(csv_files)} waveform buffers:")
    for i, p in enumerate(csv_files, start=1):
        print(f"  [{i}] {p.name}")
    while True:
        raw_in = input(
            f"Which waveform to use? 1-{len(csv_files)} "
            f"[default {len(csv_files)} - the most recent]: "
        ).strip()
        if raw_in == '':
            choice = len(csv_files)
        else:
            try:
                choice = int(raw_in)
            except ValueError:
                choice = 0
        if 1 <= choice <= len(csv_files):
            break
        print(f"  Please enter a number between 1 and {len(csv_files)}.")
    print(f"Using waveform: {csv_files[choice - 1].name}")
    return str(csv_files[choice - 1])


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
    print(f"✔ Saved figure to: {candidate}")