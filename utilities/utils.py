# general functions for the project
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