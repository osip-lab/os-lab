# general functions for the project
from typing import Optional
import matplotlib.pyplot as plt
import cv2
import pyperclip
import os
import time

I = 0


def wait_for_path_from_clipboard(filetype: Optional[str] = None, poll_interval=0.5, verbose=True, instructions_message=None):
    if instructions_message is not None:
        print(instructions_message + '\n')
    global I
    while True:
        clipboard = pyperclip.paste().strip().strip('"')  # Strip whitespace and quotes

        if os.path.isfile(clipboard) or os.path.isdir(clipboard):
            filetype_lower = filetype.lower() if filetype else None

            if filetype_lower in ['video', 'media']:
                # Try to open as a video
                cap = cv2.VideoCapture(clipboard)
                if cap.isOpened():
                    cap.release()
                    if verbose:
                        print(f"✔ Detected valid video path: {clipboard}")
                    return clipboard
                cap.release()

            if filetype_lower == 'image' or filetype_lower == 'media':
                # Try to read as an image
                img = cv2.imread(clipboard)
                if img is not None:
                    if verbose:
                        print(f"✔ Detected valid image path: {clipboard}")
                    return clipboard

            if filetype_lower == 'csv':
                if clipboard.endswith('.csv'):
                    if verbose:
                        print(f"✔ Detected valid CSV path: {clipboard}")
                    return clipboard

            if filetype_lower in ['folder', 'directory', 'dir']:
                if os.path.isdir(clipboard):
                    if verbose:
                        print(f"✔ Detected valid directory path: {clipboard}")
                    return clipboard

            if filetype is not None and filetype_lower not in ['video', 'image', 'media', 'csv', 'folder', 'directory', 'dir']:
                raise NotImplementedError(
                    f"File type '{filetype}' not implemented. For skipping filetype validation just set filetype=None."
                )

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
                os.remove(avi_path)


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