# general functions for the project
from typing import Optional
import matplotlib.pyplot as plt
import cv2
import pyperclip
import os
import time




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