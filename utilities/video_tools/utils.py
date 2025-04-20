# general functions for the project
from typing import Optional

import cv2
import pyperclip
import os
import time

I = 0


def wait_for_path_from_clipboard(filetype: Optional[str] = None, poll_interval=0.5, verbose=True):
    global I
    while True:
        clipboard = pyperclip.paste().strip().strip('"')  # Strip whitespace and quotes

        if os.path.isfile(clipboard) or os.path.isdir(clipboard):
            if filetype.lower() == 'video':
                cap = cv2.VideoCapture(clipboard)
                if cap.isOpened():
                    cap.release()
                    if verbose:
                        print(f"✔ Detected valid video path: {clipboard}")
                    return clipboard
                cap.release()
            elif filetype.lower() == 'csv':
                if clipboard.endswith('.csv'):
                    if verbose:
                        print(f"✔ Detected valid CSV path: {clipboard}")
                    return clipboard
            elif filetype.lower() in ['folder', 'directory', 'dir']:
                if os.path.isdir(clipboard):
                    if verbose:
                        print(f"✔ Detected valid directory path: {clipboard}")
                    return clipboard
            elif filetype is not None:
                raise NotImplementedError(
                    f"File type '{filetype}' not implemented., for skipping filetype validation just set filetype=None")
            else:
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
