# general functions for the project
from typing import Optional

import cv2
import pyperclip
import os
import time

I = 0
def wait_for_video_path_from_clipboard(filetype: Optional[str] = None, poll_interval=0.5, verbose=True):
    global I
    while True:
        clipboard = pyperclip.paste().strip().strip('"')  # Strip whitespace and quotes

        if os.path.isfile(clipboard):
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
            elif filetype is not None:
                raise NotImplementedError(f"File type '{filetype}' not implemented., for skipping filetype validation just set filetype=None")
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
