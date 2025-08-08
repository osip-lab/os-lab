from typing import Tuple

import cv2
import pyautogui
import time
from PIL import ImageGrab, Image
import numpy as np
import pyautogui
from mss import mss
from screeninfo import get_monitors


def get_printscreen_clipboard_image(as_numpy: bool = False) -> Image.Image | np.ndarray:
    """
    Simulates pressing Print Screen, waits, and returns image from clipboard.

    Args:
        as_numpy (bool): If True, returns a NumPy array instead of PIL Image.

    Returns:
        PIL.Image or np.ndarray: Screenshot image from clipboard.

    Raises:
        RuntimeError: If no image is found in the clipboard.
    """
    # Simulate Print Screen key press
    pyautogui.press('printscreen')
    time.sleep(0.5)  # Wait a bit for clipboard to update

    img = ImageGrab.grabclipboard()
    if isinstance(img, Image.Image):
        return np.array(img) if as_numpy else img
    else:
        raise RuntimeError("No image found in clipboard after Print Screen.")


def get_virtual_screen_bounds():
    monitors = get_monitors()
    min_x = min(monitor.x for monitor in monitors)
    min_y = min(monitor.y for monitor in monitors)
    max_x = max(monitor.x + monitor.width for monitor in monitors)
    max_y = max(monitor.y + monitor.height for monitor in monitors)
    return (min_x, min_y), (max_x - 1, max_y - 1)
# %%
from PIL import ImageGrab

img = ImageGrab.grab()  # Takes screenshot of primary monitor (or full virtual desktop)
img_np = np.array(img)
# %%
# img_np = get_printscreen_clipboard_image(as_numpy=True)
import matplotlib.pyplot as plt
from matplotlib import use
use('Qt5Agg')  # wor# ks with PyCharm Console
plt.imshow(img_np, cmap='gray')
plt.show()
# %%
def capture_all_screens() -> Tuple[np.ndarray, int, int]:
    with mss() as sct:
        full_bounds = sct.monitors[0]  # monitor[0] is the bounding box for all screens
        screenshot = np.array(sct.grab(full_bounds))
        img_rgb = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2RGB)
        return img_rgb, full_bounds["left"], full_bounds["top"]

# %%
img_np, _, _ = capture_all_screens()
import matplotlib.pyplot as plt
from matplotlib import use
use('Qt5Agg')  # works with PyCharm Console
plt.imshow(img_np, cmap='gray')
plt.show()
# %%
print(pyautogui.position())

# from screeninfo import get_monitors
#
#
#
# top_left, bottom_right = get_virtual_screen_bounds()
# print(f"Top-left: {top_left}, Bottom-right: {bottom_right}")
# # %%
# from plyer import notification
#
# notification.notify(
#     title="Hello",
#     message="This is your main screen",
#     timeout=5
# )