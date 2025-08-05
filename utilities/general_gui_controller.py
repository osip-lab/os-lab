import time

import pytesseract
import pyautogui
import cv2
import pyperclip
from pynput import keyboard
import numpy as np
from PIL import ImageGrab
import os
from typing import Optional, Tuple
from mss import mss
import pyautogui
from pynput import keyboard
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import pandas as pd
from time import sleep
import re

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def wait_for_path_from_clipboard(filetype: Optional[str] = None, poll_interval=0.5, verbose=True):
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

            if filetype is not None and filetype_lower not in ['video', 'image', 'media', 'csv', 'folder', 'directory',
                                                               'dir']:
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


def get_cursor_position(target_name):
    print(f"Place the cursor over the\n{target_name}\nand press the left 'Ctrl' on the keyboard")

    # Define a variable to store the position
    position = None

    # Function to record the cursor position
    def on_press(key):
        nonlocal position
        if key == keyboard.Key.ctrl_l:
            position = pyautogui.position()
            # Stop the listener after capturing position
            return False

    # Start the keyboard listener and wait until Enter is pressed
    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()

    print(f"Cursor position recorded for {target_name}: {position}\n")
    return position


def detect_position(
        input_template: str,
        relative_position: Optional[Tuple[float, float]] = None,
        crop_ll: Optional[Tuple[int, int]] = None,
        crop_ur: Optional[Tuple[int, int]] = None,
        click: bool = True,
        sleep: Optional[float] = None,
        **kwargs,
) -> tuple[float, float] | None:
    """
    Detect the position of a text or image template on the screen(s).

    Args:
        input_template (str): File under templates/ (.txt or image), or raw name for text search.
        relative_position (Optional[Tuple[float, float]]): (x%, y%) offset from bottom-left of matched box.
        crop_ll (Optional[Tuple[int, int]]): Lower-left (x, y) crop bounding box.
        crop_ur (Optional[Tuple[int, int]]): Upper-right (x, y) crop bounding box.

    Returns:
        (x, y): Tuple of floats in global screen coordinates or None if not found.
    """

    def capture_all_screens() -> Tuple[np.ndarray, int, int]:
        with mss() as sct:
            full_bounds = sct.monitors[0]  # full area including all monitors
            screenshot = np.array(sct.grab(full_bounds))
            img_rgb = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2RGB)
            return img_rgb, full_bounds["left"], full_bounds["top"]

    def compute_point(x, y, w, h):
        print(f"x={x}, y={y}, w={w}, h={h}")
        x0, y0 = x, y + h
        if relative_position:
            dx = w * relative_position[0]
            dy = h * relative_position[1]
            return x0 + dx, y0 - dy
        else:
            return x + w / 2, y + h / 2

    # 1. Determine the type and contents of the template
    template_path = os.path.join("templates", input_template)
    base_name, ext = os.path.splitext(template_path)
    ext = ext.lower()

    if ext == ".txt":
        try:
            with open(template_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            target_text = content if content else os.path.splitext(os.path.basename(input_template))[0]
        except FileNotFoundError:
            target_text = os.path.splitext(os.path.basename(input_template))[0]
        ext = ".txt"  # force OCR logic

    elif ext in [".png", ".jpg", ".jpeg", ".bmp"]:
        target_text = None  # Will use image logic

    elif ext == "":  # <-- NEW HANDLING FOR EXTENSION-LESS INPUTS
        target_text = input_template.strip()
        ext = ".txt"  # Treat it as a text search via OCR

    else:
        print(f"[ERROR] Unsupported template type: {ext}")
        return None

    # 2. Capture all monitors
    screen_rgb, origin_x, origin_y = capture_all_screens()
    full_h, full_w = screen_rgb.shape[:2]

    # 3. Crop image if needed
    if crop_ll and crop_ur:
        x1, y1 = crop_ll
        x2, y2 = crop_ur
        top = min(y1, y2)
        bottom = max(y1, y2)
        left = min(x1, x2)
        right = max(x1, x2)

        rel_top = max(0, top - origin_y)
        rel_bottom = min(full_h, bottom - origin_y)
        rel_left = max(0, left - origin_x)
        rel_right = min(full_w, right - origin_x)

        screen_rgb = screen_rgb[rel_top:rel_bottom, rel_left:rel_right]
        crop_offset_x, crop_offset_y = rel_left + origin_x, rel_top + origin_y
    else:
        crop_offset_x, crop_offset_y = origin_x, origin_y

    # 4. Template Matching Logic
    if ext == ".txt":
        data = pytesseract.image_to_data(screen_rgb, output_type=pytesseract.Output.DICT, **kwargs)

        for i, text in enumerate(data['text']):
            if text.strip().lower() == target_text.lower():
                x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                px, py = compute_point(x, y, w, h)
                abs_x, abs_y = px + crop_offset_x, py + crop_offset_y
                print(f"[INFO] Found text '{target_text}' at ({abs_x:.1f}, {abs_y:.1f})")
                pyautogui.moveTo(abs_x, abs_y)
                if click:
                    pyautogui.click(abs_x, abs_y)
                if sleep is not None:
                    time.sleep(sleep)
                return (abs_x, abs_y)

        print(f"[WARNING] Text '{target_text}' not found.")
        return None

    elif ext in [".png", ".jpg", ".jpeg", ".bmp"]:
        template_img = cv2.imread(template_path, cv2.IMREAD_COLOR)
        if template_img is None:
            print(f"[ERROR] Failed to load image template: {template_path}")
            return None

        template_gray = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)
        screen_gray = cv2.cvtColor(screen_rgb, cv2.COLOR_RGB2GRAY)

        best_val = -1
        best_loc = None
        best_w, best_h = 0, 0

        scales = [1.0] + [s for s in np.linspace(0.5, 2.0, 20) if abs(s - 1.0) > 1e-6]
        for scale in scales:
            resized_template = cv2.resize(template_gray, (0, 0), fx=scale, fy=scale)
            tH, tW = resized_template.shape[:2]

            if tH > screen_gray.shape[0] or tW > screen_gray.shape[1]:
                continue

            result = cv2.matchTemplate(screen_gray, resized_template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)

            if max_val > best_val:
                best_val = max_val
                best_loc = max_loc
                best_w, best_h = tW, tH
                if best_val > 0.8:
                    break

        if best_loc and best_val > 0.5:
            x, y = best_loc
            px, py = compute_point(x, y, best_w, best_h)
            abs_x, abs_y = px + crop_offset_x, py + crop_offset_y
            print(f"[INFO] Found image at ({abs_x:.1f}, {abs_y:.1f}) with score={best_val:.3f}")
            pyautogui.moveTo(abs_x, abs_y)
            if click:
                pyautogui.click(abs_x, abs_y)
            if sleep is not None:
                time.sleep(sleep)
            return (abs_x, abs_y)
        else:
            print("[WARNING] Image template not found.")
            return None

    else:
        print(f"[ERROR] Unsupported template type: {ext}")
        return None


def paste_value(value, location):
    """Pastes a given value at a specified screen location."""
    # Copy the value to clipboard
    if value is None:
        return
    pyautogui.click(location)  # Click to focus on the field
    pyautogui.hotkey("ctrl", "a")  # Select any existing text
    pyautogui.hotkey("backspace")  # Clear the field
    pyperclip.copy(str(value))  # Copy the Hebrew text to the clipboard
    pyautogui.hotkey('ctrl', 'v')
    sleep(0.1)


def clean_number(x):
    if isinstance(x, str):
        # Remove commas before extracting numbers
        x_no_commas = x.replace(',', '')
        match = re.search(r'[\d.]+', x_no_commas)
        x_extracted = float(match.group()) if match else None
        return x_extracted
    elif isinstance(x, (int, float)):
        return x
    else:
        return None


def clean_text(x):
    # Removes '*' from text:
    if isinstance(x, str):
        return x.replace('*', '').strip()
    elif isinstance(x, (int, float)):
        return str(x).strip()
    else:
        return None


def load_and_select_columns(thorlabs_format=True):
    # Hide the main Tkinter window
    Tk().withdraw()

    # Open file dialog and let the user choose a file
    file_path = wait_for_path_from_clipboard(filetype='csv')
    # file_path = askopenfilename(title="Select a CSV or Excel file",
    #                 filetypes=[("Excel or CSV files", ".csv .xlsx .xls")])
    if not file_path:
        raise ValueError("No file was selected.")

    # Check the file extension and load the file
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith('.xlsx'):
        df = pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format. Please select a .csv or .xlsx file.")

    # Define the target columns in lowercase
    required_columns = ['id', 'description', 'quantity', 'price', 'discount']
    if thorlabs_format:
        # remove the last line of the dataframe if it's Ln value is nan:
        if df.iloc[-1]['Quantity'] == 'TOTAL':
            df = df.iloc[:-1]
        df[['id', 'description']] = df['Part Number and Description'].str.split('\n', n=1, expand=True)
        # rename the column "unit price" to "price":
        df.rename(columns={'Unit Price': 'price'}, inplace=True)

    # Normalize column names to lowercase for matching
    df.columns = df.columns.str.lower()

    # Ensure all required columns exist, adding missing columns as empty
    for col in required_columns:
        if col not in df.columns:
            df[col] = None

    # Clean the columns
    # Clean the columns
    df['discount'] = df['discount'].apply(clean_number)
    df['discount'] = df['discount'].apply(lambda x: 0 if x is None else x * 100 if x < 1 else x)
    df['quantity'] = df['quantity'].apply(clean_number)
    df['price'] = df['price'].apply(clean_number)
    df['id'] = df['id'].apply(clean_text)
    df['description'] = df['description'].apply(clean_text)

    # Select and return only the required columns
    df = df.loc[:df.last_valid_index()]
    return df[required_columns]


# %%
# input("You will now be prompted to choose a csv\excel file for the quote details.\n"
#       "make sure the file has the following columns: ['id', 'description', 'quantity', 'price', 'discount'] (Capitalization of letter does not matter)\n"
#       "There is no need to remove strings from the values. that is, No need to change '10 %' to 10 and '250.00 USD' to 250.\n"
#       "Notice that in Excel you can choose Data -> Get Data -> From file -> From PDF to automaticall import tables from a PDF file to you excel.\n"
#       "Press Enter to continue")
# thorlabs_format = input("Is the quote in Thorlabs format? (Y/N) and press Enter to continue")
# if thorlabs_format.lower() == 'y':
#     thorlabs_format = True
# elif thorlabs_format.lower() == 'n':
#     thorlabs_format = False
#
# df = load_and_select_columns(thorlabs_format=thorlabs_format)

