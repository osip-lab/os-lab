import os
import numpy as np
import cv2
from glob import glob
from utilities.media_tools.utils import wait_for_path_from_clipboard
from matplotlib import pyplot as plt
from typing import Union
from matplotlib.path import Path
from matplotlib.widgets import PolygonSelector

HIGH_PASSED_FOLDER = "High pass filtered"
CLEANED_CONSISTENT_NOISES_FOLDER = "Removed noise"


def load_images_from_folder(folder, include_subfolders=False, include_substrings=None, exclude_folders=None):
    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp')
    if isinstance(include_substrings, str):
        include_substrings = [include_substrings]
    elif include_substrings is None:
        include_substrings = ['']

    if isinstance(exclude_folders, str):
        exclude_folders = [exclude_folders]
    elif exclude_folders is None:
        exclude_folders = []

    def is_excluded(path):
        return any(exclude in path for exclude in exclude_folders)

    if include_subfolders:
        image_paths = [
            os.path.join(root, fname)
            for root, _, files in os.walk(folder)
            if not is_excluded(root)
            for fname in files
            if fname.lower().endswith(supported_formats) and any(sub in fname for sub in include_substrings)
        ]
    else:
        image_paths = [
            os.path.join(folder, fname)
            for fname in os.listdir(folder)
            if fname.lower().endswith(supported_formats) and any(sub in fname for sub in include_substrings)
        ]

    image_paths.sort()
    images = [cv2.imread(path) for path in image_paths]
    return images, image_paths


def save_images_in_new_subfolder(images, original_paths, output_folder, base_folder):
    for img, path in zip(images, original_paths):
        relative_path = os.path.relpath(path, base_folder)
        output_path = os.path.join(output_folder, relative_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, img)

def high_pass_n_clip_file(img, f_threshold):
    # Split into B, G, R channels
    channels = cv2.split(img)
    filtered_channels = []

    for ch in channels:
        # FFT
        dft = np.fft.fft2(ch)
        dft_shift = np.fft.fftshift(dft)

        # Create high-pass mask
        rows, cols = ch.shape
        crow, ccol = rows // 2, cols // 2
        mask = np.ones((rows, cols), np.uint8)
        mask[crow - f_threshold:crow + f_threshold, ccol - f_threshold:ccol + f_threshold] = 0

        # Apply mask
        filtered_dft = dft_shift * mask

        # Inverse FFT
        f_ishift = np.fft.ifftshift(filtered_dft)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)

        # Normalize
        img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
        filtered_channels.append(img_back.astype(np.uint8))

    # Merge filtered channels back to a color image
    high_passed = cv2.merge(filtered_channels)
    clipped = np.clip(high_passed - np.percentile(high_passed, 95), 0, None)
    return clipped


def high_pass_n_clip_in_folder(folder_path: str,
                               f_threshold=30,
                               include_subfolders: bool = False,
                               include_substrings: Union[str, list] = None):
    if isinstance(include_substrings, str):
        include_substrings = [include_substrings]
    elif include_substrings is None:
        include_substrings = ['']

    images, paths = load_images_from_folder(folder_path, include_subfolders, exclude_folders=HIGH_PASSED_FOLDER)
    for image, full_path in zip(images, paths):
        fname = os.path.basename(full_path)
        if not any(sub in fname for sub in include_substrings):
            continue

        if image is None:
            print(f"Failed to load image: {full_path}")
            continue

        processed = high_pass_n_clip_file(image, f_threshold)
        relative_path = os.path.relpath(full_path, folder_path)
        relative_dir = os.path.dirname(relative_path)
        name, ext = os.path.splitext(os.path.basename(relative_path))
        output_base = os.path.join(folder_path, HIGH_PASSED_FOLDER, relative_dir)
        os.makedirs(output_base, exist_ok=True)
        output_path = os.path.join(output_base, f"{name}.png")
        cv2.imwrite(output_path, processed)
        print(f"Saved: {output_path}")
    return os.path.join(folder_path, HIGH_PASSED_FOLDER)


def detect_static_noise_mask(images, brightness_threshold=40, min_occurrence_fraction=0.99):
    # Convert all images to grayscale and threshold for bright pixels
    masks = []
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, brightness_threshold, 255, cv2.THRESH_BINARY)
        masks.append(mask.astype(bool))

    # Stack all masks into a 3D boolean array and sum along axis 0
    mask_stack = np.stack(masks, axis=0)
    consistent_noise = np.mean(mask_stack, axis=0) >= min_occurrence_fraction
    return consistent_noise.astype(np.uint8) * 255  # Return as 0/255 mask for inpainting


def remove_noise(images, noise_mask):
    inpainted = []
    for img in images:
        cleaned = cv2.inpaint(img, noise_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
        inpainted.append(cleaned)
    return inpainted


def clean_consistent_noises(folder_path: str, include_subfolders: bool = False, include_substrings: Union[str, list] = None):
    images, paths = load_images_from_folder(folder_path, include_subfolders=include_subfolders, include_substrings=include_substrings, exclude_folders=CLEANED_CONSISTENT_NOISES_FOLDER)
    if len(images) < 2:
        raise ValueError("Need at least two images to identify consistent noise.")

    print(f"Loaded {len(images)} images.")
    noise_mask = detect_static_noise_mask(images, brightness_threshold=2, min_occurrence_fraction=0.99)
    cleaned_images = remove_noise(images, noise_mask)
    output_folder = os.path.join(folder_path, CLEANED_CONSISTENT_NOISES_FOLDER)
    save_images_in_new_subfolder(cleaned_images, paths, output_folder, base_folder=folder_path)
    print(f"Saved cleaned images to {output_folder}")
    return output_folder


def align_images_from_3_points(
    img1: np.ndarray,
    img2: np.ndarray,
    saved_reference_points=None,
    ref_name: str = "Reference Image",
    cur_name: str = "Current Image"
):
    if saved_reference_points is None:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(img1)
        axes[0].set_title(f"{ref_name}: Click 3 points")
        axes[1].imshow(img2)
        axes[1].set_title(f"{cur_name}: Click 3 points or press 's' to skip")

        plt.suptitle("Click 3 points in Reference Image (left), then 3 points in Current Image (right)")
        points = plt.ginput(6, timeout=0)
        plt.close()

        if len(points) != 6:
            print("Skipping image. You must click exactly 3 points on each image.")
            return None, None

        pts1 = np.float32(points[:3])
        pts2 = np.float32(points[3:])
    else:
        pts1 = saved_reference_points
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(img2)
        ax.set_title(f"{cur_name}: Click 3 points or press 's' to skip")

        plt.suptitle("Click 3 points in Current Image (right)")
        points = plt.ginput(3, timeout=0)
        plt.close()

        if len(points) != 3:
            print("Skipping image. You must click exactly 3 points on the current image.")
            return None, None

        pts2 = np.float32(points)

    # Compute affine transform from img2 to img1
    M = cv2.getAffineTransform(pts2, pts1)
    h, w = img2.shape[:2]
    aligned_img2 = cv2.warpAffine(img2, M, (w, h))

    return aligned_img2, pts1


def align_folder(folder_path: str, reference_image_path: str, include_subfolders: bool = False):
    reference_image = cv2.imread(reference_image_path)
    if reference_image is None:
        raise ValueError(f"Could not load reference image: {reference_image_path}")
    images, paths = load_images_from_folder(folder_path, include_subfolders)
    aligned_images = []
    saved_reference_points = None
    ref_name = os.path.basename(reference_image_path)
    for img, path in zip(images, paths):
        if os.path.abspath(path) == os.path.abspath(reference_image_path):
            continue
        cur_name = os.path.basename(path)
        print(f"Processing: {path}")
        if saved_reference_points is None:
            aligned_img, saved_reference_points = align_images_from_3_points(reference_image, img, None, ref_name, cur_name)
        else:
            aligned_img, _ = align_images_from_3_points(reference_image, img, saved_reference_points, ref_name, cur_name)
        if aligned_img is None:
            print(f"Skipped: {path}")
            continue
        aligned_images.append(aligned_img)
        relative_path = os.path.relpath(path, folder_path)
        output_path = os.path.join(folder_path, "aligned", relative_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, aligned_img)
        print(f"Saved: {output_path}")
    return aligned_images


def sum_intensity_in_polygon(image: Union[None, str, np.ndarray] = None) -> None:
    # Step 1: Load image
    if image is None:
        path = wait_for_path_from_clipboard(filetype='image')
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif isinstance(image, str):
        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif isinstance(image, np.ndarray):
        pass
    else:
        raise ValueError("Invalid image input type.")

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.set_title("Draw polygon. Press Enter to finish.")

    # Shared mutable state
    done = {'finished': False}
    selector = {'object': None}

    def on_key(event):
        if event.key == 'enter':
            done['finished'] = True
            plt.close()

    # Create and store selector
    selector['object'] = PolygonSelector(ax, lambda verts: None, useblit=True, props={'color': 'red', 'alpha': 0.5})
    fig.canvas.mpl_connect('key_press_event', on_key)

    # Block until Enter pressed
    plt.show(block=True)

    poly = selector['object'].verts
    if not done['finished'] or poly is None or len(poly) < 3:
        print("Polygon must have at least 3 points.")
        return

    # Create mask
    height, width = gray.shape
    poly_path = Path(poly)
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    coords = np.stack((x.flatten(), y.flatten()), axis=-1)
    mask = poly_path.contains_points(coords).reshape((height, width))

    # Compute intensity metrics
    values = gray[mask]
    total = np.sum(values)
    mean = np.mean(values)
    print(len(poly))
    print(f"Total intensity inside polygon: {total}")
    print(f"Mean intensity inside polygon: {mean}")

# %%
# 🟦 Usage Example
# Replace this with your folder path
folder_path = r"C:\Users\michaeka\Weizmann Institute Dropbox\Michael Kali\Labs Dropbox\Laser Phase Plate\Elements inspection\ASA insepction\1162025 adaptor"
FREQUENCY_CUTOFF = 20
high_pass_filtered_folder = 'C:\\Users\\michaeka\\Weizmann Institute Dropbox\\Michael Kali\\Labs Dropbox\\Laser Phase Plate\\Elements inspection\\ASA insepction\\1162025 adaptor\\High pass filtered'  # high_pass_n_clip_in_folder(folder_path, f_threshold=FREQUENCY_CUTOFF, include_subfolders=True)
cleaned_noise_folder = clean_consistent_noises(high_pass_filtered_folder, include_subfolders=True)
# %%
align_folder(r"C:\Users\michaeka\Weizmann Institute Dropbox\Michael Kali\Labs Dropbox\Laser Phase Plate\Elements inspection\ASA insepction\1162025 adaptor\High pass filtered",
             r"C:\Users\michaeka\Weizmann Institute Dropbox\Michael Kali\Labs Dropbox\Laser Phase Plate\Elements inspection\ASA insepction\1162025 adaptor\High pass filtered\before tapping\5X  -  5000ms - 400%.png",
             include_subfolders=True)
# %%
sum_intensity_in_polygon()


# %%
file_path = r"C:\Users\michaeka\Weizmann Institute Dropbox\Michael Kali\Labs Dropbox\Laser Phase Plate\Elements inspection\Coast line\FS - 200mm\mirror 0104\High pass filtered\Removed noise\Planar side\5X - off center focus 5000ms - 400.png"
image = cv2.imread(file_path)


