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
    clipped = np.clip(high_passed - np.percentile(high_passed, 90), 0, None)
    return clipped


def high_pass_n_clip_in_folder(folder_path, include_substrings=None, f_threshold=30):
    if isinstance(include_substrings, str):
        include_substrings = [include_substrings]
    elif include_substrings is None:
        include_substrings = []

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".png") and any(sub in filename for sub in include_substrings):
            # Your processing logic here
            ...
            file_path = os.path.join(folder_path, filename)
            image = cv2.imread(file_path)

            if image is None:
                print(f"Failed to load image: {file_path}")
                continue

            processed = high_pass_n_clip_file(image, f_threshold)
            # remove 10'th percentile from the image and clip at 0:

            # Save the new image with "_editd.png" suffix
            name, ext = os.path.splitext(filename)
            output_path = os.path.join(folder_path, f"High pass filtered\\{name}.png")
            cv2.imwrite(output_path, processed)
            print(f"Saved: {output_path}")


def load_images_from_folder(folder):
    supported_formats = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
    image_paths = []
    for ext in supported_formats:
        image_paths.extend(sorted(glob(os.path.join(folder, ext))))
    images = [cv2.imread(path) for path in image_paths]
    return images, image_paths


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


def save_images(images, original_paths, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for img, path in zip(images, original_paths):
        filename = os.path.basename(path)
        out_path = os.path.join(output_folder, filename)
        cv2.imwrite(out_path, img)


def clean_consistent_noises(folder_path):
    images, paths = load_images_from_folder(folder_path)
    if len(images) < 2:
        raise ValueError("Need at least two images to identify consistent noise.")

    print(f"Loaded {len(images)} images.")

    noise_mask = detect_static_noise_mask(images, brightness_threshold=2, min_occurrence_fraction=0.99)
    cleaned_images = remove_noise(images, noise_mask)

    output_folder = os.path.join(folder_path, "removed_noise")
    save_images(cleaned_images, paths, output_folder)
    print(f"Saved cleaned images to {output_folder}")


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


def align_folder(folder_path: str, reference_image_path: str):
    images, paths = load_images_from_folder(folder_path)
    reference_image = cv2.imread(reference_image_path)
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

        filename = os.path.basename(path)
        output_folder = os.path.join(folder_path, f"aligned")
        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, aligned_img)

    return aligned_images


# %%
# 🟦 Usage Example
# Replace this with your folder path
FREQUENCY_CUTOFF = 20
folder_path = wait_for_path_from_clipboard('directory')
high_pass_n_clip_in_folder(folder_path, f_threshold=FREQUENCY_CUTOFF)
clean_consistent_noises(r"C:\Users\michaeka\Desktop\IPA test\High pass filtered")  # os.path.join(folder_path, HIGH_PASSED_FOLDER)
align_folder(r"C:\Users\michaeka\Desktop\IPA test\High pass filtered\removed_noise", r"C:\Users\michaeka\Desktop\IPA test\High pass filtered\removed_noise\1 - 5X - 5000ms.png")
# sum_intensity_in_polygon()


