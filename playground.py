import cv2
import numpy as np
import matplotlib.pyplot as plt

# load r""C:\Users\michaeka\Desktop\IPA test\1 - 5X - 5000ms_editd.png"" as image_ref:
image_ref = cv2.imread(r"C:\Users\michaeka\Desktop\IPA test\1 - 5X - 5000ms_editd.png")
image_to_align = cv2.imread(r"C:\Users\michaeka\Desktop\IPA test\2 - 5X - 5000ms_editd.png")


def align_images(image_ref, image_to_align):
    # Convert to grayscale
    gray_ref = cv2.cvtColor(image_ref, cv2.COLOR_RGB2GRAY)
    gray_to_align = cv2.cvtColor(image_to_align, cv2.COLOR_RGB2GRAY)

    # Get size
    h, w = gray_ref.shape

    # Step 1: Estimate rotation using log-polar + phase correlation
    center = (w // 2, h // 2)
    log_polar_ref = cv2.logPolar(gray_ref.astype(np.float32), center, 40, cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS)
    log_polar_align = cv2.logPolar(gray_to_align.astype(np.float32), center, 40, cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS)

    (dx_rot, dy_rot), _ = cv2.phaseCorrelate(log_polar_ref, log_polar_align)
    angle = -dy_rot * 360 / log_polar_ref.shape[0]  # Convert to degrees

    # Step 2: Rotate image_to_align
    rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image_to_align, rot_matrix, (w, h), flags=cv2.INTER_LINEAR)

    # Step 3: Translation using phase correlation
    gray_rotated = cv2.cvtColor(rotated, cv2.COLOR_RGB2GRAY)
    (dx, dy), _ = cv2.phaseCorrelate(gray_ref.astype(np.float32), gray_rotated.astype(np.float32))

    # Step 4: Translate image
    translation_matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    aligned = cv2.warpAffine(rotated, translation_matrix, (w, h), flags=cv2.INTER_LINEAR)

    return aligned, angle, dx, dy

# Example usage:
# image1 and image2 are NumPy arrays of shape (H, W, 3)
# aligned_image, angle, dx, dy = align_images(image1, image2)


# Plot ref, image to align and aligned image:
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(image_ref)
plt.title("Reference Image")
plt.axis("off")
plt.subplot(1, 3, 2)
plt.imshow(image_to_align)
plt.title("Image to Align")
plt.axis("off")
aligned_image, angle, dx, dy = align_images(image_ref, image_to_align)
plt.subplot(1, 3, 3)
plt.imshow(aligned_image)
plt.title("Aligned Image")
plt.axis("off")
plt.show()

