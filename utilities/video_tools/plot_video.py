import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import os
from matplotlib.patches import Circle
import matplotlib
from utilities.video_tools.utils import wait_for_video_path_from_clipboard

matplotlib.use('Qt5Agg')  # Or 'TkAgg' if Qt5Agg doesn't work

# ---- Set video path ----
video_path = wait_for_video_path_from_clipboard(filetype='video')

# ---- Set video path ----

if not os.path.exists(video_path):
    raise FileNotFoundError(f"Video not found: {video_path}")

# ---- Load video into memory ----
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise IOError(f"Cannot open video: {video_path}")

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
frames = []

while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frames.append(gray)

cap.release()

frames = np.array(frames)
frames = frames - np.quantile(frames, 0.3, axis=0, keepdims=True)

# %%
PIXEL_SIZE_MM = 0.0055  # 5.5 microns in mm

# ---- Initialize plot ----
vmax_default = np.max(frames[0])
fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.3)

img_disp = ax.imshow(frames[0], cmap='gray', vmin=0, vmax=vmax_default)
cb = plt.colorbar(img_disp, ax=ax)

# Sliders
ax_frame = plt.axes([0.25, 0.2, 0.65, 0.03])
frame_slider = Slider(ax_frame, 'Frame', 0, len(frames) - 1, valinit=0, valstep=1)

ax_vmax = plt.axes([0.25, 0.15, 0.65, 0.03])
vmax_slider = Slider(ax_vmax, 'vmax', 1, 255, valinit=vmax_default)

# For Circle Fitting
clicked_points = []
circle_patch = None
current_frame_index = 0

def calc_circle(x1, y1, x2, y2, x3, y3):
    temp = x2**2 + y2**2
    bc = (x1**2 + y1**2 - temp) / 2
    cd = (temp - x3**2 - y3**2) / 2
    det = (x1 - x2)*(y2 - y3) - (x2 - x3)*(y1 - y2)
    if abs(det) < 1e-10:
        raise ValueError("Points are colinear")
    cx = (bc*(y2 - y3) - cd*(y1 - y2)) / det
    cy = ((x1 - x2)*cd - (x2 - x3)*bc) / det
    r = np.sqrt((cx - x1)**2 + (cy - y1)**2)
    return cx, cy, r

def update_title(frame_idx, radius=None):
    time = frame_idx / fps
    if radius is not None:
        radius_mm = radius * PIXEL_SIZE_MM
        title = f"Frame {frame_idx} (Time: {time:.2f}s) | Radius: {radius:.2f}px = {radius_mm:.3f}mm"
    else:
        title = f"Frame {frame_idx} (Time: {time:.2f}s)"
    ax.set_title(title)

def update(val):
    global current_frame_index, circle_patch, clicked_points
    current_frame_index = int(frame_slider.val)
    vmax_val = vmax_slider.val
    img_disp.set_data(frames[current_frame_index])
    img_disp.set_clim(vmin=0, vmax=vmax_val)
    if circle_patch:
        circle_patch.remove()
        circle_patch = None
    clicked_points = []
    update_title(current_frame_index)
    fig.canvas.draw_idle()

frame_slider.on_changed(update)
vmax_slider.on_changed(update)

def on_key(event):
    current = int(frame_slider.val)
    if event.key == 'right' and current < len(frames) - 1:
        frame_slider.set_val(current + 1)
    elif event.key == 'left' and current > 0:
        frame_slider.set_val(current - 1)

def on_click(event):
    global clicked_points, circle_patch

    if event.inaxes != ax or plt.get_current_fig_manager().toolbar.mode != '':
        return

    if len(clicked_points) >= 3:
        clicked_points = []
        ax.images[0].set_data(frames[current_frame_index])
        if circle_patch:
            circle_patch.remove()
            circle_patch = None

    x, y = event.xdata, event.ydata
    clicked_points.append((x, y))
    ax.plot(x, y, 'ro')

    if len(clicked_points) == 3:
        try:
            (x1, y1), (x2, y2), (x3, y3) = clicked_points
            cx, cy, radius = calc_circle(x1, y1, x2, y2, x3, y3)
            radius_mm = radius * PIXEL_SIZE_MM
            print(f"Radius: {radius:.2f} pixels, {radius_mm:.3f} mm")
            circle_patch = Circle((cx, cy), radius, color='cyan', linestyle='--', fill=False, linewidth=2, alpha=0.5)
            ax.add_patch(circle_patch)
            update_title(current_frame_index, radius)
            plt.draw()
        except ValueError as e:
            print("Error:", e)

fig.canvas.mpl_connect('key_press_event', on_key)
fig.canvas.mpl_connect('button_press_event', on_click)

update_title(0)
plt.show()