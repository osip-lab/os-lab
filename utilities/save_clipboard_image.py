from PIL import ImageGrab
from datetime import datetime
import os

# Get current timestamp
now = datetime.now()
current_time = now.strftime("%y%m%d%H%M%S")

# Grab image from clipboard
im = ImageGrab.grabclipboard()

# Get the user's desktop path
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")

# Construct full path for saving the image
save_path = os.path.join(desktop_path, f"{current_time}.png")

# Save the image
if im:
    im.save(save_path)
    print(f"Image saved to: {save_path}")
else:
    print("No image found in clipboard.")
