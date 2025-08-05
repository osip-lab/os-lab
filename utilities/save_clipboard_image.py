from PIL import ImageGrab
from datetime import datetime

now = datetime.now()

current_time = now.strftime("%y%m%d%H%M%S")

im=ImageGrab.grabclipboard()
im.save(r"C:\Users\michaeka\Desktop\{}.png".format(current_time))
