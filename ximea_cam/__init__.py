"""
Subpackage description.
"""
import numpy as np
from numba import njit
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from qt_gui.qt_ext import ThreadedWidget

try:
    from ximea import xiapi  # the third-party folder that must be copied
except Exception as e:
    raise ImportError(
        "Missing dependency 'ximea' required by ximea_cam.\n\n"
        "This library is distributed with Ximea API SP and must be copied into your Python "
        "environment's site-packages. See https://www.ximea.com/support/wiki/apis/Python_inst_win "
        "for exact instructions for Windows.\n"
    ) from e


@njit(fastmath=True)
def rebin_numba(src, dst, bs):
    h, w = src.shape
    new_h, new_w = h // bs, w // bs
    norm = 1.0 / (bs * bs)
    for y in range(new_h):
        for x in range(new_w):
            s = 0
            for dy in range(bs):
                for dx in range(bs):
                    s += src[y * bs + dy, x * bs + dx]
            dst[y, x] = s * norm


# ---------------- Controller ---------------- #
class XimeaCamControlWidget(ThreadedWidget):
    new_frame = pyqtSignal(np.ndarray)

    def __init__(self, font_size=14):
        super(XimeaCamControlWidget, self).__init__(font_size=font_size)
        self.setTitle('Camera Control')

        # create instance for first connected camera
        self.cam = xiapi.Camera()
        print('Opening first camera...')
        self.cam.open_device()

        self.cam.set_exposure(1000)
        print(f'Exposure was set to {int(self.cam.get_exposure())} us')

        self.cam.set_imgdataformat('XI_MONO16')  # need it to have 10-bit image depth

        # create instance of Image to store image data and metadata
        self.img = xiapi.Image()

        # start data acquisition
        print('Starting data acquisition...')
        self.cam.start_acquisition()

        # Image size and depth
        self.width, self.height = self.cam.get_width(), self.cam.get_height()
        print(f'Height: {self.height}, Width: {self.width}')
        print(self.cam.get_sensor_bit_depth())
        self.depth = int(str(self.cam.get_sensor_bit_depth()).split('_')[2])
        print(f'Depth: {self.depth} bit')
        self.maximum = int(2 ** self.depth - 1)

        self.bin_size = 4  # change to 2, 4, or 8
        self.delay = 1  # ms

        self.frame = np.zeros((self.height, self.width), dtype=np.uint16)
        self.rebinned = np.zeros((self.height // self.bin_size, self.width // self.bin_size), dtype=np.uint16)
        self.scaled = np.zeros((self.height // self.bin_size, self.width // self.bin_size), dtype=np.uint16)

        self.timer = QTimer()
        self.timer.timeout.connect(self.generate_frame)
        self.timer.setTimerType(Qt.TimerType.CoarseTimer)
        self.timer.setInterval(30)  # put 40 for ~20 FPS, put 30 for ~30 FPS
        self.timer.start()

    def generate_frame(self):
        # Simulate 8-bit grayscale image, 512x512
        # self.frame[:] = np.random.randint(0, 2**10, (2048, 2048), dtype=np.uint16)

        self.cam.get_image(self.img)
        self.frame[:] = self.img.get_image_data_numpy()
        rebin_numba(self.frame, self.rebinned, self.bin_size)
        self.scaled[:] = self.rebinned * 2 ** (16 - self.depth)

        self.new_frame.emit(self.scaled[:])
