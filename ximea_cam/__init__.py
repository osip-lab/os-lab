"""
Subpackage description.
"""
import numpy as np
from numba import njit
from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot, QThread, QTimer
from qt_gui.qt_ext import ThreadedWidget, ThreadedWorker, QMyHBoxLayout, QMyStandardButton

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


class XimeaCamControlWorker(ThreadedWorker):
    scanned = pyqtSignal(dict, name='Scanned')
    connected = pyqtSignal(name='Connected')
    captured = pyqtSignal(dict, name='Captured')

    def __init__(self, thread):
        super(XimeaCamControlWorker, self).__init__(thread)

        self.cam = None
        self.img = None

        self.width = None
        self.height = None
        self.depth = None
        self.maximum = None
        self.bin_size = None
        self.delay = None
        self.frame = None
        self.rebinned = None
        self.scaled = None

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.grab)
        self.timer.setTimerType(Qt.TimerType.CoarseTimer)
        self.timer.setInterval(5)
        self.timer.setSingleShot(True)

    @pyqtSlot(name='Scan')
    def scan(self):
        pass

    @pyqtSlot(name='Connect')
    def connect(self):
        # create instance for first connected camera
        self.cam = xiapi.Camera()
        print('Opening the first camera...')
        self.cam.open_device()

        # settings
        self.cam.set_exposure(1000)
        self.cam.set_acq_timing_mode('XI_ACQ_TIMING_MODE_FRAME_RATE')
        self.cam.set_framerate(10)

        # mode_used = self.cam.get_acq_timing_mode()
        # if mode_used == 'XI_ACQ_TIMING_MODE_FRAME_RATE':
        #     print('Mode is XI_ACQ_TIMING_MODE_FRAME_RATE')
        # else:
        #     print('Mode is not XI_ACQ_TIMING_MODE_FRAME_RATE')
        print(f'Frame rate: {self.cam.get_framerate()}')
        print('Exposure was set to %i us' % self.cam.get_exposure())

        # print('The maximal width of this camera is %i.' % self.cam.get_width_maximum())
        # print('The minimal width of this camera is %i.' % self.cam.get_width_minimum())
        # print('The increment of the width of this camera is %i.' % self.cam.get_width_increment())

        self.cam.set_imgdataformat('XI_MONO16')  # need it to have 10-bit image depth

        # create instance of Image to store image data and metadata
        self.img = xiapi.Image()

        # Image size and depth
        self.width, self.height = self.cam.get_width(), self.cam.get_height()
        print(f'Height: {self.height}, Width: {self.width}')
        # print(self.cam.get_sensor_bit_depth())
        self.depth = int(str(self.cam.get_sensor_bit_depth()).split('_')[2])
        print(f'Depth: {self.depth} bit')
        # self.maximum = int(2 ** self.depth - 1)

        self.bin_size = 4  # change to 2, 4, or 8
        # self.delay = 1  # ms

        self.frame = np.zeros((self.height, self.width), dtype=np.uint16)
        self.rebinned = np.zeros((self.height // self.bin_size, self.width // self.bin_size), dtype=np.uint16)
        self.scaled = np.zeros((self.height // self.bin_size, self.width // self.bin_size), dtype=np.uint16)

        print('Camera is ready.')
        self.connected.emit()

    @pyqtSlot(name='Start')
    def start(self):
        # start data acquisition
        print('Starting data acquisition...')
        self.cam.start_acquisition()
        self.timer.start()

    @pyqtSlot(name='Grab')
    def grab(self):
        self.cam.get_image(self.img)
        self.frame[:] = self.img.get_image_data_numpy()
        rebin_numba(self.frame, self.rebinned, self.bin_size)
        self.scaled[:] = self.rebinned * 2 ** (16 - self.depth)
        self.captured.emit({'image': self.scaled})
        self.timer.start()

    @pyqtSlot(name='Stop')
    def stop(self):
        self.timer.stop()
        print('Stopping acquisition...')
        self.cam.stop_acquisition()


# ---------------- Controller ---------------- #
class XimeaCamControlWidget(ThreadedWidget):
    new_frame = pyqtSignal(np.ndarray)
    sig_connect = pyqtSignal(name='Connect')
    sig_start = pyqtSignal(name='Start')
    sig_stop = pyqtSignal(name='Stop')

    def __init__(self, font_size=14):
        super(XimeaCamControlWidget, self).__init__(font_size=font_size)
        self.setTitle('Camera Control')

        self.btn_connect = QMyStandardButton('connect', font_size=self.font_size)
        self.btn_connect.setToolTip('connect')
        self.btn_connect.clicked.connect(self.connect)

        self.btn_start = QMyStandardButton('start', font_size=self.font_size)
        self.btn_start.setToolTip('start')
        self.btn_start.clicked.connect(self.start)

        self.btn_stop = QMyStandardButton('stop', font_size=self.font_size)
        self.btn_stop.setToolTip('stop')
        self.btn_stop.clicked.connect(self.stop)

        self.worker = XimeaCamControlWorker(self.thread())
        self.worker_thread = None
        self.sig_connect.connect(self.worker.connect)
        self.sig_start.connect(self.worker.start)
        self.sig_stop.connect(self.worker.stop)
        self.worker.captured.connect(self.generate_frame)

        layout = QMyHBoxLayout(self.btn_connect, self.btn_start, self.btn_stop)
        self.setLayout(layout)

    @pyqtSlot(name='Connect')
    def connect(self):
        self.worker_thread = QThread()
        self.start_branch(self.worker, self.worker_thread, self.sig_connect)

    @pyqtSlot(name='Start')
    def start(self):
        self.sig_start.emit()

    @pyqtSlot(name='Stop')
    def stop(self):
        self.sig_stop.emit()

    @pyqtSlot(dict, name='Generate')
    def generate_frame(self, data):
        self.new_frame.emit(data['image'])
