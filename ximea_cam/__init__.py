"""
Subpackage description.
"""
import numpy as np
from numba import njit
from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot, QThread, QTimer
from PyQt6.QtWidgets import QComboBox
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
    closed = pyqtSignal(name='Closed')

    def __init__(self, thread):
        super(XimeaCamControlWorker, self).__init__(thread)

        self.cam = None
        self.img = None

        self.width = None
        self.height = None
        self.depth = None
        self.bin_size = None
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
        self.cam = xiapi.Camera()
        n = self.cam.get_number_devices()
        serials = []
        for i in range(n):
            self.cam = xiapi.Camera(dev_id=i)
            self.cam.open_device()
            sn = self.cam.get_device_sn(buffer_size=256)
            sn = sn.decode('ascii')
            serials.append(sn)
            self.cam.close_device()
        self.cam = None
        info = {'serial_numbers': serials}
        self.finish(self.scanned, info)

    @pyqtSlot(dict, name='Open')
    def open(self, info):
        self.cam = xiapi.Camera()
        print(f"Opening camera with SN {info['sn']}")
        self.cam.open_device_by_SN(info['sn'])

        # settings
        self.cam.set_exposure(1000)
        self.cam.set_acq_timing_mode('XI_ACQ_TIMING_MODE_FRAME_RATE')
        self.cam.set_framerate(10)

        print(f'Frame rate: {self.cam.get_framerate()}')
        print('Exposure was set to %i us' % self.cam.get_exposure())

        self.cam.set_imgdataformat('XI_MONO16')  # need it to have 10-bit image depth

        # create instance of Image to store image data and metadata
        self.img = xiapi.Image()

        # Image size and depth
        self.width, self.height = self.cam.get_width(), self.cam.get_height()
        print(f'Height: {self.height}, Width: {self.width}')
        self.depth = int(str(self.cam.get_sensor_bit_depth()).split('_')[2])
        print(f'Depth: {self.depth} bit')

        self.bin_size = 4  # change to 2, 4, or 8

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

    @pyqtSlot(name='Close')
    def close(self):
        self.cam.close_device()
        self.cam = None
        print('Camera closed')
        self.finish(self.closed)


# ---------------- Controller ---------------- #
class XimeaCamControlWidget(ThreadedWidget):
    new_frame = pyqtSignal(np.ndarray)
    sig_scan = pyqtSignal(name='Scan')
    sig_open = pyqtSignal(dict, name='Open')
    sig_start = pyqtSignal(name='Start')
    sig_stop = pyqtSignal(name='Stop')
    sig_close = pyqtSignal(name='Close')

    def __init__(self, font_size=14):
        super(XimeaCamControlWidget, self).__init__(font_size=font_size)
        self.setTitle('Camera Control')

        self.settings = {'exposure': 100, 'gain': 0.0}

        self.btn_scan = QMyStandardButton('scan', font_size=self.font_size)
        self.btn_scan.setToolTip('scan for possible camera S/N')
        self.btn_scan.clicked.connect(self.scan)

        self.combobox_sn = QComboBox()
        self.combobox_sn.setToolTip('serial numbers of cameras connected to PC')
        self.combobox_sn.setMinimumContentsLength(12)

        self.btn_open = QMyStandardButton('open', font_size=self.font_size)
        self.btn_open.setToolTip('open')
        self.btn_open.clicked.connect(self.open_cam)

        self.btn_start = QMyStandardButton('start', font_size=self.font_size)
        self.btn_start.setToolTip('start')
        self.btn_start.clicked.connect(self.start_cam)

        self.btn_stop = QMyStandardButton('stop', font_size=self.font_size)
        self.btn_stop.setToolTip('stop')
        self.btn_stop.clicked.connect(self.stop_cam)

        self.btn_close = QMyStandardButton('close', font_size=self.font_size)
        self.btn_close.setToolTip('close')
        self.btn_close.clicked.connect(self.close_cam)

        self.worker = XimeaCamControlWorker(self.thread())
        self.worker_thread = None
        self.sig_scan.connect(self.worker.scan)
        self.sig_open.connect(self.worker.open)
        self.sig_start.connect(self.worker.start)
        self.sig_stop.connect(self.worker.stop)
        self.sig_close.connect(self.worker.close)
        self.worker.scanned.connect(self.scanned)
        self.worker.captured.connect(self.generate_frame)

        layout = QMyHBoxLayout(self.btn_scan, self.combobox_sn, self.btn_open, self.btn_start, self.btn_stop, self.btn_close)
        self.setLayout(layout)

    def get_settings(self):
        self.settings['sn'] = self.combobox_sn.currentText()
        # self.settings['exposure'] = int(self.spinbox_exposure.value())
        # self.settings['gain'] = int(self.spinbox_gain.value())
        return self.settings

    @pyqtSlot(name='Scan')
    def scan(self):
        self.worker_thread = QThread()
        self.start_branch(self.worker, self.worker_thread, self.sig_scan)

    @pyqtSlot(dict, name='Scanned')
    def scanned(self, info):
        items = [self.combobox_sn.itemText(i) for i in range(self.combobox_sn.count())]
        for sn in info['serial_numbers']:
            if sn not in items:
                self.combobox_sn.addItem(sn)

    @pyqtSlot(name='Open')
    def open_cam(self):
        self.worker_thread = QThread()
        self.start_branch(self.worker, self.worker_thread, self.sig_open, self.get_settings())

    @pyqtSlot(name='Start')
    def start_cam(self):
        self.sig_start.emit()

    @pyqtSlot(name='Stop')
    def stop_cam(self):
        self.sig_stop.emit()

    @pyqtSlot(name='Close')
    def close_cam(self):
        self.sig_close.emit()

    @pyqtSlot(dict, name='Generate')
    def generate_frame(self, data):
        self.new_frame.emit(data['image'])
