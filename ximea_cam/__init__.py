"""
Subpackage description.
"""
import json
import numpy as np
from pathlib import Path
from numba import njit
from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot, QThread, QTimer
from PyQt6.QtWidgets import QComboBox, QVBoxLayout
from qt_gui.qt_ext import ThreadedWidget, ThreadedWorker, QMyHBoxLayout, QMyStandardButton, QMySpinBox

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

        self.last_settings = None

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.grab)
        self.timer.setTimerType(Qt.TimerType.CoarseTimer)
        self.timer.setInterval(3)
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
        print('Scan finished')
        self.finish(self.scanned, info)

    def prepare_buffers(self):
        self.frame = np.zeros((self.height, self.width), dtype=np.uint16)
        self.rebinned = np.zeros((self.height // self.bin_size, self.width // self.bin_size), dtype=np.uint16)
        self.scaled = np.zeros((self.height // self.bin_size, self.width // self.bin_size), dtype=np.uint16)

    @pyqtSlot(dict, name='Open')
    def open(self, settings):
        self.cam = xiapi.Camera()
        print(f"Opening camera with SN {settings['sn']}")
        self.cam.open_device_by_SN(settings['sn'])

        # settings
        self.last_settings = settings.copy()
        self.cam.set_exposure(settings['exposure'])
        self.cam.set_gain(settings['gain'])
        self.cam.set_acq_timing_mode('XI_ACQ_TIMING_MODE_FRAME_RATE')
        self.cam.set_framerate(settings['fps'])
        self.bin_size = settings['bin_size']

        print(f'Frame rate: {self.cam.get_framerate()}')
        print(f'Exposure was set to {int(self.cam.get_exposure()):d} us')
        print(f'Gain was set to {self.cam.get_gain():.2f} dB')

        self.set_depth(settings['depth'])

        # create instance of Image to store image data and metadata
        self.img = xiapi.Image()

        # Image size and depth
        self.width, self.height = self.cam.get_width(), self.cam.get_height()
        print(f'Height: {self.height}, Width: {self.width}')
        # self.depth = int(str(self.cam.get_sensor_bit_depth()).split('_')[2])
        self.depth = settings['depth']
        print(f'Depth: {self.depth} bit')

        self.prepare_buffers()

        print('Camera is ready.')
        self.connected.emit()

    def set_depth(self, d):
        if d == 8:
            self.cam.set_imgdataformat('XI_MONO8')  # gives 8-bit image, need for 90 FPS
        elif d == 10:
            self.cam.set_imgdataformat('XI_MONO16')  # need it to have 10-bit image depth
        else:
            raise ValueError('The supported depths are 8 or 10')

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
        self.captured.emit({'image': self.scaled, 'raw': self.frame})
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

    @pyqtSlot(dict, name='UpdateSettings')
    def update_settings(self, settings):

        if settings['exposure'] != self.last_settings['exposure']:
            self.cam.set_exposure(settings['exposure'])

        if settings['gain'] != self.last_settings['gain']:
            self.cam.set_gain(settings['gain'])

        if settings['fps'] != self.last_settings['fps']:
            self.cam.set_framerate(settings['fps'])

        if settings['bin_size'] != self.last_settings['bin_size']:
            self.bin_size = settings['bin_size']
            self.prepare_buffers()

        if settings['depth'] != self.last_settings['depth']:
            self.depth = settings['depth']
            self.set_depth(self.depth)

        self.last_settings = settings.copy()


# ---------------- Controller ---------------- #
class XimeaCamControlWidget(ThreadedWidget):
    new_frame = pyqtSignal(np.ndarray)
    sig_scan = pyqtSignal(name='Scan')
    sig_open = pyqtSignal(dict, name='Open')
    sig_start = pyqtSignal(name='Start')
    sig_stop = pyqtSignal(name='Stop')
    sig_close = pyqtSignal(name='Close')
    sig_settings_changed = pyqtSignal(dict, name='SettingsChanged')

    def __init__(self, font_size=14):
        super(XimeaCamControlWidget, self).__init__(font_size=font_size)
        self.setTitle('Camera Control')

        self.settings = {'sn': None, 'exposure': 1000, 'gain': 0.0, 'fps': 30, 'bin_size': 4, 'depth': 8}

        project_dir = Path(__file__).resolve().parents[1]
        template_path = project_dir / 'settings' / 'templates' / 'ximea_viewer.json'
        local_settings_path = project_dir / 'settings' / 'local' / 'ximea_viewer.json'

        template_settings = {}
        local_settings = {}

        if template_path.exists():
            with open(template_path, 'r') as f:
                template_settings = json.load(f)

        if local_settings_path.exists():
            with open(local_settings_path, 'r') as f:
                local_settings = json.load(f)

        for key, default_value in self.settings.items():
            if key in local_settings:
                self.settings[key] = local_settings[key]
            elif key in template_settings:
                self.settings[key] = template_settings[key]
                print(f"'{key}' missing in local settings, using template value")
            else:
                print(f"'{key}' missing in both local and template settings, using hardcoded value")

        if local_settings_path.exists():
            print('Loaded local settings')
        elif template_path.exists():
            print('Local setting do not exist, loaded template settings')
        else:
            print('Both local and template settings do not exist, using hardcoded settings')

        self.btn_scan = QMyStandardButton('scan', font_size=self.font_size)
        self.btn_scan.setToolTip('scan for possible camera S/N')
        self.btn_scan.clicked.connect(self.scan)

        self.combobox_sn = QComboBox()
        self.combobox_sn.setToolTip('serial numbers of cameras connected to PC')
        self.combobox_sn.setMinimumContentsLength(12)
        if self.settings['sn'] is not None:
            self.combobox_sn.addItem(self.settings['sn'])

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

        self.settings_changed_flag = False

        self.spinbox_exposure = QMySpinBox(decimals=0, v_ini=self.settings['exposure'],
                                           v_min=26, v_max=1000000, suffix=' μs', step=10)
        self.spinbox_exposure.setToolTip('camera exposure time in μs')
        self.spinbox_exposure.adjust_width()
        self.spinbox_exposure.valueChanged.connect(self.settings_changed)

        self.spinbox_gain = QMySpinBox(decimals=2, v_ini=self.settings['gain'],
                                       v_min=-1.5, v_max=6.0, suffix=' dB', step=1)
        self.spinbox_gain.setToolTip('camera gain time in dB (20 dB - factor of 10)')
        self.spinbox_gain.adjust_width()
        self.spinbox_gain.valueChanged.connect(self.settings_changed)

        self.spinbox_fps = QMySpinBox(decimals=0, v_ini=self.settings['fps'],
                                       v_min=1, v_max=90, suffix=' FPS', step=1)
        self.spinbox_fps.setToolTip('camera FPS, up to 45 for 10-bit image')
        self.spinbox_fps.adjust_width()
        self.spinbox_fps.valueChanged.connect(self.settings_changed)

        self.combobox_bin_size = QComboBox()
        self.combobox_bin_size.setToolTip('bin size for rescaling the image')
        self.combobox_bin_size.setMinimumContentsLength(2)
        self.combobox_bin_size.addItems(['1', '2', '4', '8'])
        self.combobox_bin_size.setCurrentText(str(self.settings['bin_size']))
        self.combobox_bin_size.currentTextChanged.connect(self.settings_changed)

        self.combobox_depth = QComboBox()
        self.combobox_depth.setToolTip('bin size for rescaling the image')
        self.combobox_depth.setMinimumContentsLength(2)
        self.combobox_depth.addItems(['8', '10'])
        self.combobox_depth.setCurrentText(str(self.settings['depth']))
        self.combobox_depth.currentTextChanged.connect(self.settings_changed)

        self.worker = XimeaCamControlWorker(self.thread())
        self.worker_thread = None
        self.sig_scan.connect(self.worker.scan)
        self.sig_open.connect(self.worker.open)
        self.sig_start.connect(self.worker.start)
        self.sig_stop.connect(self.worker.stop)
        self.sig_close.connect(self.worker.close)
        self.sig_settings_changed.connect(self.worker.update_settings)
        self.worker.scanned.connect(self.scanned)
        self.worker.captured.connect(self.generate_frame)

        layout = QVBoxLayout()
        layout.addLayout(QMyHBoxLayout(self.btn_scan, self.combobox_sn, self.btn_open, self.btn_start, self.btn_stop, self.btn_close))
        lt = QMyHBoxLayout(self.spinbox_exposure, self.spinbox_gain, self.spinbox_fps, self.combobox_bin_size, self.combobox_depth)
        lt.addStretch()
        layout.addLayout(lt)
        self.setLayout(layout)

    def get_settings(self):
        self.settings['sn'] = self.combobox_sn.currentText()
        self.settings['exposure'] = int(self.spinbox_exposure.value())
        self.settings['gain'] = float(self.spinbox_gain.value())
        self.settings['fps'] = int(self.spinbox_fps.value())
        self.settings['bin_size'] = int(self.combobox_bin_size.currentText())
        self.settings['depth'] = int(self.combobox_depth.currentText())
        return self.settings

    @pyqtSlot()
    def settings_changed(self):
        self.get_settings()
        self.settings_changed_flag = True

    @pyqtSlot(name='Scan')
    def scan(self):
        self.worker_thread = QThread()
        self.start_branch(self.worker, self.worker_thread, self.sig_scan)

    @pyqtSlot(dict, name='Scanned')
    def scanned(self, info):
        serial_numbers = info['serial_numbers']
        current_sn = self.combobox_sn.currentText()
        self.combobox_sn.clear()
        self.combobox_sn.addItems(serial_numbers)
        if current_sn in serial_numbers:
            self.combobox_sn.setCurrentText(current_sn)

    @pyqtSlot(name='Open')
    def open_cam(self):
        self.worker_thread = QThread()
        self.start_branch(self.worker, self.worker_thread, self.sig_open, self.get_settings().copy())

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

        if self.settings_changed_flag:
            self.sig_settings_changed.emit(self.get_settings().copy())
            self.settings_changed_flag = False
