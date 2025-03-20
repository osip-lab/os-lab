import os
import sys
import time
import logging
import pyperclip
import numpy as np
from scipy.optimize import curve_fit
from scipy.ndimage import center_of_mass
from numba import njit, float64
from pypylon import pylon

from PyQt6.QtWidgets import QApplication, QWidget, QCheckBox, QAbstractSpinBox, QComboBox
from PyQt6.QtCore import Qt, QThread, pyqtSignal, pyqtSlot, QTimer

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

from qt_gui.qt_ext import (MyStandardWindow, QMyVBoxLayout, QMyHBoxLayout, ThreadedWidget, ThreadedWorker,
                           QMyStandardButton, QMySpinBox)

from local_config import path_data_local


class BaslerCamControlWorker(ThreadedWorker):
    scanned = pyqtSignal(dict, name='Scanned')
    connected = pyqtSignal(dict, name='Connected')
    loaded = pyqtSignal(name='Loaded')
    captured = pyqtSignal(dict, name='Captured')

    def __init__(self, thread):
        super(BaslerCamControlWorker, self).__init__(thread)
        self.fac = None
        self.cam = None
        self.settings = dict()

    @pyqtSlot(name='Scan')
    def scan(self):
        self.fac = pylon.TlFactory.GetInstance()
        devices = self.fac.EnumerateDevices()
        info = {'serial_numbers': [d.GetSerialNumber() for d in devices]}
        self.finish(self.scanned, info)

    @pyqtSlot(dict, name='Connect')
    def connect(self, settings):
        devices = self.fac.EnumerateDevices()
        devices = list(filter(lambda d: d.GetSerialNumber() == settings['sn'], devices))
        self.cam = pylon.InstantCamera(self.fac.CreateDevice(devices[0]))
        self.cam.Open()
        info = {'model': self.cam.DeviceModelName(),
                'id': self.cam.DeviceVersion(),
                'sn': self.cam.DeviceSerialNumber()}

        # set exposure time
        self.cam.ExposureMode.SetValue('Timed')
        self.cam.ExposureAuto.SetValue('Off')
        # ExposureTimeMode parameter is not available
        self.cam.ExposureTime.SetValue(settings['exposure'])
        # set gain
        self.cam.GainSelector.SetValue('All')
        self.cam.GainAuto.SetValue('Off')
        self.cam.Gain.SetValue(settings['gain'])
        # set pixel depth
        self.cam.PixelFormat.SetValue('Mono12')

        self.finish(self.connected, info)

    @pyqtSlot(dict, name='load')
    def load(self, settings):
        self.cam.ExposureTime.SetValue(settings['exposure'])
        self.cam.Gain.SetValue(settings['gain'])
        self.finish(self.loaded)

    @pyqtSlot(name='Capture')
    def capture(self):
        result = self.cam.GrabOne(10000)  # timeout of 10 s
        img = result.Array
        self.finish(self.captured, {'image': img})


class BaslerCamControlWidget(ThreadedWidget):
    sig_scan = pyqtSignal(name='Scan')
    sig_connect = pyqtSignal(dict, name='Connect')
    sig_load = pyqtSignal(dict, name='Load')
    sig_capture = pyqtSignal(name='Capture')
    sig_captured = pyqtSignal(dict, name='Captured')

    def __init__(self, font_size=14):
        super(BaslerCamControlWidget, self).__init__(font_size=font_size)
        self.setTitle('Camera Control')

        self.settings = {'exposure': 100, 'gain': 0.0}

        self.btn_scan = QMyStandardButton('scan', font_size=self.font_size)
        self.btn_scan.setToolTip('scan for possible camera S/N')
        self.btn_scan.clicked.connect(self.scan)

        self.combobox_sn = QComboBox()
        self.combobox_sn.setToolTip('serial numbers of cameras connected to PC')
        self.combobox_sn.setMinimumContentsLength(8)

        self.btn_connect = QMyStandardButton('connect', font_size=self.font_size)
        self.btn_connect.setToolTip('connect to a device')
        self.btn_connect.clicked.connect(self.connect)

        self.spinbox_exposure = QMySpinBox(decimals=0, v_ini=self.settings['exposure'],
                                           v_min=28, v_max=10000000, suffix=' μs', step=10)
        self.spinbox_exposure.setToolTip('camera exposure time')

        self.spinbox_gain = QMySpinBox(decimals=1, v_ini=self.settings['gain'],
                                       v_min=0.0, v_max=23.0, suffix=' dB', step=1)
        self.spinbox_gain.setToolTip('camera gain')

        self.btn_load = QMyStandardButton('load', font_size=self.font_size)
        self.btn_load.setToolTip('load camera settings')
        self.btn_load.clicked.connect(self.load)

        self.btn_capture = QMyStandardButton('capture', font_size=self.font_size)
        self.btn_capture.setToolTip('capture a picture')
        self.btn_capture.clicked.connect(self.capture)

        self.auto_switch = QCheckBox('auto')
        self.auto_switch.setToolTip('auto capture after 1 s')
        self.auto_switch.setChecked(False)
        self.auto_timer = QTimer()
        self.auto_timer.setInterval(1000)
        self.auto_timer.setSingleShot(True)
        self.auto_timer.timeout.connect(self.timer_command)

        self.permission_timer = True
        self.permission_plotter = True

        self.worker = BaslerCamControlWorker(self.thread())
        self.worker_thread = None
        self.sig_scan.connect(self.worker.scan)
        self.worker.scanned.connect(self.scanned)
        self.sig_connect.connect(self.worker.connect)
        self.worker.connected.connect(self.connected)
        self.sig_load.connect(self.worker.load)
        self.sig_capture.connect(self.worker.capture)
        self.worker.captured.connect(self.captured)

        layout = QMyHBoxLayout(self.btn_scan, self.combobox_sn, self.btn_connect, self.spinbox_exposure,
                               self.spinbox_gain, self.btn_load, self.btn_capture, self.auto_switch)
        self.setLayout(layout)

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

    def get_settings(self):
        self.settings['sn'] = self.combobox_sn.currentText()
        self.settings['exposure'] = int(self.spinbox_exposure.value())
        self.settings['gain'] = int(self.spinbox_gain.value())
        return self.settings

    @pyqtSlot(name='Connect')
    def connect(self):
        self.worker_thread = QThread()
        self.start_branch(self.worker, self.worker_thread, self.sig_connect, self.get_settings())

    @pyqtSlot(dict, name='Connected')
    def connected(self, info):
        log_msg = f"connected to camera - model {info['model']}, id {info['id']}, s/n {info['sn']}"
        logging.info(log_msg)

    @pyqtSlot(name='Load')
    def load(self):
        self.worker_thread = QThread()
        self.start_branch(self.worker, self.worker_thread, self.sig_load, self.get_settings())

    @pyqtSlot(name='Capture')
    def capture(self):
        if self.permission_timer & self.permission_plotter:
            if self.auto_switch.isChecked():
                self.permission_timer = False
            self.permission_plotter = False
            self.worker_thread = QThread()
            self.start_branch(self.worker, self.worker_thread, self.sig_capture)

    @pyqtSlot(name='TimerCommand')
    def timer_command(self):
        self.permission_timer = True
        self.capture()

    @pyqtSlot(name='PlotterCommand')
    def plotter_command(self):
        self.permission_plotter = True
        if self.auto_switch.isChecked():
            self.capture()

    @pyqtSlot(dict, name='Captured')
    def captured(self, data):
        self.sig_captured.emit(data)
        if self.auto_switch.isChecked():
            self.auto_timer.start()


class MplCanvas(FigureCanvas):

    def __init__(self, width=6, height=6, dpi=100):

        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax = self.fig.add_subplot(111)
        div = make_axes_locatable(self.ax)
        self.hax = div.append_axes('top', size='20%', pad=0.2)
        self.hax.sharex(self.ax)
        self.hax.tick_params(bottom=False, top=True, labelbottom=False, labeltop=True)
        self.vax = div.append_axes('right', size='20%', pad=0.2)
        self.vax.sharey(self.ax)
        self.vax.tick_params(left=False, right=True, labelleft=False, labelright=True)
        self.hax.set_ylim(0, 4095)
        self.vax.set_xlim(0, 4095)

        x = np.linspace(0, 2047, 2048)
        y = np.linspace(0, 2047, 2048)
        self.xx, self.yy = np.meshgrid(x, y)
        gaussian = 256 + 1024 * np.exp(-((self.xx - 1024)**2 + (self.yy - 512)**2) / (2 * 256**2))
        image = gaussian + np.random.randn(*gaussian.shape) * 32

        self.iax = self.ax.imshow(image, cmap='gray', vmin=0, vmax=4095, origin='lower')
        self.cax = self.ax.contour(self.xx, self.yy, gaussian, 8, colors='r')
        self.hsi = self.hax.plot(np.arange(2048), image[1024, :])[0]
        self.hsg = self.hax.plot(np.arange(2048), gaussian[1024, :])[0]
        self.vsi = self.vax.plot(image[:, 1024], np.arange(2048))[0]
        self.vsg = self.vax.plot(gaussian[:, 1024], np.arange(2048))[0]

        self.fig.tight_layout()

        super(MplCanvas, self).__init__(self.fig)


class MatplotlibWidget(QWidget):
    sig_plotted = pyqtSignal(name='Plotted')

    def __init__(self, parent=None):
        super(MatplotlibWidget, self).__init__(parent)
        self.canvas = MplCanvas()

        layout = QMyVBoxLayout(self.canvas)
        self.setLayout(layout)

    def plot(self, data):
        self.canvas.iax.set_data(data['image'])
        if 'parameters' in data.keys():
            x0, y0 = int(data['parameters']['x_0']), int(data['parameters']['y_0'])
        else:
            x0, y0 = 1024, 1024
        self.canvas.hsi.set_data(np.arange(2048), data['image'][y0, :])
        self.canvas.vsi.set_data(data['image'][:, x0], np.arange(2048))
        if 'parameters' in data.keys():
            self.canvas.cax.remove()
            self.canvas.cax = self.canvas.ax.contour(self.canvas.xx, self.canvas.yy, data['gaussian'], 8, colors='r')
            self.canvas.hsg.set_data(np.arange(2048), data['gaussian'][y0, :])
            self.canvas.vsg.set_data(data['gaussian'][:, x0], np.arange(2048))
        self.canvas.draw()
        self.sig_plotted.emit()


@njit(float64[:](float64[:, :, ::0], float64, float64, float64, float64, float64, float64, float64),
      locals={'x': float64[:, ::0], 'y': float64[:, ::0], 'g': float64[:, ::1],
              'a': float64, 'b': float64, 'c': float64})
def gaussian2d(xy, ampl, xo, yo, sigma_x, sigma_y, theta, offset):
    x = xy[0]
    y = xy[1]
    a = np.cos(theta)**2 / (2 * sigma_x**2) + np.sin(theta)**2 / (2 * sigma_y**2)
    b = -np.sin(2 * theta) / (4 * sigma_x**2) + np.sin(2 * theta) / (4 * sigma_y**2)
    c = np.sin(theta)**2 / (2 * sigma_x**2) + np.cos(theta)**2 / (2 * sigma_y**2)
    g = offset + ampl * np.exp(-(a * (x - xo)**2 + 2 * b * (x - xo) * (y - yo) + c * (y - yo)**2))
    return np.ravel(g)


def rebin(a, shape):
    sh = shape[0], a.shape[0]//shape[0], shape[1], a.shape[1]//shape[1]
    return a.reshape(sh).mean(-1).mean(1)


def fit_gaussian(arr, rebinning=1):

    sy0, sx0 = np.shape(arr)
    arr = rebin(arr, (int(sy0 / rebinning), int(sx0 / rebinning)))
    sy, sx = np.shape(arr)

    xx = np.linspace(0, sx - 1, sx)
    yy = np.linspace(0, sy - 1, sy)
    xx, yy = np.meshgrid(xx, yy)

    background = np.percentile(arr, 15)
    mh = arr > np.percentile(arr - background, (1 - 100 / sx0 / sy0) * 100)
    amplitude = np.mean(arr[mh]) - background
    y0, x0 = center_of_mass(np.array(mh, dtype=np.float64))
    mc = arr > amplitude / np.e**0.5
    radius = max((np.sum(mc) / np.pi) ** 0.5, 1)
    initial_guess = (amplitude, x0, y0, radius, radius, 0.0, background)
    tic = time.time()
    try:
        p = curve_fit(gaussian2d, np.array((xx, yy)), arr.ravel(), p0=initial_guess, full_output=True,
                      bounds=((0.0, 0.0, 0.0, 0.0, 0.0, -np.pi / 4, 0.0), (4095, sx, sy, np.inf, np.inf, np.pi / 4, 4095)),
                      ftol=1e-3, xtol=1e-3)
        # p = curve_fit(gaussian2d, np.array((xx, yy)), arr.ravel(), p0=initial_guess, full_output=True)
    except RuntimeError:
        p = (initial_guess, np.zeros_like(initial_guess), 'fitting unsuccessful')
    dt = time.time() - tic

    pars = p[0]
    pars = (pars[0], pars[1] * rebinning, pars[2] * rebinning, pars[3] * rebinning, pars[4] * rebinning,
            pars[5], pars[6])

    xx = np.linspace(0, sx0 - 1, sx0)
    yy = np.linspace(0, sy0 - 1, sy0)
    xx, yy = np.meshgrid(xx, yy)

    gauss = np.reshape(gaussian2d(np.array((xx, yy)), *pars), (sy0, sx0))
    # gauss = zoom(gauss, rebinning, order=0)
    pars = {'amplitude': pars[0], 'offset': pars[6], 'angle': pars[5], 'time': dt,
            'x_0': pars[1], 'y_0': pars[2], 's_x': pars[3], 's_y': pars[4],
            'w_x': pars[3] * 2**0.5, 'w_y': pars[4] * 2**0.5}

    return gauss, pars


class GaussianFitterWorker(ThreadedWorker):
    fitted = pyqtSignal(dict, name='Fitted')

    def __init__(self, thread):
        super(GaussianFitterWorker, self).__init__(thread)

    @pyqtSlot(dict, name='Fit')
    def fit(self, data):
        fit, pars = fit_gaussian(data['image'], rebinning=4)
        data['gaussian'] = fit
        data['parameters'] = pars
        self.finish(self.fitted, data)


class GaussianFitterWidget(ThreadedWidget):
    sig_fit = pyqtSignal(dict, name='Fit')
    sig_fitted = pyqtSignal(dict, name='Fitted')

    def __init__(self, font_size=14):
        super(GaussianFitterWidget, self).__init__(font_size=font_size)
        self.setTitle('Gaussian Fitter')

        self.parameters = dict()

        self.fit_switch = QCheckBox('fit')
        self.fit_switch.setToolTip('fit Gaussian to image')
        self.fit_switch.setChecked(False)

        self.spinbox_threshold = QMySpinBox(decimals=0, v_ini=1000, v_min=0, v_max=9999, suffix='', step=100)

        self.labels = ('x_0', 'y_0', 'w_x', 'w_y')
        self.spinboxes = dict()
        for lbl in self.labels:
            spinbox = QMySpinBox(decimals=1, v_min=0.0, v_max=9999.0, prefix=f'{lbl}: ', suffix=' pxl')
            spinbox.setReadOnly(True)
            spinbox.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
            self.spinboxes[lbl] = spinbox

        self.btn_copy = QMyStandardButton('copy', font_size=self.font_size)
        self.btn_copy.setToolTip('copy last parameters to clipboard')
        self.btn_copy.clicked.connect(self.copy)

        self.worker = GaussianFitterWorker(self.thread())
        self.worker_thread = None
        self.sig_fit.connect(self.worker.fit)
        self.worker.fitted.connect(self.fitted)

        layout = QMyHBoxLayout(self.fit_switch, self.spinbox_threshold, *[self.spinboxes[lbl] for lbl in self.labels], self.btn_copy)
        self.setLayout(layout)

    @pyqtSlot(dict, name='Fit')
    def fit(self, data):
        if self.fit_switch.isChecked():
            if np.max(data['image']) > self.spinbox_threshold.value():
                self.worker_thread = QThread()
                self.start_branch(self.worker, self.worker_thread, self.sig_fit, data)
            else:
                self.sig_fitted.emit(data)
        else:
            self.sig_fitted.emit(data)

    def log_msg(self):
        pars = self.parameters
        log_msg = (f"gaussian parameters - x_0 = {pars['x_0']:06.1f} pxl, y_0 = {pars['y_0']:06.1f} pxl, "
                   f"w_x = {pars['w_x']:06.1f} pxl, w_y = {pars['w_y']:06.1f} pxl, angle = {pars['angle']:+01.2f} rad, "
                   f"ampl = {pars['amplitude']:06.1f}, offset = {pars['offset']:06.1f}, time = {pars['time']:05.2f} s")
        return log_msg

    def show_parameters(self):
        for lbl in self.labels:
            self.spinboxes[lbl].setValue(self.parameters[lbl])

    @pyqtSlot(dict, name='Fitted')
    def fitted(self, data):
        self.parameters = data['parameters']
        self.show_parameters()
        logging.info(self.log_msg())
        self.sig_fitted.emit(data)

    @pyqtSlot(name='Copy')
    def copy(self):
        log_msg = f"{time.strftime('%Y.%m.%d %H:%M:%S')}: {self.log_msg()}"
        pyperclip.copy(log_msg)


class MainWidget(QWidget):

    def __init__(self):
        super(MainWidget, self).__init__()
        self.font_size = 14

        self.controller = BaslerCamControlWidget(font_size=self.font_size)
        self.fitter = GaussianFitterWidget(font_size=self.font_size)
        self.plotter = MatplotlibWidget()

        self.controller.sig_captured.connect(self.fitter.fit)
        self.fitter.sig_fitted.connect(self.plotter.plot)
        self.plotter.sig_plotted.connect(self.controller.plotter_command)

        layout = QMyVBoxLayout(self.controller, self.fitter, self.plotter, alignment=Qt.AlignmentFlag.AlignCenter)
        self.setLayout(layout)


class MainWindow(MyStandardWindow):

    def __init__(self):
        super().__init__()
        font = self.font()
        font.setPointSize(14)
        self.widget = MainWidget()
        self.widget.setFont(font)
        self.widget.font_size = 14
        self.appear_with_central_widget('Mode Position', self.widget)


if __name__ == '__main__':

    start_time = time.strftime('%Y-%m-%d %H-%M-%S')
    logging.basicConfig(filename=os.path.join(path_data_local, 'mode_position', f'{start_time} log.txt'),
                        level=logging.INFO, format='%(asctime)s.%(msecs)03d: %(message)s', datefmt='%Y.%m.%d %H:%M:%S')

    app = QApplication(sys.argv)
    ex = MainWindow()
    sys.exit(app.exec())
