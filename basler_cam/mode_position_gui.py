import sys
import time
import numpy as np
from scipy.optimize import curve_fit
from scipy.ndimage import center_of_mass
from numba import njit, float64
from pypylon import pylon

from PyQt6.QtWidgets import QApplication, QWidget, QCheckBox
from PyQt6.QtCore import Qt, QThread, pyqtSignal, pyqtSlot, QTimer

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

from qt_gui.qt_ext import (MyStandardWindow, QMyVBoxLayout, QMyHBoxLayout, ThreadedWidget, ThreadedWorker,
                           QMyStandardButton)


class BaslerCamControlWorker(ThreadedWorker):
    connected = pyqtSignal(name='Connected')
    captured = pyqtSignal(dict, name='Captured')

    def __init__(self, thread):
        super(BaslerCamControlWorker, self).__init__(thread)
        self.cam = None
        self.settings = dict()

    @pyqtSlot(name='Connect')
    def connect(self):
        self.cam = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
        self.cam.Open()
        print(f'Model: {self.cam.DeviceModelName()}')
        print(f'ID: {self.cam.DeviceVersion()}')
        print(f'S/N: {self.cam.DeviceSerialNumber()}')

        # set exposure time
        self.cam.ExposureMode.SetValue('Timed')
        self.cam.ExposureAuto.SetValue('Off')
        # ExposureTimeMode parameter is not available
        self.cam.ExposureTime.SetValue(100)
        # set gain
        self.cam.GainSelector.SetValue('All')
        self.cam.GainAuto.SetValue('Off')
        self.cam.Gain.SetValue(0.0)
        # set pixel depth
        self.cam.PixelFormat.SetValue('Mono12')

        self.finish(self.connected)

    @pyqtSlot(name='Capture')
    def capture(self):
        result = self.cam.GrabOne(100)
        img = result.Array
        self.finish(self.captured, {'image': img})


class BaslerCamControlWidget(ThreadedWidget):
    sig_connect = pyqtSignal(name='Connect')
    sig_capture = pyqtSignal(name='Capture')
    sig_captured = pyqtSignal(dict, name='Captured')

    def __init__(self, font_size=14):
        super(BaslerCamControlWidget, self).__init__(font_size=font_size)
        self.setTitle('Camera Control')

        self.btn_connect = QMyStandardButton('connect', font_size=self.font_size)
        self.btn_connect.setToolTip('connect to a device')
        self.btn_connect.clicked.connect(self.connect)

        self.btn_capture = QMyStandardButton('capture', font_size=self.font_size)
        self.btn_capture.setToolTip('capture a picture')
        self.btn_capture.clicked.connect(self.capture)

        self.auto_switch = QCheckBox('auto')
        self.auto_switch.setChecked(False)
        self.auto_timer = QTimer()
        self.auto_timer.setInterval(1000)
        self.auto_timer.setSingleShot(True)
        self.auto_timer.timeout.connect(self.timer_command)

        self.permission_timer = True
        self.permission_plotter = True

        self.worker = BaslerCamControlWorker(self.thread())
        self.worker_thread = None
        self.sig_connect.connect(self.worker.connect)
        self.sig_capture.connect(self.worker.capture)
        self.worker.captured.connect(self.captured)

        layout = QMyHBoxLayout(self.btn_connect, self.btn_capture, self.auto_switch)
        self.setLayout(layout)

    @pyqtSlot(name='Connect')
    def connect(self):
        self.worker_thread = QThread()
        self.start_branch(self.worker, self.worker_thread, self.sig_connect)

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
    mh = arr > np.percentile(arr - background, 98)
    amplitude = np.mean(arr[mh]) - background
    y0, x0 = center_of_mass(np.array(mh, dtype=np.float64))
    mc = arr <= np.percentile(arr - background, 45)
    radius = max((np.sum(mc) / np.pi) ** 0.5, sx / 128)
    initial_guess = (amplitude, x0, y0, radius, radius, 0.0, background)
    # print(initial_guess)
    tic = time.time()
    try:
        p = curve_fit(gaussian2d, np.array((xx, yy)), arr.ravel(), p0=initial_guess, full_output=True,
                      bounds=((0.0, 0.0, 0.0, 0.0, 0.0, -np.pi / 2, 0.0), (4095, sx, sy, np.inf, np.inf, np.pi / 2, 4095)),
                      ftol=1e-3, xtol=1e-3)
        # p = curve_fit(gaussian2d, np.array((xx, yy)), arr.ravel(), p0=initial_guess, full_output=True)
    except RuntimeError:
        p = (initial_guess, np.zeros_like(initial_guess), 'fitting unsuccessful')
    dt = time.time() - tic
    # print(f'Fitting time: {time.time() - tic:.2f} s')
    # print(tuple(p[0]))
    # print(tuple(np.sqrt(np.diag(p[1]))))
    # print(p[2])

    pars = p[0]
    pars = (pars[0], pars[1] * rebinning, pars[2] * rebinning, pars[3] * rebinning, pars[4] * rebinning,
            pars[5], pars[6])

    print(f'time: {dt:05.2f} s, ampl: {int(pars[0]):04d}, x_0: {int(pars[1]):04d}, y_0: {int(pars[2]):04d}, '
          f's_x: {int(pars[3]):04d}, s_y: {int(pars[4]):04d}, angle: {pars[5]:+.2f}, offset: {int(pars[6]):04d}')

    xx = np.linspace(0, sx0 - 1, sx0)
    yy = np.linspace(0, sy0 - 1, sy0)
    xx, yy = np.meshgrid(xx, yy)

    gauss = np.reshape(gaussian2d(np.array((xx, yy)), *pars), (sy0, sx0))
    # gauss = zoom(gauss, rebinning, order=0)
    pars = {'amplitude': pars[0], 'offset': pars[6], 'angle': pars[5],
            'x_0': pars[1], 'y_0': pars[2],
            's_x': pars[3], 's_y': pars[4]}

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

        self.fit_switch = QCheckBox('fit')
        self.fit_switch.setToolTip('fit Gaussian to image')
        self.fit_switch.setChecked(False)

        self.worker = GaussianFitterWorker(self.thread())
        self.worker_thread = None
        self.sig_fit.connect(self.worker.fit)
        self.worker.fitted.connect(self.fitted)

        layout = QMyHBoxLayout(self.fit_switch)
        self.setLayout(layout)

    def fit(self, data):
        if self.fit_switch.isChecked():
            self.worker_thread = QThread()
            self.start_branch(self.worker, self.worker_thread, self.sig_fit, data)
        else:
            self.sig_fitted.emit(data)

    @pyqtSlot(dict, name='Fitted')
    def fitted(self, data):
        self.sig_fitted.emit(data)


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
        # layout = QMyVBoxLayout(self.controller, self.fitter, alignment=Qt.AlignmentFlag.AlignCenter)
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
    app = QApplication(sys.argv)
    ex = MainWindow()
    sys.exit(app.exec())
