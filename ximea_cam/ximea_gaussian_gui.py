"""
Ximea Camera Control and Gaussian Fitting GUI

Requirements:
    pip install ximea PyQt6 numpy scipy numba matplotlib pyperclip

Also install the Ximea Software Package / driver from Ximea.

Tested conceptually for xiAPI-compatible Ximea cameras such as MQ042MG-CM-S7-TG.
"""

import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pyperclip
from numba import njit, float64
from scipy.ndimage import center_of_mass, median_filter
from scipy.optimize import curve_fit

from ximea import xiapi

from PyQt6.QtCore import QObject, QThread, QTimer, Qt, pyqtSignal, pyqtSlot
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable


# ----------------------------- small helpers -----------------------------


def rebin_image(arr: np.ndarray, factor: int) -> np.ndarray:
    """Downsample image by block averaging."""
    factor = int(max(1, factor))
    if factor == 1:
        return arr.astype(np.float64, copy=False)

    sy, sx = arr.shape
    sy2 = sy // factor * factor
    sx2 = sx // factor * factor
    cropped = arr[:sy2, :sx2]
    return cropped.reshape(sy2 // factor, factor, sx2 // factor, factor).mean(axis=(1, 3))


@njit(float64[:](float64[:, :, ::1], float64, float64, float64, float64, float64, float64, float64))
def gaussian2d(xy, ampl, xo, yo, sigma_x, sigma_y, theta, offset):
    x, y = xy[0], xy[1]
    a = (np.cos(theta) ** 2) / (2 * sigma_x**2) + (np.sin(theta) ** 2) / (2 * sigma_y**2)
    b = -(np.sin(2 * theta)) / (4 * sigma_x**2) + (np.sin(2 * theta)) / (4 * sigma_y**2)
    c = (np.sin(theta) ** 2) / (2 * sigma_x**2) + (np.cos(theta) ** 2) / (2 * sigma_y**2)
    g = ampl * np.exp(-(a * (x - xo) ** 2 + 2 * b * (x - xo) * (y - yo) + c * (y - yo) ** 2)) + offset
    return g.ravel()


def fit_gaussian(arr: np.ndarray, rebinning: int = 4, median_size: int = 1) -> dict:
    """Fit rotated 2D Gaussian. Returns beam params in original-pixel units."""
    arr = np.asarray(arr)
    if arr.ndim != 2:
        raise ValueError("fit_gaussian expects a 2D monochrome image")

    if median_size and median_size > 1:
        arr = median_filter(arr, size=median_size)

    arr_rebinned = rebin_image(arr, rebinning)
    arr_rebinned = arr_rebinned.astype(np.float64, copy=False)

    sy, sx = arr_rebinned.shape
    xx, yy = np.meshgrid(np.arange(sx, dtype=np.float64), np.arange(sy, dtype=np.float64))
    xy = np.ascontiguousarray(np.stack((xx, yy)))

    background = float(np.percentile(arr_rebinned, 15))
    signal = arr_rebinned - background
    signal[signal < 0] = 0

    amplitude = float(np.max(arr_rebinned) - background)
    if amplitude <= 0:
        raise RuntimeError("No positive signal above background")

    y0, x0 = center_of_mass(signal)
    if not np.isfinite(x0) or not np.isfinite(y0):
        y0, x0 = sy / 2, sx / 2

    max_val = float(np.iinfo(arr.dtype).max) if np.issubdtype(arr.dtype, np.integer) else float(np.max(arr_rebinned))
    max_val = max(max_val, float(np.max(arr_rebinned)), 1.0)

    p0 = (amplitude, float(x0), float(y0), 10.0, 10.0, 0.0, background)
    bounds = (
        [0, 0, 0, 0.1, 0.1, -np.pi / 4, 0],
        [max_val, sx - 1, sy - 1, max(sx, sy), max(sx, sy), np.pi / 4, max_val],
    )

    popt, _ = curve_fit(
        gaussian2d,
        xy,
        arr_rebinned.ravel(),
        p0=p0,
        bounds=bounds,
        maxfev=3000,
    )

    fit_img_rebinned = gaussian2d(xy, *popt).reshape(sy, sx)

    return {
        "amplitude": float(popt[0]),
        "x_0": float(popt[1] * rebinning),
        "y_0": float(popt[2] * rebinning),
        "sigma_x": float(popt[3] * rebinning),
        "sigma_y": float(popt[4] * rebinning),
        "w_x": float(2 * popt[3] * rebinning),
        "w_y": float(2 * popt[4] * rebinning),
        "angle_deg": float(np.degrees(popt[5])),
        "offset": float(popt[6]),
        "fit_img_rebinned": fit_img_rebinned,
        "rebinning": int(rebinning),
    }


# ----------------------------- Ximea worker -----------------------------


class XimeaCamControlWorker(QObject):
    scanned = pyqtSignal(dict)
    connected = pyqtSignal(dict)
    loaded = pyqtSignal(dict)
    captured = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.cam = None
        self.img = xiapi.Image()
        self.settings = {}
        self.acquiring = False

    def _emit_error(self, prefix: str):
        self.error.emit(prefix + "\n" + traceback.format_exc())

    @pyqtSlot()
    def scan(self):
        try:
            probe = xiapi.Camera()
            n = probe.get_number_devices()
            serial_numbers = []
            devices = []
            for i in range(n):
                c = xiapi.Camera(dev_id=i)
                sn = c.get_device_info_string("device_sn")
                name = c.get_device_info_string("device_name")
                dtype = c.get_device_info_string("device_type")
                serial_numbers.append(sn)
                devices.append({"dev_id": i, "sn": sn, "name": name, "type": dtype})
            self.scanned.emit({"serial_numbers": serial_numbers, "devices": devices})
        except Exception:
            self._emit_error("Scan failed")

    @pyqtSlot(dict)
    def connect(self, settings):
        try:
            self.close_camera()
            self.settings = dict(settings)
            self.cam = xiapi.Camera()
            self.cam.open_device_by_SN(str(settings["sn"]))
            self.apply_settings(settings)

            info = {
                "model": self.cam.get_device_info_string("device_name"),
                "id": self.cam.get_device_info_string("device_type"),
                "sn": self.cam.get_device_info_string("device_sn"),
            }
            self.connected.emit(info)
        except Exception:
            self._emit_error("Connect failed")

    def apply_settings(self, settings):
        if self.cam is None:
            raise RuntimeError("Camera is not connected")

        self.settings = dict(settings)
        self.cam.set_exposure(int(settings["exposure_us"]))
        self.cam.set_gain(float(settings["gain_db"]))

        fmt = settings.get("img_format", "XI_MONO16")
        self.cam.set_imgdataformat(fmt)

        # Optional ROI. Use full sensor if width/height are 0.
        width = int(settings.get("width", 0))
        height = int(settings.get("height", 0))
        offset_x = int(settings.get("offset_x", 0))
        offset_y = int(settings.get("offset_y", 0))

        if width > 0:
            self.cam.set_width(width)
        if height > 0:
            self.cam.set_height(height)
        if offset_x >= 0:
            try:
                self.cam.set_offsetX(offset_x)
            except Exception:
                self.cam.set_offset_x(offset_x)
        if offset_y >= 0:
            try:
                self.cam.set_offsetY(offset_y)
            except Exception:
                self.cam.set_offset_y(offset_y)

    @pyqtSlot(dict)
    def load(self, settings):
        try:
            self.apply_settings(settings)
            self.loaded.emit(dict(settings))
        except Exception:
            self._emit_error("Load settings failed")

    @pyqtSlot()
    def capture(self):
        try:
            if self.cam is None:
                raise RuntimeError("Camera is not connected")

            if not self.acquiring:
                self.cam.start_acquisition()
                self.acquiring = True

            self.cam.get_image(self.img, timeout=5000)
            arr = self.img.get_image_data_numpy().copy()

            self.captured.emit(
                {
                    "arr": arr,
                    "timestamp": time.time(),
                    "settings": dict(self.settings),
                    "dtype": str(arr.dtype),
                    "shape": arr.shape,
                }
            )
        except Exception:
            self._emit_error("Capture failed")

    @pyqtSlot()
    def stop_acquisition(self):
        try:
            if self.cam is not None and self.acquiring:
                self.cam.stop_acquisition()
            self.acquiring = False
        except Exception:
            pass

    def close_camera(self):
        if self.cam is not None:
            try:
                if self.acquiring:
                    self.cam.stop_acquisition()
            except Exception:
                pass
            try:
                self.cam.close_device()
            except Exception:
                pass
        self.cam = None
        self.acquiring = False


# ----------------------------- fitter worker -----------------------------


class GaussianFitterWorker(QObject):
    fitted = pyqtSignal(dict)
    error = pyqtSignal(str)

    @pyqtSlot(dict)
    def fit(self, data):
        try:
            arr = data["arr"]
            params = fit_gaussian(
                arr,
                rebinning=int(data.get("rebinning", 4)),
                median_size=int(data.get("median_size", 1)),
            )
            params["timestamp"] = data.get("timestamp", time.time())
            self.fitted.emit(params)
        except Exception:
            self.error.emit("Fit failed\n" + traceback.format_exc())


# ----------------------------- plotting -----------------------------


class MplCanvas(FigureCanvas):
    def __init__(self):
        self.fig = Figure(figsize=(7, 6), dpi=100)
        super().__init__(self.fig)
        self.ax = self.fig.add_subplot(111)
        div = make_axes_locatable(self.ax)
        self.hax = div.append_axes("top", size="20%", pad=0.2, sharex=self.ax)
        self.vax = div.append_axes("right", size="20%", pad=0.2, sharey=self.ax)
        self.hax.tick_params(bottom=False, labelbottom=False, top=True, labeltop=True)
        self.vax.tick_params(left=False, labelleft=False, right=True, labelright=True)
        self.fig.tight_layout()
        self.last_arr = None
        self.last_fit = None
        self.draw_demo()

    def draw_demo(self):
        yy, xx = np.mgrid[:400, :400]
        arr = 1000 * np.exp(-((xx - 200) ** 2 + (yy - 190) ** 2) / (2 * 35**2)) + 80
        self.update_image(arr.astype(np.uint16), None)

    def update_image(self, arr: np.ndarray, fit: dict | None):
        self.last_arr = arr
        self.last_fit = fit

        self.ax.clear()
        self.hax.clear()
        self.vax.clear()

        arrf = arr.astype(float)
        self.ax.imshow(arrf, cmap="gray", origin="upper")
        self.ax.set_title(f"Image {arr.shape}, {arr.dtype}")

        hprof = arrf.sum(axis=0)
        vprof = arrf.sum(axis=1)
        self.hax.plot(np.arange(arr.shape[1]), hprof)
        self.vax.plot(vprof, np.arange(arr.shape[0]))
        self.vax.invert_xaxis()

        if fit is not None:
            x0 = fit["x_0"]
            y0 = fit["y_0"]
            wx = fit["w_x"]
            wy = fit["w_y"]
            self.ax.plot([x0], [y0], marker="+", markersize=12)
            self.ax.text(
                0.02,
                0.98,
                f"x0={x0:.1f}\ny0={y0:.1f}\nw_x={wx:.1f}\nw_y={wy:.1f}\nangle={fit['angle_deg']:.2f}°",
                transform=self.ax.transAxes,
                va="top",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
            )

        self.fig.tight_layout()
        self.draw_idle()


# ----------------------------- GUI -----------------------------


class XimeaCamControlWidget(QGroupBox):
    sig_scan = pyqtSignal()
    sig_connect = pyqtSignal(dict)
    sig_load = pyqtSignal(dict)
    sig_capture = pyqtSignal()
    sig_stop = pyqtSignal()

    def __init__(self):
        super().__init__("Ximea Camera Control")
        self.btn_scan = QPushButton("scan")
        self.combo_sn = QComboBox()
        self.btn_connect = QPushButton("connect")

        self.spin_exposure = QSpinBox()
        self.spin_exposure.setRange(1, 10_000_000)
        self.spin_exposure.setValue(10_000)
        self.spin_exposure.setSuffix(" us")

        self.spin_gain = QDoubleSpinBox()
        self.spin_gain.setRange(0.0, 24.0)
        self.spin_gain.setDecimals(1)
        self.spin_gain.setValue(0.0)
        self.spin_gain.setSuffix(" dB")

        self.combo_format = QComboBox()
        self.combo_format.addItems(["XI_MONO16", "XI_MONO8", "XI_RAW16", "XI_RAW8"])

        self.spin_width = QSpinBox()
        self.spin_width.setRange(0, 4096)
        self.spin_width.setValue(0)
        self.spin_width.setToolTip("0 = leave camera default/full width")

        self.spin_height = QSpinBox()
        self.spin_height.setRange(0, 4096)
        self.spin_height.setValue(0)
        self.spin_height.setToolTip("0 = leave camera default/full height")

        self.spin_offset_x = QSpinBox()
        self.spin_offset_x.setRange(0, 4096)
        self.spin_offset_x.setValue(0)

        self.spin_offset_y = QSpinBox()
        self.spin_offset_y.setRange(0, 4096)
        self.spin_offset_y.setValue(0)

        self.btn_load = QPushButton("load")
        self.btn_capture = QPushButton("capture")
        self.auto_switch = QCheckBox("auto")

        layout = QGridLayout(self)
        row = 0
        layout.addWidget(self.btn_scan, row, 0)
        layout.addWidget(self.combo_sn, row, 1)
        layout.addWidget(self.btn_connect, row, 2)
        row += 1
        layout.addWidget(QLabel("Exposure"), row, 0)
        layout.addWidget(self.spin_exposure, row, 1)
        layout.addWidget(QLabel("Gain"), row, 2)
        layout.addWidget(self.spin_gain, row, 3)
        row += 1
        layout.addWidget(QLabel("Format"), row, 0)
        layout.addWidget(self.combo_format, row, 1)
        layout.addWidget(QLabel("W/H"), row, 2)
        layout.addWidget(self.spin_width, row, 3)
        layout.addWidget(self.spin_height, row, 4)
        row += 1
        layout.addWidget(QLabel("Offset X/Y"), row, 0)
        layout.addWidget(self.spin_offset_x, row, 1)
        layout.addWidget(self.spin_offset_y, row, 2)
        layout.addWidget(self.btn_load, row, 3)
        layout.addWidget(self.btn_capture, row, 4)
        layout.addWidget(self.auto_switch, row, 5)

        self.timer = QTimer(self)
        self.timer.setInterval(1000)

        self.btn_scan.clicked.connect(self.sig_scan.emit)
        self.btn_connect.clicked.connect(lambda: self.sig_connect.emit(self.settings()))
        self.btn_load.clicked.connect(lambda: self.sig_load.emit(self.settings()))
        self.btn_capture.clicked.connect(self.sig_capture.emit)
        self.timer.timeout.connect(self.sig_capture.emit)
        self.auto_switch.toggled.connect(self._on_auto)

    def settings(self):
        return {
            "sn": self.combo_sn.currentText(),
            "exposure_us": int(self.spin_exposure.value()),
            "gain_db": float(self.spin_gain.value()),
            "img_format": self.combo_format.currentText(),
            "width": int(self.spin_width.value()),
            "height": int(self.spin_height.value()),
            "offset_x": int(self.spin_offset_x.value()),
            "offset_y": int(self.spin_offset_y.value()),
        }

    def _on_auto(self, checked: bool):
        if checked:
            self.timer.start()
        else:
            self.timer.stop()
            self.sig_stop.emit()

    @pyqtSlot(dict)
    def on_scanned(self, info):
        self.combo_sn.clear()
        self.combo_sn.addItems(info.get("serial_numbers", []))


class GaussianFitterWidget(QGroupBox):
    sig_fit = pyqtSignal(dict)

    def __init__(self):
        super().__init__("Gaussian Fit")
        self.enable_fit = QCheckBox("fit")
        self.enable_fit.setChecked(True)

        self.spin_rebin = QSpinBox()
        self.spin_rebin.setRange(1, 32)
        self.spin_rebin.setValue(4)

        self.spin_median = QSpinBox()
        self.spin_median.setRange(1, 15)
        self.spin_median.setSingleStep(2)
        self.spin_median.setValue(1)

        self.btn_copy = QPushButton("copy params")
        self.label = QLabel("No fit yet")
        self.label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.last_params = None

        layout = QVBoxLayout(self)
        top = QHBoxLayout()
        top.addWidget(self.enable_fit)
        top.addWidget(QLabel("rebin"))
        top.addWidget(self.spin_rebin)
        top.addWidget(QLabel("median"))
        top.addWidget(self.spin_median)
        top.addWidget(self.btn_copy)
        layout.addLayout(top)
        layout.addWidget(self.label)

        self.btn_copy.clicked.connect(self.copy_params)

    def maybe_fit(self, data):
        if not self.enable_fit.isChecked():
            return
        data = dict(data)
        data["rebinning"] = self.spin_rebin.value()
        data["median_size"] = self.spin_median.value()
        self.sig_fit.emit(data)

    @pyqtSlot(dict)
    def on_fitted(self, params):
        self.last_params = params
        txt = (
            f"x0 = {params['x_0']:.2f} px\n"
            f"y0 = {params['y_0']:.2f} px\n"
            f"w_x = {params['w_x']:.2f} px\n"
            f"w_y = {params['w_y']:.2f} px\n"
            f"sigma_x = {params['sigma_x']:.2f} px\n"
            f"sigma_y = {params['sigma_y']:.2f} px\n"
            f"angle = {params['angle_deg']:.3f} deg\n"
            f"amplitude = {params['amplitude']:.2f}\n"
            f"offset = {params['offset']:.2f}"
        )
        self.label.setText(txt)

    def copy_params(self):
        if self.last_params is None:
            return
        keys = ["x_0", "y_0", "w_x", "w_y", "sigma_x", "sigma_y", "angle_deg", "amplitude", "offset"]
        txt = "\n".join(f"{k}: {self.last_params[k]}" for k in keys)
        pyperclip.copy(txt)


class MainWindow(QMainWindow):
    request_fit = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Ximea Gaussian Camera GUI")
        self.resize(1200, 850)

        self.cam_widget = XimeaCamControlWidget()
        self.fit_widget = GaussianFitterWidget()
        self.canvas = MplCanvas()
        self.status = QLabel("Ready")
        self.status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        left = QVBoxLayout()
        left.addWidget(self.cam_widget)
        left.addWidget(self.fit_widget)
        left.addWidget(self.status)
        left.addStretch(1)

        main = QHBoxLayout()
        main.addLayout(left, stretch=0)
        main.addWidget(self.canvas, stretch=1)

        central = QWidget()
        central.setLayout(main)
        self.setCentralWidget(central)

        self.cam_thread = QThread(self)
        self.cam_worker = XimeaCamControlWorker()
        self.cam_worker.moveToThread(self.cam_thread)
        self.cam_thread.start()

        self.fit_thread = QThread(self)
        self.fit_worker = GaussianFitterWorker()
        self.fit_worker.moveToThread(self.fit_thread)
        self.fit_thread.start()

        self._connect_signals()

    def _connect_signals(self):
        self.cam_widget.sig_scan.connect(self.cam_worker.scan)
        self.cam_widget.sig_connect.connect(self.cam_worker.connect)
        self.cam_widget.sig_load.connect(self.cam_worker.load)
        self.cam_widget.sig_capture.connect(self.cam_worker.capture)
        self.cam_widget.sig_stop.connect(self.cam_worker.stop_acquisition)

        self.cam_worker.scanned.connect(self.cam_widget.on_scanned)
        self.cam_worker.scanned.connect(lambda info: self.status.setText(f"Found {len(info.get('serial_numbers', []))} camera(s)"))
        self.cam_worker.connected.connect(lambda info: self.status.setText(f"Connected: {info}"))
        self.cam_worker.loaded.connect(lambda s: self.status.setText(f"Settings loaded: {s}"))
        self.cam_worker.captured.connect(self.on_captured)
        self.cam_worker.error.connect(self.on_error)

        self.fit_widget.sig_fit.connect(self.fit_worker.fit)
        self.fit_worker.fitted.connect(self.on_fitted)
        self.fit_worker.error.connect(self.on_error)

    @pyqtSlot(dict)
    def on_captured(self, data):
        arr = data["arr"]
        self.canvas.update_image(arr, self.fit_widget.last_params)
        self.status.setText(f"Captured {arr.shape}, {arr.dtype}, max={np.max(arr)}")
        self.fit_widget.maybe_fit(data)

    @pyqtSlot(dict)
    def on_fitted(self, params):
        self.fit_widget.on_fitted(params)
        if self.canvas.last_arr is not None:
            self.canvas.update_image(self.canvas.last_arr, params)

    @pyqtSlot(str)
    def on_error(self, msg):
        self.status.setText(msg)
        print(msg, file=sys.stderr)

    def closeEvent(self, event):
        self.cam_widget.timer.stop()
        self.cam_worker.stop_acquisition()
        self.cam_worker.close_camera()
        self.cam_thread.quit()
        self.cam_thread.wait(2000)
        self.fit_thread.quit()
        self.fit_thread.wait(2000)
        event.accept()


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
