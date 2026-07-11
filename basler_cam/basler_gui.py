"""Live GUI for Basler cameras.

Interface layer only: all hardware access goes through basler_cameras.py
(the device layer), which stays GUI-free so the app can later be ported to
a unified HTML GUI without touching the camera code.

Run:
    python basler_gui.py
    python basler_gui.py --self-test 5   # auto-enable cameras, quit after 5 s
"""

import argparse
import sys
import threading

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (QApplication, QCheckBox, QDoubleSpinBox,
                             QGroupBox, QHBoxLayout, QLabel, QMainWindow,
                             QMessageBox, QPushButton, QVBoxLayout, QWidget)

try:
    from basler_cam.basler_cameras import BaslerCamera, CameraStreamer
    from basler_cam.gaussian_fit import FitLoop
except ImportError:
    from basler_cameras import BaslerCamera, CameraStreamer
    from gaussian_fit import FitLoop

pg.setConfigOptions(imageAxisOrder='row-major')

MAX_LIVE_CAMERAS = 2
DISPLAY_INTERVAL_MS = 33  # GUI refresh rate (~30 Hz); shows only the latest frame
LEVELS_12BIT = (0, 4095)
PIXEL_SIZE_MM = 5.5 / 1000.0  # acA2040 pixel pitch
FIT_PEN = pg.mkPen((255, 90, 90, 170), width=1)  # thin and translucent, to not cover the mode
DATA_PEN = pg.mkPen((70, 140, 220), width=1)  # cross-section data
FIT_CURVE_PEN = pg.mkPen((255, 165, 40), width=1)  # cross-section fit curve
CIRCLE_PEN = pg.mkPen((0, 220, 220, 220), width=1)  # user circle annotation
GUESS_PEN = pg.mkPen((110, 255, 110, 220), width=1,
                     style=Qt.PenStyle.DashLine)  # fit initial-guess circle


class CircleDragViewBox(pg.ViewBox):
    """ViewBox where, while circle_mode is on, a left-button drag draws a
    circle (center at press, radius follows the cursor) instead of panning.
    All other interactions (wheel zoom, right-drag) stay untouched."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.circle_mode = False
        self.on_circle_drag = None  # callable(stage, x, y), stage: start/move/finish

    def mouseDragEvent(self, ev, axis=None):
        if not (self.circle_mode and ev.button() == Qt.MouseButton.LeftButton
                and self.on_circle_drag is not None):
            return super().mouseDragEvent(ev, axis)
        ev.accept()
        position = self.mapSceneToView(ev.scenePos())
        if ev.isStart():
            stage = 'start'
        elif ev.isFinish():
            stage = 'finish'
        else:
            stage = 'move'
        self.on_circle_drag(stage, position.x(), position.y())


class CameraPanel(QGroupBox):
    """Live view and controls for one connected camera.

    The streamer delivers frames from a background thread; they are only
    *stored* there (cheap, thread-safe). A GUI timer picks up the latest
    frame at display rate, so a slow display (or, later, a slow fit) can
    never accumulate a backlog behind the camera.
    """

    stream_error = pyqtSignal(str, str)  # serial_number, error message
    setting_applied = pyqtSignal(str, float)  # setting name, value the camera accepted
    fit_done = pyqtSignal(bool, dict)  # success, fit parameters

    def __init__(self, serial_number, camera, parent=None):
        super().__init__(f'camera {serial_number}', parent)
        self.serial_number = serial_number
        self.camera = camera

        self._frame_lock = threading.Lock()
        self._latest_frame = None
        self._displayed_frame_id = 0
        self._latest_frame_id = 0

        self.streamer = CameraStreamer(camera,
                                       on_frame=self._on_frame,
                                       on_error=self._on_error)

        # --- controls
        self.btn_play = QPushButton('play')
        self.btn_play.setToolTip('resume continuous grabbing')
        self.btn_play.clicked.connect(self.play)
        self.btn_pause = QPushButton('pause')
        self.btn_pause.setToolTip('pause continuous grabbing')
        self.btn_pause.clicked.connect(self.pause)
        self.btn_single = QPushButton('single frame')
        self.btn_single.setToolTip('pause and grab one frame')
        self.btn_single.clicked.connect(self.single)
        self.lbl_status = QLabel('streaming')

        # camera is still ours alone here (streamer not started yet),
        # so reading limits and current values directly is safe
        self.spin_exposure = QDoubleSpinBox()
        self.spin_exposure.setToolTip('exposure time; applied on Enter or '
                                      'when the box loses focus')
        self.spin_exposure.setSuffix(' μs')
        self.spin_exposure.setDecimals(0)
        self.spin_exposure.setRange(*camera.exposure_limits_us)
        self.spin_exposure.setValue(camera.exposure_us)
        self.spin_exposure.setKeyboardTracking(False)
        self.spin_exposure.valueChanged.connect(self.apply_exposure)

        self.spin_gain = QDoubleSpinBox()
        self.spin_gain.setToolTip('gain; applied on Enter or when the box '
                                  'loses focus')
        self.spin_gain.setSuffix(' dB')
        self.spin_gain.setDecimals(1)
        self.spin_gain.setRange(*camera.gain_limits_db)
        self.spin_gain.setValue(camera.gain_db)
        self.spin_gain.setKeyboardTracking(False)
        self.spin_gain.valueChanged.connect(self.apply_gain)

        self.setting_applied.connect(self._on_setting_applied)

        self.check_fit = QCheckBox('fit')
        self.check_fit.setToolTip('fit a 2D Gaussian to each displayed frame')
        self.check_fit.toggled.connect(self._on_fit_toggled)
        self.lbl_fit = QLabel('')

        controls = QHBoxLayout()
        controls.addWidget(self.btn_play)
        controls.addWidget(self.btn_pause)
        controls.addWidget(self.btn_single)
        controls.addWidget(QLabel('exposure:'))
        controls.addWidget(self.spin_exposure)
        controls.addWidget(QLabel('gain:'))
        controls.addWidget(self.spin_gain)
        controls.addWidget(self.check_fit)
        controls.addWidget(self.lbl_status)
        controls.addStretch()

        fit_row = QHBoxLayout()
        fit_row.addWidget(self.lbl_fit)
        fit_row.addStretch()

        # --- live view: image with cross-section plots through the fit
        # center (row on top, column on the right), like the original GUI.
        # The cross-section plots keep their slots also when the fit is off.
        self.graphics = pg.GraphicsLayoutWidget()
        self.graphics.setBackground((70, 70, 70))
        self.view_box = CircleDragViewBox()
        self.view_box.on_circle_drag = self._on_circle_drag
        self.image_plot = self.graphics.addPlot(row=1, col=0,
                                                viewBox=self.view_box)
        self.image_plot.setAspectLocked(True)
        self.image_plot.invertY(True)  # row 0 at the top, like the raw sensor data
        self.image_item = pg.ImageItem()
        self.image_plot.addItem(self.image_item)

        # The area outside the sensor is gray with a border marking the frame
        # edge, so it cannot be confused with dark pixels. Zooming out is
        # allowed up to one frame of slack on each side (with the aspect
        # ratio locked, a wide window needs extra horizontal range to show
        # the full vertical range); panning cannot lose the frame off-screen.
        height, width = camera.frame_shape
        border = pg.PlotCurveItem(x=[0, width, width, 0, 0],
                                  y=[0, 0, height, height, 0],
                                  pen=pg.mkPen((160, 160, 160), width=1))
        self.image_plot.addItem(border)
        view_box = self.image_plot.getViewBox()
        view_box.setLimits(xMin=-width, xMax=2 * width,
                           yMin=-height, yMax=2 * height)
        view_box.setRange(xRange=(0, width), yRange=(0, height), padding=0)

        self.hplot = self.graphics.addPlot(row=0, col=0)
        self.hplot.setXLink(self.image_plot)
        self.hplot.setYRange(*LEVELS_12BIT, padding=0)
        self.curve_h_data = self.hplot.plot(pen=DATA_PEN)
        self.curve_h_fit = self.hplot.plot(pen=FIT_CURVE_PEN)

        self.vplot = self.graphics.addPlot(row=1, col=1)
        self.vplot.setYLink(self.image_plot)
        self.vplot.invertY(True)
        self.vplot.setXRange(*LEVELS_12BIT, padding=0)
        self.curve_v_data = self.vplot.plot(pen=DATA_PEN)
        self.curve_v_fit = self.vplot.plot(pen=FIT_CURVE_PEN)

        grid = self.graphics.ci.layout
        grid.setRowStretchFactor(0, 1)
        grid.setRowStretchFactor(1, 4)
        grid.setColumnStretchFactor(0, 4)
        grid.setColumnStretchFactor(1, 1)

        # --- user circles, drawn by pressing on the center and dragging to
        # the circumference. Two kinds: the marker circle is a pure
        # annotation persisting across frames; the guess circle feeds the
        # Gaussian fit its initial parameters (center -> x_0/y_0,
        # radius -> sigma_x = sigma_y).
        self.btn_circle = QPushButton('mark circle')
        self.btn_circle.setCheckable(True)
        self.btn_circle.setToolTip('press on the circle center in the image '
                                   'and drag to its circumference')
        self.btn_circle.toggled.connect(
            lambda active: self._on_circle_mode('marker', active))
        self.btn_circle_clear = QPushButton('clear circle')
        self.btn_circle_clear.clicked.connect(self.clear_circle)
        self.lbl_circle = QLabel('')
        self._circle_center = None
        self._drag_target = None  # which circle a drag is drawing
        self.circle_item = pg.PlotCurveItem(pen=CIRCLE_PEN)
        self.circle_center_item = pg.ScatterPlotItem(symbol='+', size=12,
                                                     pen=CIRCLE_PEN, brush=None)
        self.image_plot.addItem(self.circle_item)
        self.image_plot.addItem(self.circle_center_item)

        self.btn_guess = QPushButton('guess circle')
        self.btn_guess.setCheckable(True)
        self.btn_guess.setToolTip('draw the fit initial guess: circle center '
                                  '-> (x_0, y_0), radius -> sigma')
        self.btn_guess.toggled.connect(
            lambda active: self._on_circle_mode('guess', active))
        self.btn_guess_clear = QPushButton('clear guess')
        self.btn_guess_clear.clicked.connect(self.clear_guess_circle)
        self.lbl_guess = QLabel('')
        self.guess_circle_item = pg.PlotCurveItem(pen=GUESS_PEN)
        self.guess_center_item = pg.ScatterPlotItem(symbol='+', size=12,
                                                    pen=GUESS_PEN, brush=None)
        self.image_plot.addItem(self.guess_circle_item)
        self.image_plot.addItem(self.guess_center_item)

        # fit overlay: thin translucent ellipses at 1 sigma and at the beam
        # radius w = 2 sigma, plus a center marker
        self.fit_ellipse_sigma = pg.PlotCurveItem(pen=FIT_PEN)
        self.fit_ellipse_w = pg.PlotCurveItem(pen=FIT_PEN)
        self.fit_center = pg.ScatterPlotItem(symbol='+', size=14, pen=FIT_PEN,
                                             brush=None)
        for item in (self.fit_ellipse_sigma, self.fit_ellipse_w, self.fit_center):
            self.image_plot.addItem(item)

        self._fit_pars = None  # last successful fit, for the cross sections
        self.fit_loop = FitLoop(
            on_result=lambda success, pars: self.fit_done.emit(success, pars))
        self.fit_done.connect(self._on_fit_done)

        circle_row = QHBoxLayout()
        circle_row.addWidget(self.btn_circle)
        circle_row.addWidget(self.btn_circle_clear)
        circle_row.addWidget(self.lbl_circle)
        circle_row.addWidget(self.btn_guess)
        circle_row.addWidget(self.btn_guess_clear)
        circle_row.addWidget(self.lbl_guess)
        circle_row.addStretch()

        layout = QVBoxLayout()
        layout.addLayout(controls)
        layout.addLayout(fit_row)
        layout.addLayout(circle_row)
        layout.addWidget(self.graphics)
        self.setLayout(layout)

        self.display_timer = QTimer(self)
        self.display_timer.setInterval(DISPLAY_INTERVAL_MS)
        self.display_timer.timeout.connect(self._update_display)
        self.display_timer.start()

        self.streamer.start()
        self._set_playing(True)

    # ------------------------------------------------- streamer callbacks
    # called from the streaming thread: store only, no GUI work here
    def _on_frame(self, frame):
        with self._frame_lock:
            self._latest_frame = frame
            self._latest_frame_id += 1

    def _on_error(self, error):
        self.stream_error.emit(self.serial_number, str(error))

    # ---------------------------------------------------------- GUI side
    def _update_display(self):
        with self._frame_lock:
            if self._latest_frame_id == self._displayed_frame_id:
                return  # no new frame since last refresh
            frame = self._latest_frame
            self._displayed_frame_id = self._latest_frame_id
        self.image_item.setImage(frame, autoLevels=False, levels=LEVELS_12BIT)
        self._update_cross_sections(frame)
        if self.check_fit.isChecked():
            # newest-frame-only: if the fit is slower than the frame rate,
            # intermediate frames are skipped, never queued
            self.fit_loop.submit(frame)

    def _update_cross_sections(self, frame):
        """Plot the image row/column through the fit center (or the frame
        center when there is no fit), with the fitted curve on top."""
        height, width = frame.shape
        pars = self._fit_pars if self.check_fit.isChecked() else None
        if pars is not None:
            x0, y0 = pars['x_0'], pars['y_0']
        else:
            x0, y0 = width / 2, height / 2
        column = int(np.clip(round(x0), 0, width - 1))
        row = int(np.clip(round(y0), 0, height - 1))
        xs = np.arange(width)
        ys = np.arange(height)
        self.curve_h_data.setData(xs, frame[row, :])
        self.curve_v_data.setData(frame[:, column], ys)
        if pars is not None:
            # cross sections of the fitted 2D Gaussian along the lines
            # y = y0 and x = x0
            sin2, cos2 = np.sin(pars['angle']) ** 2, np.cos(pars['angle']) ** 2
            a = cos2 / (2 * pars['s_x'] ** 2) + sin2 / (2 * pars['s_y'] ** 2)
            c = sin2 / (2 * pars['s_x'] ** 2) + cos2 / (2 * pars['s_y'] ** 2)
            offset, amplitude = pars['offset'], pars['amplitude']
            self.curve_h_fit.setData(xs, offset + amplitude * np.exp(-a * (xs - x0) ** 2))
            self.curve_v_fit.setData(offset + amplitude * np.exp(-c * (ys - y0) ** 2), ys)
        else:
            self.curve_h_fit.setData([], [])
            self.curve_v_fit.setData([], [])

    def _set_playing(self, playing):
        self.btn_play.setEnabled(not playing)
        self.btn_pause.setEnabled(playing)
        self.lbl_status.setText('streaming' if playing else 'paused')

    def play(self):
        self.streamer.resume()
        self._set_playing(True)

    def pause(self):
        self.streamer.pause()
        self._set_playing(False)

    def single(self):
        self.pause()
        self.streamer.snap()

    # -------------------------------------------------- exposure and gain
    # The pylon camera object is not thread-safe, so settings changes are
    # submitted to the streaming thread and executed there between grabs.
    # The value the camera actually accepted comes back via setting_applied.
    def apply_exposure(self, value):
        def command(camera):
            camera.exposure_us = value
            self.setting_applied.emit('exposure', camera.exposure_us)
        self.streamer.submit(command)

    def apply_gain(self, value):
        def command(camera):
            camera.gain_db = value
            self.setting_applied.emit('gain', camera.gain_db)
        self.streamer.submit(command)

    def _on_setting_applied(self, name, value):
        spinbox = self.spin_exposure if name == 'exposure' else self.spin_gain
        if abs(spinbox.value() - value) > 10 ** -spinbox.decimals() / 2:
            spinbox.blockSignals(True)
            spinbox.setValue(value)
            spinbox.blockSignals(False)

    # ------------------------------------------------------- gaussian fit
    def _on_fit_toggled(self, checked):
        if checked:
            self.fit_loop.start()
            self.lbl_fit.setText('fitting...')
            # fit the current frame right away, also when paused
            with self._frame_lock:
                frame = self._latest_frame
            if frame is not None:
                self.fit_loop.submit(frame)
        else:
            self.fit_loop.stop()
            self.lbl_fit.setText('')
            self._fit_pars = None
            self._clear_fit_overlay()
            self._refresh_after_fit()

    def _clear_fit_overlay(self):
        self.fit_ellipse_sigma.setData([], [])
        self.fit_ellipse_w.setData([], [])
        self.fit_center.setData([], [])
        self.curve_h_fit.setData([], [])
        self.curve_v_fit.setData([], [])

    def _refresh_after_fit(self):
        """Redraw the cross sections when a fit result arrives, so they
        also update while paused (no new frames coming)."""
        with self._frame_lock:
            frame = self._latest_frame
        if frame is not None:
            self._update_cross_sections(frame)

    def _on_fit_done(self, success, pars):
        if not self.check_fit.isChecked():
            return
        if not success:
            self.lbl_fit.setText(f"fit: {pars.get('reason', 'did not converge')}")
            self._fit_pars = None
            self._clear_fit_overlay()
            self._refresh_after_fit()
            return
        self._fit_pars = pars
        x0, y0 = pars['x_0'], pars['y_0']
        theta = pars['angle']
        t = np.linspace(0, 2 * np.pi, 120)
        ex = pars['s_x'] * np.cos(t)
        ey = pars['s_y'] * np.sin(t)
        cos_r, sin_r = np.cos(theta), np.sin(theta)
        for item, scale in ((self.fit_ellipse_sigma, 1.0), (self.fit_ellipse_w, 2.0)):
            item.setData(x0 + scale * (ex * cos_r - ey * sin_r),
                         y0 + scale * (ex * sin_r + ey * cos_r))
        self.fit_center.setData([x0], [y0])
        self.lbl_fit.setText(
            f"x₀ = {x0:.1f} px, y₀ = {y0:.1f} px, "
            f"w_x = {pars['w_x'] * PIXEL_SIZE_MM:.3f} mm, "
            f"w_y = {pars['w_y'] * PIXEL_SIZE_MM:.3f} mm, "
            f"θ = {theta:+.2f} rad, fit took {pars['time']:.2f} s")
        self._refresh_after_fit()

    # ------------------------------------------------------- user circles
    def _on_circle_mode(self, which, active):
        if active:
            other = self.btn_guess if which == 'marker' else self.btn_circle
            if other.isChecked():
                other.setChecked(False)
            self._drag_target = which
        elif self._drag_target == which:
            self._drag_target = None
        self._circle_center = None
        self.view_box.circle_mode = self._drag_target is not None
        self.btn_circle.setText('drag center → edge...'
                                if self._drag_target == 'marker' else 'mark circle')
        self.btn_guess.setText('drag center → edge...'
                               if self._drag_target == 'guess' else 'guess circle')

    def _on_circle_drag(self, stage, x, y):
        if self._drag_target is None:
            return
        marker = self._drag_target == 'marker'
        if stage == 'start':
            self._circle_center = (x, y)
            (self.circle_center_item if marker
             else self.guess_center_item).setData([x], [y])
            (self.circle_item if marker else self.guess_circle_item).setData([], [])
            return
        if self._circle_center is None:
            return
        center_x, center_y = self._circle_center
        radius = float(np.hypot(x - center_x, y - center_y))
        # live preview while dragging
        if marker:
            self.set_circle(center_x, center_y, radius)
        else:
            self.set_guess_circle(center_x, center_y, radius)
        if stage == 'finish':
            if not marker:
                self._apply_guess(center_x, center_y, max(radius, 1.0))
            # resets the button text
            (self.btn_circle if marker else self.btn_guess).setChecked(False)

    def set_circle(self, center_x, center_y, radius):
        t = np.linspace(0, 2 * np.pi, 120)
        self.circle_item.setData(center_x + radius * np.cos(t),
                                 center_y + radius * np.sin(t))
        self.circle_center_item.setData([center_x], [center_y])
        self.lbl_circle.setText(
            f'circle: center ({center_x:.0f}, {center_y:.0f}) px, '
            f'r = {radius:.1f} px = {radius * PIXEL_SIZE_MM:.3f} mm')

    def clear_circle(self):
        self._circle_center = None
        self.circle_item.setData([], [])
        self.circle_center_item.setData([], [])
        self.lbl_circle.setText('')

    def set_guess_circle(self, center_x, center_y, radius):
        t = np.linspace(0, 2 * np.pi, 120)
        self.guess_circle_item.setData(center_x + radius * np.cos(t),
                                       center_y + radius * np.sin(t))
        self.guess_center_item.setData([center_x], [center_y])
        self.lbl_guess.setText(
            f'guess: center ({center_x:.0f}, {center_y:.0f}) px, '
            f'σ = {radius:.1f} px')

    def _apply_guess(self, center_x, center_y, sigma):
        self.fit_loop.guess = {'x_0': center_x, 'y_0': center_y, 'sigma': sigma}
        if self.check_fit.isChecked():
            # refit the current frame right away, also when paused
            with self._frame_lock:
                frame = self._latest_frame
            if frame is not None:
                self.fit_loop.submit(frame)

    def clear_guess_circle(self):
        self.fit_loop.guess = None
        self.guess_circle_item.setData([], [])
        self.guess_center_item.setData([], [])
        self.lbl_guess.setText('')

    def shutdown(self):
        self.display_timer.stop()
        self.fit_loop.stop()
        self.streamer.stop()
        self.camera.close()


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle('Basler Cameras')

        self.panels = {}  # serial number -> CameraPanel
        self.checkboxes = {}  # serial number -> QCheckBox

        self.btn_rescan = QPushButton('rescan')
        self.btn_rescan.setToolTip('search again for connected cameras')
        self.btn_rescan.clicked.connect(self.rescan)

        self.checkbox_row = QHBoxLayout()
        selector_box = QGroupBox('available cameras (check to connect)')
        selector_layout = QHBoxLayout()
        selector_layout.addLayout(self.checkbox_row)
        selector_layout.addStretch()
        selector_layout.addWidget(self.btn_rescan)
        selector_box.setLayout(selector_layout)

        self.panel_row = QHBoxLayout()

        central = QWidget()
        layout = QVBoxLayout()
        layout.addWidget(selector_box)
        layout.addLayout(self.panel_row, stretch=1)
        central.setLayout(layout)
        self.setCentralWidget(central)

        self.rescan()

    # ------------------------------------------------------ camera list
    def rescan(self):
        try:
            devices = BaslerCamera.list_devices()
        except Exception as error:
            QMessageBox.critical(self, 'scan failed', str(error))
            return
        found = [d['serial_number'] for d in devices]
        # add checkboxes for newly found cameras, keep existing ones
        for device in devices:
            sn = device['serial_number']
            if sn not in self.checkboxes:
                checkbox = QCheckBox(f"{sn} ({device['model']})")
                checkbox.toggled.connect(
                    lambda checked, sn=sn: self.on_camera_toggled(sn, checked))
                self.checkboxes[sn] = checkbox
                self.checkbox_row.addWidget(checkbox)
        # grey out checkboxes of cameras that disappeared (unless connected)
        for sn, checkbox in self.checkboxes.items():
            checkbox.setEnabled(sn in found or sn in self.panels)
        if not self.checkboxes:
            QMessageBox.information(self, 'no cameras',
                                    'no Basler cameras found on this computer')

    # ------------------------------------------------- connect / disconnect
    def on_camera_toggled(self, serial_number, checked):
        if checked:
            self.connect_camera(serial_number)
        else:
            self.disconnect_camera(serial_number)

    def _uncheck_silently(self, serial_number):
        checkbox = self.checkboxes[serial_number]
        checkbox.blockSignals(True)
        checkbox.setChecked(False)
        checkbox.blockSignals(False)

    def connect_camera(self, serial_number):
        if len(self.panels) >= MAX_LIVE_CAMERAS:
            self._uncheck_silently(serial_number)
            QMessageBox.warning(self, 'limit reached',
                                f'more than {MAX_LIVE_CAMERAS} live cameras '
                                f'are not supported')
            return
        camera = BaslerCamera(serial_number)
        try:
            camera.open()
        except Exception as error:
            self._uncheck_silently(serial_number)
            camera.close()
            QMessageBox.warning(
                self, 'connection failed',
                f'could not connect to camera {serial_number}.\n\n'
                f'if it is open in another program (e.g. the pylon viewer '
                f'or the old GUI), close it there first.\n\n'
                f'details: {error}')
            return
        panel = CameraPanel(serial_number, camera)
        panel.stream_error.connect(self.on_stream_error)
        self.panels[serial_number] = panel
        self.panel_row.addWidget(panel, stretch=1)

    def disconnect_camera(self, serial_number):
        panel = self.panels.pop(serial_number, None)
        if panel is None:
            return
        panel.shutdown()
        self.panel_row.removeWidget(panel)
        panel.deleteLater()

    def on_stream_error(self, serial_number, message):
        self._uncheck_silently(serial_number)
        self.disconnect_camera(serial_number)
        QMessageBox.warning(self, 'streaming error',
                            f'camera {serial_number} stopped streaming.\n\n'
                            f'details: {message}')

    def closeEvent(self, event):
        for serial_number in list(self.panels):
            self.disconnect_camera(serial_number)
        event.accept()


def main():
    parser = argparse.ArgumentParser(description='live GUI for Basler cameras')
    parser.add_argument('--self-test', type=float, metavar='SECONDS', default=None,
                        help='auto-connect all cameras (up to the limit), '
                             'stream for SECONDS, then quit')
    args = parser.parse_args()

    app = QApplication(sys.argv)
    window = MainWindow()
    window.resize(1200, 700)
    window.show()

    if args.self_test is not None:
        def auto_connect():
            for sn in list(window.checkboxes)[:MAX_LIVE_CAMERAS]:
                window.checkboxes[sn].setChecked(True)
            for panel in window.panels.values():
                panel.setting_applied.connect(
                    lambda name, value, sn=panel.serial_number:
                    print(f'{sn}: applied {name} = {value:.1f}'))

        def exercise_controls():
            for panel in window.panels.values():
                panel.spin_exposure.setValue(5000)
                panel.spin_gain.setValue(1.5)
                panel.fit_done.connect(
                    lambda success, pars, sn=panel.serial_number:
                    print(f'{sn}: fit success={success} '
                          f'{pars.get("reason", "")}'
                          f'{" w_x=%.1f px" % pars["w_x"] if success else ""}'))
                panel.check_fit.setChecked(True)
                panel.set_circle(1024, 1024, 300)
                panel.set_guess_circle(1024, 1024, 200)
                panel._apply_guess(1024, 1024, 200)
                panel.single()

        QTimer.singleShot(500, auto_connect)
        QTimer.singleShot(int(500 + args.self_test * 500), exercise_controls)
        QTimer.singleShot(int(500 + args.self_test * 1000), window.close)

    sys.exit(app.exec())


if __name__ == '__main__':
    main()
