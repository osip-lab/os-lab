# basic libraries
import sys
import os
import time
import logging
import numpy as np
# gui import
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, pyqtSlot
from PyQt6.QtWidgets import QApplication, QWidget, QAbstractSpinBox, QCheckBox
# my extension of gui
from qt_gui.qt_ext import MyStandardWindow, QMyHBoxLayout, QMyVBoxLayout, ThreadedWidget, QMySpinBox, QMyStandardButton
# controllers for other devices
from rigol_gen.rigol_gen_sin_gui import RigolGenSinWidget
from pico_scope.pico_scope_control_gui import PicoControlWidget, SaverWidget, PyQtPlotterWidget
# function for a spectre point calculation
from bode_plot_measure_gui import calculate_spectral_point
# local path to data bank
from local_config import path_data_local


class LinewidthMeasurerWidget(ThreadedWidget):
    sig_start_pico = pyqtSignal(name='StartPico')
    sig_set_timebase = pyqtSignal(float, name='SetTimeBase')
    sig_set_duration = pyqtSignal(float, name='SetDuration')
    sig_set_pre_duration = pyqtSignal(float, name='SetPreDuration')
    sig_load_pico = pyqtSignal(name='LoadPico')

    def __init__(self, font_size=14):
        super(LinewidthMeasurerWidget, self).__init__(font_size=font_size)
        self.setTitle('Linewidth Measurer')

        self.freq = None

        self.spinbox_fsr = QMySpinBox(decimals=3, v_ini=150.0, v_min=1.0, v_max=9999.0, step=10,
                                      prefix='FSR: ', suffix=' MHz')

        self.btn_measure = QMyStandardButton('measure', font_size=self.font_size)
        self.btn_measure.clicked.connect(self.start_measuring)
        self.measuring = False

        self.spinbox_tpl = QMySpinBox(decimals=3, v_min=0.0, v_max=9999.0, prefix=f'T+L: ', suffix=' ppm')
        self.spinbox_tpl.setReadOnly(True)
        self.spinbox_tpl.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)

        self.moment = QTimer()
        self.moment.setInterval(10)
        self.moment.setSingleShot(True)
        self.on_wait = None
        self.moment.timeout.connect(self.timeout)

        self.switch_auto = QCheckBox('auto')
        self.switch_auto.setChecked(False)

        self.timer_auto = QTimer()
        self.timer_auto.setInterval(1000)
        self.timer_auto.setSingleShot(True)
        self.timer_auto.timeout.connect(self.start_measuring)

        layout = QMyVBoxLayout(self.spinbox_fsr, self.btn_measure, self.switch_auto,self.spinbox_tpl)
        self.setLayout(layout)

    def wait_and_emit(self, signal):
        self.on_wait = signal
        self.moment.start()

    def timeout(self):
        self.on_wait.emit()

    @pyqtSlot(name='StartMeasuring')
    def start_measuring(self):
        self.measuring = True
        self.sig_start_pico.emit()

    @pyqtSlot(dict, name='Fit')
    def fit(self, signals):
        if self.measuring:

            freq = self.freq
            fsr = self.spinbox_fsr.value() * 1e6
            t = signals['time']
            ac_in = signals['1']
            ac_out = signals['2']
            dc_in = signals['3']
            dc_out = signals['4']

            ampl_in, phase_in = calculate_spectral_point(t, ac_in, freq)
            ampl_out, phase_out = calculate_spectral_point(t, ac_out, freq)
            v_dc_in = np.mean(dc_in)
            v_dc_out = np.mean(dc_out)

            big_a = ampl_out / ampl_in * v_dc_in / v_dc_out
            if big_a < 1:
                tpl = 2 * np.pi * freq / fsr / (1 / big_a**2 - 1)**0.5
                log_msg = f'measured linewidth successfully - T + L = {tpl * 1e6:.2f} ppm'
                logging.info(log_msg)
                self.spinbox_tpl.setValue(tpl * 1e6)
            else:
                log_msg = f'linewidth measurement failed'
                logging.info(log_msg)

        self.measuring = False
        if self.switch_auto.isChecked():
            self.timer_auto.start()

    @pyqtSlot(dict, name='RigolLoaded')
    def rigol_loaded(self, settings):
        self.freq = settings['ch1']['freq']


class MainWidget(QWidget):

    def __init__(self):
        super(MainWidget, self).__init__()
        self.font_size = 14
        self.channels = tuple('1234')

        self.main_controller = LinewidthMeasurerWidget(font_size=self.font_size)
        self.rigol_controller = RigolGenSinWidget(font_size=self.font_size)
        self.pico_controller = PicoControlWidget(font_size=self.font_size, channels=self.channels)
        self.saver = SaverWidget(font_size=self.font_size, default_folder='linewidth_measurer')
        self.plotter = PyQtPlotterWidget(self.channels, font_size=self.font_size)

        # pass loaded modulation frequency to the main controller
        self.rigol_controller.sig_loaded.connect(self.main_controller.rigol_loaded)

        # start picoscope and get data
        self.main_controller.sig_start_pico.connect(self.pico_controller.ps_start)
        self.pico_controller.plot_signals.connect(self.main_controller.fit)

        # connections to the plotter
        self.pico_controller.plot_signals.connect(self.plotter.plot_signals)
        self.pico_controller.settings_loaded.connect(self.plotter.load_settings)

        # connections to enable settings saving
        self.saver.auto_save_changed.connect(self.pico_controller.set_auto_save)
        self.saver.claim_settings.connect(self.pico_controller.provide_settings)
        self.pico_controller.save_settings.connect(self.saver.save_settings)
        self.saver.load_settings.connect(self.pico_controller.load_settings)

        self.saver.load_defaults()

        layout = QMyVBoxLayout()
        layout.addLayout(QMyHBoxLayout(self.rigol_controller, self.main_controller,self.saver))
        layout.addWidget(self.pico_controller)
        layout.addWidget(self.plotter)
        self.setLayout(layout)


class MainWindow(MyStandardWindow):

    def __init__(self):
        super().__init__()
        font = self.font()
        font.setPointSize(14)
        self.widget = MainWidget()
        self.widget.setFont(font)
        self.widget.font_size = 14
        self.appear_with_central_widget('Linewidth Measurer', self.widget)


if __name__ == '__main__':

    start_time = time.strftime('%Y-%m-%d %H-%M-%S')
    logging.basicConfig(filename=os.path.join(path_data_local, 'linewidth_measurer', f'{start_time} log.txt'),
                        level=logging.INFO, format='%(asctime)s.%(msecs)03d: %(message)s', datefmt='%Y.%m.%d %H:%M:%S')

    app = QApplication(sys.argv)
    ex = MainWindow()
    sys.exit(app.exec())
