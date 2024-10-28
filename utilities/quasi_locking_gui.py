"""
TODO:
    - add on-fly rdc fitting
    - block full ramp during ramp to prevent conflict in loading of settings
"""

import sys
import numpy as np
from scipy.signal import find_peaks

from PyQt6.QtWidgets import QApplication, QWidget, QCheckBox
from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot

from qt_gui.qt_ext import MyStandardWindow, ThreadedWidget, QMyVBoxLayout, QMyHBoxLayout, QMySpinBox
from pico_scope.pico_scope_control_gui import PicoControlWidget, SaverWidget, PyQtPlotterWidget
from rigol_gen.rigol_gen_ramp_gui import RigolGenRampBiasWidget


class QuasiLockingWidget(ThreadedWidget):
    switch_changed = pyqtSignal(bool, name='SwitchChanged')
    plot_signals = pyqtSignal(dict, name='PlotSignals')
    change_bias = pyqtSignal(float, name='ChangeBias')

    def __init__(self, font_size=14):
        super(QuasiLockingWidget, self).__init__(font_size=font_size)
        self.setTitle('Locker')

        self.lock_switch = QCheckBox('lock')
        self.lock_switch.stateChanged.connect(lambda x: self.switch_changed.emit(True) if x == 2 else self.switch_changed.emit(False))
        self.test_switch = QCheckBox('test')

        self.threshold = QMySpinBox(v_min=0.1, v_max=5.0, v_ini=2.0, decimals=1, step=1.0, suffix=' V')
        self.step = QMySpinBox(v_min=0.1, v_max=5.0, v_ini=1.0, decimals=1, step=1.0, suffix=' mV')

        layout = QMyHBoxLayout(self.lock_switch, self.test_switch, self.threshold, self.step)
        self.setLayout(layout)

    @pyqtSlot(dict, name='process_signals')
    def process_signals(self, signals):
        if self.test_switch.isChecked():
            pos = (np.random.random() - 0.5) * 2e-3 * 10.0
            tc = signals['time'][len(signals['time']) // 2]
            ts = 20e-6
            signals['2'] = (np.exp(-(signals['time'] - tc - pos)**2 / (2 * ts**2)) / 5 +
                            np.exp(-(signals['time'] - tc - pos - 3e-3) ** 2 / (2 * ts**2)) / 20 +
                            np.exp(-(signals['time'] - tc - pos + 3e-3)**2 / (2 * ts**2)) / 20)
        if self.lock_switch.isChecked():
            peaks = find_peaks(signals['2'], height=1e-3)
            index = peaks[0][peaks[1]['peak_heights'].argmax(axis=0)]
            ramp_volt = signals['1'][index]
            thr = self.threshold.value()
            step = self.step.value() * 1e-3
            if ramp_volt > thr:
                self.change_bias.emit(-step)
            elif ramp_volt < -thr:
                self.change_bias.emit(step)
        self.plot_signals.emit(signals)


class QuasiLockingMainWidget(QWidget):
    def __init__(self):
        super(QuasiLockingMainWidget, self).__init__(flags=Qt.Window)
        self.font_size = 14
        self.pico_channels = tuple('1234')

        self.ramp_controller = RigolGenRampBiasWidget(font_size=self.font_size)
        self.pico_controller = PicoControlWidget(self.pico_channels, font_size=self.font_size)
        self.locker = QuasiLockingWidget(font_size=self.font_size)
        self.saver = SaverWidget(font_size=self.font_size, default_folder='quasi_locking')
        self.plotter = PyQtPlotterWidget(self.pico_channels, font_size=self.font_size)

        # self.pico_controller.plot_signals.connect(self.plotter.plot_signals)
        self.pico_controller.plot_signals.connect(self.locker.process_signals)
        self.locker.plot_signals.connect(self.plotter.plot_signals)
        self.locker.switch_changed.connect(self.ramp_controller.block_bias)
        self.locker.change_bias.connect(self.ramp_controller.change_bias)

        self.pico_controller.settings_loaded.connect(self.plotter.load_settings)
        self.pico_controller.save_data.connect(self.saver.save_data)
        self.pico_controller.save_settings.connect(self.saver.save_settings)
        self.pico_controller.data_received.connect(self.saver.data_received)
        self.saver.load_settings.connect(self.pico_controller.load_settings)
        self.saver.claim_data.connect(self.pico_controller.provide_data)
        self.saver.claim_settings.connect(self.pico_controller.provide_settings)
        self.saver.auto_save_changed.connect(self.pico_controller.set_auto_save)

        # load defaults settings
        self.saver.load_defaults()

        layout = QMyVBoxLayout()
        lt = QMyHBoxLayout()
        lt.addLayout(QMyVBoxLayout(self.ramp_controller, self.locker))
        lt.addLayout(QMyHBoxLayout(self.pico_controller, self.saver))
        layout.addLayout(lt)
        layout.addWidget(self.plotter)
        self.setLayout(layout)


class QuasiLockingWindow(MyStandardWindow):
    def __init__(self):
        super().__init__()
        self.quasi_locking_widget = QuasiLockingMainWidget()
        font = self.font()
        font.setPointSize(14)
        self.quasi_locking_widget.setFont(font)
        self.quasi_locking_widget.font_size = 14
        self.appear_with_central_widget('Quasi Locking', self.quasi_locking_widget)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = QuasiLockingWindow()
    sys.exit(app.exec_())
