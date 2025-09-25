import sys
import os
import time
import logging
import numpy as np
from scipy.optimize import curve_fit

from PyQt6.QtWidgets import QApplication, QWidget, QCheckBox, QAbstractSpinBox
from PyQt6.QtCore import pyqtSignal, pyqtSlot, QTimer

from qt_gui.qt_ext import MyStandardWindow, QMyVBoxLayout, QMyHBoxLayout, QMySpinBox, ThreadedWidget
from rigol_dsa import RigolDSAWidget

from local_config import PATH_DATA_LOCAL


def calc_mode(freq, fsr, tpl):
    return np.exp(1j * np.pi * freq / fsr) / (1 - (1 - tpl) * np.exp(1j * 2 * np.pi * freq / fsr))


def calc_line_shape(freq, fsr, fo, tpl, ampl):
    return ampl * tpl ** 2 / 2 * np.abs(calc_mode(fo, fsr, tpl) * np.conj(calc_mode(fo + freq, fsr, tpl)) +
                                        np.conj(calc_mode(fo, fsr, tpl)) * calc_mode(fo - freq, fsr, tpl))


def fit_line_shape(freq, ampl):
    guess = (np.mean(freq), 0.0, 25e-6, np.max(ampl) - np.min(ampl))
    pars = curve_fit(calc_line_shape, freq, ampl, full_output=True, p0=guess)
    pars = list(map(float, pars[0]))
    answer = {'fsr': pars[0], 'freq_offset': pars[1], 'tpl': pars[2], 'ampl': pars[3],
              'fit': calc_line_shape(freq, *pars)}
    return answer


class FitterWidget(ThreadedWidget):
    sig_start = pyqtSignal(name='Start')

    def __init__(self, font_size=14):
        super(FitterWidget, self).__init__(font_size=font_size)
        self.setTitle('Line Shape Fitter')

        self.switch_fit = QCheckBox('fit')
        self.switch_fit.setChecked(False)
        self.switch_fit.setToolTip('enable fitting')

        self.switch_auto = QCheckBox('continuous')
        self.switch_auto.setChecked(False)
        self.switch_auto.setToolTip('enable continuous measurement')

        self.labels = ('FSR', 'TpL', 'ampl', 'f_off')
        self.spinboxes = dict()
        self.spinboxes['FSR'] = QMySpinBox(decimals=3, v_ini=150e6, v_min=0.0, v_max=999999999.0, prefix='FSR: ', suffix=' Hz')
        self.spinboxes['TpL'] = QMySpinBox(decimals=3, v_ini=25.0, v_min=0.0, v_max=999.0, prefix='TpL: ', suffix=' ppm')
        self.spinboxes['ampl'] = QMySpinBox(decimals=6, v_ini=1.0, v_min=0.0, v_max=999.0, prefix='ampl: ', suffix=' mV')
        self.spinboxes['f_off'] = QMySpinBox(decimals=3, v_ini=0.0, v_min=-999.0, v_max=999.0, prefix='f_off: ', suffix=' Hz')
        for lbl in self.labels:
            self.spinboxes[lbl].setReadOnly(True)
            self.spinboxes[lbl].setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)

        self.moment = QTimer()
        self.moment.setInterval(10)
        self.moment.setSingleShot(True)
        self.on_wait = None
        self.moment.timeout.connect(self.timeout)

        layout = QMyVBoxLayout()
        layout.addLayout(QMyHBoxLayout(self.switch_fit, self.switch_auto))
        for lbl in self.labels:
            layout.addWidget(self.spinboxes[lbl])
        self.setLayout(layout)

    def wait_and_emit(self, signal, delay=None):
        self.on_wait = signal
        if delay is None:
            self.moment.setInterval(10)
        else:
            self.moment.setInterval(int(delay))
        self.moment.start()

    def timeout(self):
        self.on_wait.emit()

    @pyqtSlot(dict, name='Fit')
    def process(self, data):

        # this part is just for tests
        # data = np.loadtxt(os.path.join(PATH_DATA_LOCAL, 'line_shape_measurer', 'test.txt'))
        # data = data.transpose()
        # data = {'freq': data[0], 'ampl': data[1]}

        if self.switch_fit.isChecked():
            try:
                data = fit_line_shape(data['freq'], data['ampl'])

                self.spinboxes['FSR'].setValue(data['fsr'])
                self.spinboxes['TpL'].setValue(data['tpl'] * 1e6)
                self.spinboxes['ampl'].setValue(data['ampl'] * 1e3)
                self.spinboxes['f_off'].setValue(data['freq_offset'])

                log_msg = (f"measured line shape - FSR = {data['fsr']:013.3f} Hz, T + L = {data['tpl'] * 1e6:07.3f} ppm, "
                           f"ampl = {data['ampl'] * 1e3:010.6f}, frequency offset = {data['freq_offset']:07.3f} Hz")
                logging.info(log_msg)
            except RuntimeError:
                logging.info('line shape measurement failed')
        else:
            logging.info('line shape measurement skipped')

        if self.switch_auto.isChecked():
            if self.switch_fit.isChecked():
                self.wait_and_emit(self.sig_start)
            else:
                self.wait_and_emit(self.sig_start, delay=300)


class MainWidget(QWidget):

    def __init__(self):
        super(MainWidget, self).__init__()
        self.font_size = 14

        self.controller = RigolDSAWidget(font_size=self.font_size)
        self.fitter = FitterWidget(font_size=self.font_size)

        self.controller.sig_measured.connect(self.fitter.process)
        self.fitter.sig_start.connect(self.controller.start)

        layout = QMyVBoxLayout(self.controller, self.fitter)
        self.setLayout(layout)


class MainWindow(MyStandardWindow):

    def __init__(self):
        super().__init__()
        font = self.font()
        font.setPointSize(14)
        self.widget = MainWidget()
        self.widget.setFont(font)
        self.widget.font_size = 14
        self.appear_with_central_widget('Line Shape Measurer', self.widget)


if __name__ == '__main__':

    start_time = time.strftime('%Y-%m-%d %H-%M-%S')
    logging.basicConfig(filename=os.path.join(PATH_DATA_LOCAL, 'line_shape_measurer', f'{start_time} log.txt'),
                        level=logging.INFO, format='%(asctime)s.%(msecs)03d: %(message)s', datefmt='%Y.%m.%d %H:%M:%S')

    app = QApplication(sys.argv)
    ex = MainWindow()
    sys.exit(app.exec())
