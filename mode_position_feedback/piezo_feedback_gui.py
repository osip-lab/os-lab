import sys
import numpy as np
import pyvisa
from pyvisa.resources import Resource

from PyQt6.QtWidgets import QApplication, QWidget, QAbstractSpinBox, QCheckBox
from PyQt6.QtCore import Qt, QThread, pyqtSignal, pyqtSlot

from qt_gui.qt_ext import (MyStandardWindow, QMyStandardButton, QMyHBoxLayout, QMyVBoxLayout, ThreadedWorker,
                           ThreadedWidget, QMySpinBox)

def command(inst: Resource, com, silent=True):
    inst.write(com)
    ans = b''
    ch = b''
    while ch != b'>':
        ch = inst.read_bytes(1)
        ans = ans + ch
    com, ans = ans.split(b'\n')
    ans = ans[:-1].split(b'\r')
    while b'' in ans:
        ans.remove(b'')
    print(f"command: {com.decode('ascii')}") if not silent else None
    if len(ans) == 1:
        print(f"answer: {ans[0].decode('ascii')}") if not silent else None
    elif len(ans) > 1:
        print('answers:')
        for a in ans:
            print(a.decode('ascii')) if not silent else None
    return com.decode('ascii'), [a.decode('ascii') for a in ans]

class PiezoFeedbackWorker(ThreadedWorker):
    connected = pyqtSignal(name='Connected')
    measured = pyqtSignal(dict, name='Measured')
    loaded = pyqtSignal(name='Loaded')

    def __init__(self, thread):
        super(PiezoFeedbackWorker, self).__init__(thread)
        self.controller = None
        self.settings = dict()

    @pyqtSlot(name='Connect')
    def connect(self):
        rm = pyvisa.ResourceManager()
        rl = rm.list_resources()
        print(rl)

        for i in range(len(rl)):
            print('connecting', rl[i])
            try:
                self.controller = rm.open_resource(rl[i])
                command(self.controller, 'id?', silent=False)
            except pyvisa.errors.VisaIOError:
                print(rl[i], 'not connected')

        print(self.controller)

        self.finish(self.connected)

    def measure_voltages(self):
        for ax in ('x', 'y', 'z'):
            com, ans = command(self.controller, f'{ax}voltage?')
            self.settings[f'{ax}_volt'] = float(ans[0][1:-1])

    @pyqtSlot(name='Measure')
    def measure(self):
        self.measure_voltages()
        self.finish(self.measured, self.settings)

    @pyqtSlot(dict, name='Load')
    def load(self, s):
        command(self.controller, f'yvoltage={s["2"]}')
        command(self.controller, f'zvoltage={s["3"]}')
        self.finish(self.loaded)

class PiezoFeedbackWidget(ThreadedWidget):
    # commands for internal worker
    sig_connect = pyqtSignal(name='Connect')
    sig_center_to_fit = pyqtSignal(name='Center_to_fit')
    sig_read_V = pyqtSignal(name='Measure')
    sig_write_V = pyqtSignal(dict, name='Load')

    xi, yi = 1024.0, 1024.0
    V23 = np.array([0, 0])
    mat = np.array([[1/np.sqrt(3), -1.], [2/np.sqrt(3), 0.]])

    def __init__(self, font_size=14):
        super(PiezoFeedbackWidget, self).__init__(font_size=font_size)
        self.setTitle('Piezo feedback')

        self.btn_connect = QMyStandardButton('connect', font_size=self.font_size)
        #self.btn_connect.setToolTip('scan for available Rigol Generator')
        self.btn_connect.clicked.connect(self.connect)
        self.spin_box_x_s = QMySpinBox(v_min=-10000.0, v_max=10000.0, v_ini=0, decimals=1, step=1.0, suffix=' pxl', prefix='x_s: ')
        self.spin_box_y_s = QMySpinBox(v_min=-10000.0, v_max=10000.0, v_ini=0, decimals=1, step=1.0, suffix=' pxl', prefix='y_s: ')
        self.btn_center_to_fit = QMyStandardButton('center to fit', font_size=self.font_size)
        self.btn_center_to_fit.clicked.connect(self.center_to_fit)
        #self.spin_box_V1 = QMySpinBox(v_min=-10000.0, v_max=10000.0, v_ini=0, decimals=2, step=1.0, suffix=' V', prefix='V1: ')
        #self.spin_box_V1.setReadOnly(True)
        #self.spin_box_V1.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.spin_box_V2 = QMySpinBox(v_min=-10000.0, v_max=10000.0, v_ini=0, decimals=2, step=1.0, suffix=' V', prefix='V2: ')
        self.spin_box_V2.setReadOnly(True)
        self.spin_box_V2.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.spin_box_V3 = QMySpinBox(v_min=-10000.0, v_max=10000.0, v_ini=0, decimals=2, step=1.0, suffix=' V', prefix='V3: ')
        self.spin_box_V3.setReadOnly(True)
        self.spin_box_V3.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.btn_read_V = QMyStandardButton('read voltage', font_size=self.font_size)
        self.btn_read_V.clicked.connect(self.read_V)
        self.lock_switch = QCheckBox('lock')
        self.lock_switch.setChecked(False)
        self.spin_sensitivity = QMySpinBox(v_min=-10000.0, v_max=10000.0, v_ini=20, decimals=3, step=1, suffix=' pxl/V', prefix='sensitivity: ')
        self.spin_integral_coef = QMySpinBox(v_min=1.0, v_max=10000.0, v_ini=10, decimals=1, step=1.0, suffix='', prefix='I_c: ')

        self.worker = PiezoFeedbackWorker(self.thread())
        self.worker_thread = None
        self.sig_connect.connect(self.worker.connect)
        self.sig_read_V.connect(self.worker.measure)
        self.sig_write_V.connect(self.worker.load)
        self.worker.measured.connect(self.measured)
        self.worker.loaded.connect(self.loaded)

        layout = QMyVBoxLayout()
        lt = QMyHBoxLayout(self.btn_connect, self.spin_box_x_s, self.spin_box_y_s, self.btn_center_to_fit)
        lt.addStretch(0)
        layout.addLayout(lt)
        lt = QMyHBoxLayout(self.spin_box_V2, self.spin_box_V3, self.btn_read_V)#self.spin_box_V1, self.spin_box_V2, self.spin_box_V3, self.btn_read_V)
        lt.addStretch(0)
        layout.addLayout(lt)
        lt = QMyHBoxLayout(self.lock_switch, self.spin_sensitivity, self.spin_integral_coef)
        lt.addStretch(0)
        layout.addLayout(lt)
        self.setLayout(layout)

    @pyqtSlot(name='Connect')
    def connect(self):
        self.worker_thread = QThread()
        self.start_branch(self.worker, self.worker_thread, self.sig_connect)

    def feedback(self, data):
        if 'parameters' in data.keys():
            self.xi, self.yi = float(data['parameters']['x_0']), float(data['parameters']['y_0'])
            if self.lock_switch.isChecked():
                self.V23 = self.V23 - (1/(self.spin_sensitivity.value() * self.spin_integral_coef.value())) * np.dot(self.mat,[self.xi - self.spin_box_x_s.value(), self.yi - self.spin_box_y_s.value()])
                # print(self.V23)
                self.worker_thread = QThread()
                self.start_branch(self.worker, self.worker_thread, self.sig_write_V, {"2": self.V23[0], "3": self.V23[1]})
                self.spin_box_V2.setValue(self.V23[0])
                self.spin_box_V3.setValue(self.V23[1])

    def center_to_fit(self):
        self.spin_box_x_s.setValue(self.xi)
        self.spin_box_y_s.setValue(self.yi)

    def read_V(self):
        self.worker_thread = QThread()
        self.start_branch(self.worker, self.worker_thread, self.sig_read_V)

    def measured(self, data):
        #self.spin_box_V1.setValue(data['x_volt'])
        self.V23 = [data['y_volt'], data['z_volt']]
        self.spin_box_V2.setValue(self.V23[0])
        self.spin_box_V3.setValue(self.V23[1])

    def loaded(self):
        a = 1

class PiezoFeedbackMainWidget(QWidget):
    def __init__(self):
        super(PiezoFeedbackMainWidget, self).__init__()
        self.font_size = 14

        self.controller = PiezoFeedbackWidget(font_size=self.font_size)

        layout = QMyVBoxLayout()
        layout.addWidget(self.controller, alignment=Qt.AlignmentFlag.AlignCenter)
        self.setLayout(layout)


class PiezoFeedbackWindow(MyStandardWindow):
    def __init__(self):
        super().__init__()
        self.piezo_feedback_widget = PiezoFeedbackMainWidget()
        font = self.font()
        font.setPointSize(14)
        self.piezo_feedback_widget.setFont(font)
        self.piezo_feedback_widget.font_size = 14
        self.appear_with_central_widget('PiezoDriver_MDT693B', self.piezo_feedback_widget)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = PiezoFeedbackWindow()
    sys.exit(app.exec())