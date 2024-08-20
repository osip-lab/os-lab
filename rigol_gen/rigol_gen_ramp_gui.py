import sys
import pyvisa

from PyQt6.QtWidgets import QApplication, QWidget
from PyQt6.QtCore import Qt, QThread, pyqtSignal, pyqtSlot

from qt_gui.qt_ext import MyStandardWindow, QMyStandardButton, QCenteredLabel, QMyHBoxLayout, QMyVBoxLayout, \
    QMySpinBox, ThreadedWorker, ThreadedWidget


class RigolGenRampWorker(ThreadedWorker):
    connected = pyqtSignal(name='Connected')
    started = pyqtSignal(name='Started')
    stopped = pyqtSignal(name='Stopped')
    loaded = pyqtSignal(name='Loaded')
    bias_loaded = pyqtSignal(name='BiasLoaded')

    def __init__(self, thread):
        super(RigolGenRampWorker, self).__init__(thread)
        self.gen = None
        self.settings = dict()

    @pyqtSlot(name='Connect')
    def connect(self):
        rm = pyvisa.ResourceManager()
        rl = rm.list_resources()

        if len(list(filter(lambda x: 'DG8' in x, rl))) == 1:
            self.gen = rm.open_resource(list(filter(lambda x: 'DG8' in x, rl))[0])
        else:
            raise ValueError('connect only one DG8xx Rigol Generator')
        self.finish(self.connected)

    @pyqtSlot(name='Start')
    def start(self):
        self.gen.write('OUTP1 ON')
        self.gen.write('OUTP2 ON')
        self.finish(self.started)

    @pyqtSlot(name='Stop')
    def stop(self):
        self.gen.write('OUTP1 OFF')
        self.gen.write('OUTP2 OFF')
        self.finish(self.stopped)

    @pyqtSlot(name='Load')
    def load(self):
        self.gen.write(f":SOUR1:APPL:RAMP {self.settings['freq']:.1f},{self.settings['ampl']:.1f},0,0")
        self.gen.write(f":SOUR2:APPL:DC 1,1,{self.settings['bias']}")
        self.finish(self.loaded)

    @pyqtSlot(name='LoadBias')
    def load_bias(self):
        self.gen.write(f":SOUR2:APPL:DC 1,1,{self.settings['bias']}")
        self.finish(self.bias_loaded)


class RigolGenRampWidget(ThreadedWidget):
    sig_connect = pyqtSignal(name='Connect')
    sig_start = pyqtSignal(name='Start')
    sig_stop = pyqtSignal(name='Stop')
    sig_load = pyqtSignal(name='Load')
    sig_load_bias = pyqtSignal(name='LoadBias')

    def __init__(self, font_size=14):
        super(RigolGenRampWidget, self).__init__(font_size=font_size)
        self.setTitle('Ramp and Bias')

        self.btn_connect = QMyStandardButton('connect', font_size=self.font_size)
        self.btn_connect.setToolTip('connect to a device')
        self.btn_connect.clicked.connect(self.connect)
        self.btn_start = QMyStandardButton('start', font_size=self.font_size)
        self.btn_start.setEnabled(False)
        self.btn_start.clicked.connect(self.start)
        self.btn_stop = QMyStandardButton('stop', font_size=self.font_size)
        self.btn_stop.clicked.connect(self.stop)
        self.btn_stop.setEnabled(False)

        self.btn_load = QMyStandardButton('load', font_size=self.font_size)
        self.btn_load.clicked.connect(self.load)
        self.btn_load.setEnabled(False)

        self.ramp_ampl = QMySpinBox(v_min=1.0, v_max=20.0, v_ini=20.0, decimals=1, step=1.0, suffix=' V')
        self.ramp_freq = QMySpinBox(v_min=5.0, v_max=50.0, v_ini=10.0, decimals=1, step=5.0, suffix=' Hz')
        self.temp_bias = QMySpinBox(v_min=-200.0, v_max=200.0, v_ini=0.0, decimals=1, step=1.0, suffix=' mV')
        self.temp_bias.valueChanged.connect(self.load_bias)

        self.settings = dict()

        self.rg_worker = RigolGenRampWorker(self.thread())
        self.rg_thread = None
        self.sig_connect.connect(self.rg_worker.connect)
        self.sig_start.connect(self.rg_worker.start)
        self.sig_stop.connect(self.rg_worker.stop)
        self.sig_load.connect(self.rg_worker.load)
        self.sig_load_bias.connect(self.rg_worker.load_bias)
        self.rg_worker.connected.connect(lambda: self.btn_load.setEnabled(True))
        self.rg_worker.loaded.connect(lambda: self.btn_start.setEnabled(True))
        self.rg_worker.loaded.connect(lambda: self.btn_stop.setEnabled(True))

        layout = QMyVBoxLayout()
        layout.addLayout(QMyHBoxLayout(self.btn_connect, self.btn_load, self.btn_start, self.btn_stop))
        layout.addLayout(QMyHBoxLayout(QCenteredLabel('Ch1 Ramp'), self.ramp_ampl, self.ramp_freq))
        lt = QMyHBoxLayout(QCenteredLabel('Ch2 DC'), self.temp_bias)
        lt.addStretch(0)
        layout.addLayout(lt)
        self.setLayout(layout)

    def connect(self):
        self.rg_thread = QThread()
        self.start_branch(self.rg_worker, self.rg_thread, self.sig_connect)

    def start(self):
        self.rg_thread = QThread()
        self.start_branch(self.rg_worker, self.rg_thread, self.sig_start)

    def stop(self):
        self.rg_thread = QThread()
        self.start_branch(self.rg_worker, self.rg_thread, self.sig_stop)

    def load(self):
        self.settings['ampl'] = self.ramp_ampl.value()
        self.settings['freq'] = self.ramp_freq.value()
        self.settings['bias'] = self.temp_bias.value() * 0.001
        self.rg_worker.settings = self.settings
        self.rg_thread = QThread()
        self.start_branch(self.rg_worker, self.rg_thread, self.sig_load)

    @pyqtSlot(name='LoadBias')
    def load_bias(self):
        self.settings['bias'] = self.temp_bias.value() * 0.001
        self.rg_worker.settings['bias'] = self.settings['bias']
        self.rg_thread = QThread()
        self.start_branch(self.rg_worker, self.rg_thread, self.sig_load_bias)

    @pyqtSlot(bool, name='BlockBias')
    def block_bias(self, state: bool):
        if state:
            self.temp_bias.setEnabled(False)
        else:
            self.temp_bias.setEnabled(True)

    @pyqtSlot(float, name='ChangeBias')
    def change_bias(self, v):
        self.temp_bias.setValue((self.temp_bias.value() * 1e-3 + v) * 1e3)


class RigolGenMainWidget(QWidget):
    def __init__(self):
        super(RigolGenMainWidget, self).__init__()
        self.font_size = 14

        self.controller = RigolGenRampWidget(font_size=self.font_size)

        layout = QMyVBoxLayout()
        layout.addWidget(self.controller, alignment=Qt.AlignmentFlag.AlignCenter)
        self.setLayout(layout)


class RigolGenRampWindow(MyStandardWindow):
    def __init__(self):
        super().__init__()
        self.rigol_gen_widget = RigolGenMainWidget()
        font = self.font()
        font.setPointSize(14)
        self.rigol_gen_widget.setFont(font)
        self.rigol_gen_widget.font_size = 14
        self.appear_with_central_widget('Rigol Generator Ramp and Bias', self.rigol_gen_widget)


if __name__ == '__main__':
    # QLocale.setDefault(QLocale(QLocale.C))
    app = QApplication(sys.argv)
    ex = RigolGenRampWindow()
    sys.exit(app.exec())
