import sys
import pyvisa

from PyQt6.QtWidgets import QApplication, QWidget, QComboBox
from PyQt6.QtCore import Qt, QThread, pyqtSignal, pyqtSlot

from qt_gui.qt_ext import (MyStandardWindow, QMyStandardButton, QMyHBoxLayout, QMyVBoxLayout, ThreadedWorker,
                           ThreadedWidget)


class RigolGenWorker(ThreadedWorker):
    scanned = pyqtSignal(dict, name='Scanned')
    connected = pyqtSignal(name='Connected')
    started = pyqtSignal(name='Started')
    stopped = pyqtSignal(name='Stopped')
    loaded = pyqtSignal(name='Loaded')
    bias_loaded = pyqtSignal(name='BiasLoaded')

    def __init__(self, thread):
        super(RigolGenWorker, self).__init__(thread)
        self.rm = None
        self.rl = None
        self.gen = None
        self.settings = {'ch1': {'type': 'none'}, 'ch2': {'type': 'none'}}
        self.channels = {'ch1': '1', 'ch2': '2'}
        self.types = {'dc': 'DC', 'ramp': 'RAMP', 'sin': 'SIN'}

    @pyqtSlot(name='Scan')
    def scan(self):
        self.rm = pyvisa.ResourceManager()
        self.rl = self.rm.list_resources()

        rg_sns = list(filter(lambda x: 'DG8' in x, self.rl))
        rg_sns = list(map(lambda x: x.split('::')[3], rg_sns))

        info = {'serial_numbers': rg_sns}
        self.finish(self.scanned, info)

    @pyqtSlot(dict, name='Connect')
    def connect(self, settings):
        self.settings['sn'] = settings['sn']
        self.gen = self.rm.open_resource(list(filter(lambda x: self.settings['sn'] in x, self.rl))[0])
        self.finish(self.connected)

    @pyqtSlot(dict, name='Start')
    def start(self, settings):
        for c in self.channels.keys():
            if settings[c]['status'] is True:
                self.gen.write(f'OUTP{self.channels[c]} ON')
            self.settings[c] = settings[c]
        self.finish(self.started)

    @pyqtSlot(dict, name='Stop')
    def stop(self, settings):
        for c in self.channels.keys():
            if settings[c]['status'] is False:
                self.gen.write(f'OUTP{self.channels[c]} OFF')
            self.settings[c] = settings[c]
        self.finish(self.stopped)

    @pyqtSlot(dict, name='Load')
    def load(self, settings):
        for c in self.channels.keys():
            old = self.apply_command(c, self.settings[c])
            new = self.apply_command(c, settings[c])
            if old != new:
                self.gen.write(new)
                self.settings[c] = settings[c]
        self.finish(self.loaded)

    def apply_command(self, c, s):
        if s['type'] == 'none':
            return ''
        elif s['type'] == 'dc':
            return f":SOUR{self.channels[c]}:APPL:{self.types[s['type']]} 1,1,{s['ampl']:.3f}"
        else:
            return f":SOUR{self.channels[c]}:APPL:{self.types[s['type']]} {s['freq']:.3f},{s['ampl']:.3f},0,0"


class RigolGenWidget(ThreadedWidget):
    sig_scan = pyqtSignal(name='Scan')
    sig_connect = pyqtSignal(dict, name='Connect')
    sig_start = pyqtSignal(dict, name='Start')
    sig_stop = pyqtSignal(dict, name='Stop')
    sig_load = pyqtSignal(dict, name='Load')

    def __init__(self, font_size=14):
        super(RigolGenWidget, self).__init__(font_size=font_size)
        self.setTitle('Rigol Gen 0 V DC')

        self.btn_scan = QMyStandardButton('scan', font_size=self.font_size)
        self.btn_scan.setToolTip('scan for available Rigol Generator')
        self.btn_scan.clicked.connect(self.scan)
        self.combobox_sn = QComboBox()
        self.combobox_sn.setToolTip('serial numbers of Rigol Generators connected to PC')
        self.combobox_sn.setMinimumContentsLength(14)

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

        self.settings = dict()
        self.channels = ('ch1', 'ch2')
        self.active = {c: True for c in self.channels}  # defines which channels to switch
        self.statuses = {c: False for c in self.channels}  # defines the status of the channels

        self.worker = RigolGenWorker(self.thread())
        self.worker_thread = None
        self.sig_scan.connect(self.worker.scan)
        self.worker.scanned.connect(self.scanned)
        self.sig_connect.connect(self.worker.connect)
        self.sig_start.connect(self.worker.start)
        self.sig_stop.connect(self.worker.stop)
        self.sig_load.connect(self.worker.load)
        self.worker.connected.connect(lambda: self.btn_load.setEnabled(True))
        self.worker.loaded.connect(lambda: self.btn_start.setEnabled(True))
        self.worker.loaded.connect(lambda: self.btn_stop.setEnabled(True))

        layout = QMyVBoxLayout()
        lt = QMyHBoxLayout(self.btn_scan, self.combobox_sn)
        lt.addStretch(0)
        layout.addLayout(lt)
        lt = QMyHBoxLayout(self.btn_connect, self.btn_load, self.btn_start, self.btn_stop)
        lt.addStretch(0)
        layout.addLayout(lt)
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
        for c in self.channels:
            self.settings[c] = {'type': 'dc', 'ampl': 0.0, 'status': self.statuses[c]}
        return self.settings

    @pyqtSlot(name='Connect')
    def connect(self):
        self.worker_thread = QThread()
        self.start_branch(self.worker, self.worker_thread, self.sig_connect, self.get_settings())

    @pyqtSlot(name='Start')
    def start(self):
        for c in self.channels:
            self.statuses[c] = True if self.active[c] else False
        self.worker_thread = QThread()
        self.start_branch(self.worker, self.worker_thread, self.sig_start, self.get_settings())

    @pyqtSlot(name='Stop')
    def stop(self):
        for c in self.channels:
            self.statuses[c] = False
        self.worker_thread = QThread()
        self.start_branch(self.worker, self.worker_thread, self.sig_stop, self.get_settings())

    @pyqtSlot(name='Load')
    def load(self):
        self.worker_thread = QThread()
        self.start_branch(self.worker, self.worker_thread, self.sig_load, self.get_settings())


class RigolGenMainWidget(QWidget):
    def __init__(self):
        super(RigolGenMainWidget, self).__init__()
        self.font_size = 14

        self.controller = RigolGenWidget(font_size=self.font_size)

        layout = QMyVBoxLayout()
        layout.addWidget(self.controller, alignment=Qt.AlignmentFlag.AlignCenter)
        self.setLayout(layout)


class RigolGenWindow(MyStandardWindow):
    def __init__(self):
        super().__init__()
        self.rigol_gen_widget = RigolGenMainWidget()
        font = self.font()
        font.setPointSize(14)
        self.rigol_gen_widget.setFont(font)
        self.rigol_gen_widget.font_size = 14
        self.appear_with_central_widget('Rigol Gen 0 V DC', self.rigol_gen_widget)


if __name__ == '__main__':
    # QLocale.setDefault(QLocale(QLocale.C))
    app = QApplication(sys.argv)
    ex = RigolGenWindow()
    sys.exit(app.exec())