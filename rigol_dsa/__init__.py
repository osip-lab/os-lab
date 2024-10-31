import sys
import time
import numpy as np
import pyvisa

from PyQt6.QtWidgets import QApplication, QWidget
from PyQt6.QtCore import QThread, pyqtSignal, pyqtSlot

from qt_gui.qt_ext import (MyStandardWindow, ThreadedWidget, ThreadedWorker, QMyVBoxLayout, QMyHBoxLayout,
                           QMyStandardButton)


class RigolDSAWorker(ThreadedWorker):
    connected = pyqtSignal(name='Connected')
    measured = pyqtSignal(dict, name='Measured')

    def __init__(self, thread):
        super(RigolDSAWorker, self).__init__(thread)

        self.dsa = None

    @pyqtSlot(name='Connect')
    def connect(self):

        rm = pyvisa.ResourceManager()
        rl = rm.list_resources()

        if len(list(filter(lambda x: 'DSA' in x, rl))) == 1:
            self.dsa = rm.open_resource(list(filter(lambda x: 'DSA' in x, rl))[0])
        else:
            raise ValueError

        self.finish(self.connected)

    @pyqtSlot(name='Start')
    def start(self):

        fs = float(self.dsa.query(':SENS:FREQ:START?'))
        ff = float(self.dsa.query(':SENS:FREQ:STOP?'))

        self.dsa.write(':INIT')

        while int(self.dsa.query(':STATUS:OPERATION:CONDITION?')) != 0:
            time.sleep(0.1)

        data = self.dsa.query(':TRACE:DATA? TRACE1')
        data = data.split(', ')
        data[0] = data[0].split()[1]
        ampl = np.array(data, dtype='float64')

        freq = np.linspace(fs, ff, len(ampl))

        self.finish(self.measured, {'ampl': ampl, 'freq': freq})


class RigolDSAWidget(ThreadedWidget):
    sig_connect = pyqtSignal(name='Connect')
    sig_start = pyqtSignal(name='Start')
    sig_measured = pyqtSignal(dict, name='Measured')

    def __init__(self, font_size=14):
        super(RigolDSAWidget, self).__init__(font_size=font_size)
        self.setTitle('Rigol DSA Controller')

        self.btn_connect = QMyStandardButton('connect', font_size=self.font_size)
        self.btn_connect.setToolTip('connect to a device')
        self.btn_connect.clicked.connect(self.connect)

        self.btn_start = QMyStandardButton('start', font_size=self.font_size)
        self.btn_start.setToolTip('start a single sweep')
        self.btn_start.clicked.connect(self.start)

        self.worker = RigolDSAWorker(self.thread())
        self.worker_thread = None
        self.sig_connect.connect(self.worker.connect)
        self.sig_start.connect(self.worker.start)
        self.worker.measured.connect(self.sig_measured)
        self.worker.measured.connect(lambda: self.btn_start.setEnabled(True))

        self.layout = QMyHBoxLayout(self.btn_connect, self.btn_start)
        self.setLayout(self.layout)

    @pyqtSlot(name='Connect')
    def connect(self):
        self.worker_thread = QThread()
        self.start_branch(self.worker, self.worker_thread, self.sig_connect)

    @pyqtSlot(name='Start')
    def start(self):
        self.btn_start.setEnabled(False)
        self.worker_thread = QThread()
        self.start_branch(self.worker, self.worker_thread, self.sig_start)


class MainWidget(QWidget):

    def __init__(self):
        super(MainWidget, self).__init__()
        self.font_size = 14

        self.controller = RigolDSAWidget(font_size=self.font_size)

        layout = QMyVBoxLayout(self.controller)
        self.setLayout(layout)


class MainWindow(MyStandardWindow):

    def __init__(self):
        super().__init__()
        font = self.font()
        font.setPointSize(14)
        self.widget = MainWidget()
        self.widget.setFont(font)
        self.widget.font_size = 14
        self.appear_with_central_widget('Rigol DSA Controller', self.widget)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MainWindow()
    sys.exit(app.exec())
