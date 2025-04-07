# basic libraries
import os
import sys
import time
import numpy as np
import pandas as pd
# gui import
from PyQt6.QtWidgets import QApplication, QWidget, QTableWidget, QTableWidgetItem
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, pyqtSlot
# my extension of gui
from qt_gui.qt_ext import MyStandardWindow, ThreadedWidget, QMyHBoxLayout, QMyVBoxLayout, QMyStandardButton
# controllers for other devices
from rigol_gen.rigol_gen_sin_gui import RigolGenSinWidget
from pico_scope.pico_scope_control_gui import PicoControlWidget, SaverWidget
# local path to data bank
from local_config import PATH_DATA_LOCAL

import matplotlib.pyplot as plt


class BodeMeasurerWidget(ThreadedWidget):
    sig_set_freq = pyqtSignal(float, name='SetFreq')
    sig_set_ampl = pyqtSignal(float, name='SetAmpl')
    sig_load_rigol = pyqtSignal(name='LoadRigol')
    sig_start_pico = pyqtSignal(name='StartPico')
    sig_set_timebase = pyqtSignal(float, name='SetTimeBase')
    sig_set_duration = pyqtSignal(float, name='SetDuration')
    sig_set_pre_duration = pyqtSignal(float, name='SetPreDuration')
    sig_set_range = pyqtSignal(str, float, name='SetRange')
    sig_load_pico = pyqtSignal(name='LoadPico')

    def __init__(self, font_size=14):
        super(BodeMeasurerWidget, self).__init__(font_size=font_size)
        self.setTitle('Bode Measurer')

        self.columns = ('freq', 'gen_ampl', 'in_range', 'out_range', 'in_ampl', 'out_ampl', 'gain', 'phase')
        self.observables = ('in_ampl', 'out_ampl', 'gain', 'phase')
        self.df = pd.DataFrame(columns=self.columns)
        self.fdf = None
        self.current = None

        self.btn_measure = QMyStandardButton('measure', font_size=self.font_size)
        self.btn_measure.setToolTip('start measuring Bode Plot')
        self.btn_measure.clicked.connect(self.measure)

        self.btn_save = QMyStandardButton('save', font_size=self.font_size)
        self.btn_save.setToolTip('save Bode Plot')
        self.btn_save.clicked.connect(self.save)

        self.moment = QTimer()
        self.moment.setInterval(10)
        self.moment.setSingleShot(True)
        self.on_wait = None
        self.moment.timeout.connect(self.timeout)

        self.table = QTableWidget()
        self.table.setColumnCount(len(self.columns))
        self.table.setHorizontalHeaderLabels(self.columns)
        font = self.table.horizontalHeader().font()
        font.setPointSize(self.font_size)
        self.table.horizontalHeader().setFont(font)
        self.table.setFixedWidth(30 + 100 * len(self.columns))
        self.table.setFixedHeight(40 + 30 * 10)

        # generate test table
        for i, f in enumerate((100.0, 300.0, 1000.0, 3000.0)):
            self.add_row(freq=f, gen_ampl=1.0, in_range=1.0, out_range=1.0)
        self.add_row()  # extra row for new entry

        self.table.model().dataChanged.connect(self.check_data)

        layout = QMyVBoxLayout(alignment=Qt.AlignmentFlag.AlignCenter)
        lt = QMyHBoxLayout(self.btn_measure, self.btn_save, alignment=Qt.AlignmentFlag.AlignCenter)
        lt.addStretch(0)
        layout.addLayout(lt)
        layout.addWidget(self.table)
        self.setLayout(layout)

    def check_data(self, i):
        try:
            v = str(float(i.data()))
            self.table.item(i.row(), i.column()).setText(v)
            self.df.loc[i.row(), self.columns[i.column()]] = float(v)
            if i.row() == self.table.rowCount() - 1:
                self.add_row()
        except ValueError:
            v = self.df.loc[i.row(), self.columns[i.column()]]
            if pd.isna(v):
                self.table.item(i.row(), i.column()).setText('')
            else:
                self.table.item(i.row(), i.column()).setText(str(float(v)))

    def add_row(self, **data):

        self.df.loc[len(self.df)] = data

        self.table.setRowCount(self.table.rowCount() + 1)

        r = self.table.rowCount() - 1

        for k, v in data.items():
            self.table.setItem(r, self.columns.index(k), QTableWidgetItem(str(float(v))))

        for col in self.observables:
            item = QTableWidgetItem('')
            item.setFlags(item.flags() & Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(r, self.columns.index(col), item)

    def fix_table(self):
        self.fdf = self.df.copy()
        for c in set(self.columns) - set(self.observables):
            self.fdf = self.fdf[self.fdf[c].notna()]
        print(self.fdf)

    @pyqtSlot(name='Measure')
    def measure(self):
        self.fix_table()
        self.current = -1
        self.start_cycle()

    def start_cycle(self):
        self.current += 1
        if self.current == len(self.fdf):
            self.fdf = None
            self.current = None
            print('measure finished')
        else:
            self.load_rigol()

    def wait_and_emit(self, signal):
        self.on_wait = signal
        self.moment.start()

    def timeout(self):
        self.on_wait.emit()

    def load_rigol(self):
        if self.fdf is not None:
            print(f"load and start {self.fdf.loc[self.current]['freq']}")
            self.sig_set_freq.emit(float(self.fdf.loc[self.current]['freq']))
            self.sig_set_ampl.emit(float(self.fdf.loc[self.current]['gen_ampl']))
            self.wait_and_emit(self.sig_load_rigol)

    @pyqtSlot(name='LoadPico')
    def load_pico(self):
        if self.fdf is not None:
            periods = 10
            points = 100
            freq = float(self.fdf.loc[self.current]['freq'])
            self.sig_set_timebase.emit(1 / freq / points)
            self.sig_set_duration.emit(periods / freq)
            self.sig_set_pre_duration.emit(periods / 2 / freq)
            for ch, col in zip(('1', '2'), ('in_range', 'out_range')):
                self.sig_set_range.emit(ch, float(self.fdf.loc[self.current][col]))
            self.wait_and_emit(self.sig_load_pico)

    @pyqtSlot(name='StartPico')
    def start_pico(self):
        if self.fdf is not None:
            self.wait_and_emit(self.sig_start_pico)

    @pyqtSlot(dict, name='Fit')
    def fit(self, signals):

        if self.fdf is not None:

            freq = float(self.fdf.loc[self.current]['freq'])
            t = signals['time']
            signal = signals['1']
            response = signals['2']

            print(f'sample length {len(t)} points')

            ampl1, phase1 = calculate_spectral_point(t, signal, freq)
            ampl2, phase2 = calculate_spectral_point(t, response, freq)

            self.table.item(self.current, self.columns.index('in_ampl')).setText(f'{ampl1:.3f}')
            self.df.loc[self.current, 'in_ampl'] = float(ampl1)
            self.table.item(self.current, self.columns.index('out_ampl')).setText(f'{ampl2:.3f}')
            self.df.loc[self.current, 'out_ampl'] = float(ampl2)

            gain = 20 * np.log10(ampl2 / ampl1)
            phase = phase2 - phase1
            self.table.item(self.current, self.columns.index('gain')).setText(f'{gain:.1f}')
            self.df.loc[self.current, 'gain'] = float(gain)
            self.table.item(self.current, self.columns.index('phase')).setText(f'{phase:.1f}')
            self.df.loc[self.current, 'phase'] = float(phase)
            print(f'freq = {freq:.1e} Hz, ampl1 = {ampl1:.3e}, ampl2 = {ampl2:.3e}, phase1 = {phase1:.3f}°, phase2 = {phase2:.3f}°, gain = {gain:.3f} dBV, delta phase = {phase:.3f}°')

            fig, ax = plt.subplots(1, 1, figsize=(5, 3), dpi=300)
            ax.plot(t, signal)
            ax.plot(t, response)
            ax.grid()
            fig.savefig(os.path.join(PATH_DATA_LOCAL, 'bode_plotter', time.strftime('%Y-%m-%d %H-%M-%S') + f' f={int(freq)}.png'))
            plt.close(fig)

            self.start_cycle()

    @pyqtSlot()
    def save(self):
        self.fix_table()
        self.fdf.to_excel(os.path.join(PATH_DATA_LOCAL, 'bode_plotter', time.strftime('%Y-%m-%d %H-%M-%S') + '.xlsx'), index=False)


def calculate_spectral_point(t, x, f):
    a = np.sum(np.sin(2 * np.pi * f * t) * x)
    b = np.sum(np.cos(2 * np.pi * f * t) * x)
    ampl = (a ** 2 + b ** 2) ** 0.5 / len(t) * 2
    phase = np.rad2deg(np.arctan(b / a))
    return ampl, phase


class MainWidget(QWidget):

    def __init__(self):
        super(MainWidget, self).__init__()
        self.font_size = 14
        self.channels = tuple('1234')

        self.main_controller = BodeMeasurerWidget(font_size=self.font_size)
        self.rigol_controller = RigolGenSinWidget(font_size=self.font_size)
        self.pico_controller = PicoControlWidget(font_size=self.font_size, channels=self.channels)
        self.saver = SaverWidget(font_size=self.font_size)

        # load settings to rigol generator and get its answer
        self.main_controller.sig_set_freq.connect(self.rigol_controller.set_sin_freq)
        self.main_controller.sig_set_ampl.connect(self.rigol_controller.set_sin_ampl)
        self.main_controller.sig_load_rigol.connect(self.rigol_controller.load)
        self.rigol_controller.sig_loaded.connect(self.main_controller.load_pico)

        # load settings to pico scope and get its answer
        self.main_controller.sig_set_timebase.connect(self.pico_controller.set_timebase)
        self.main_controller.sig_set_duration.connect(self.pico_controller.set_duration)
        self.main_controller.sig_set_pre_duration.connect(self.pico_controller.set_pre_duration)
        self.main_controller.sig_set_range.connect(self.pico_controller.set_range)
        self.main_controller.sig_load_pico.connect(self.pico_controller.ps_load)
        self.pico_controller.pico_loaded.connect(self.main_controller.start_pico)

        # start picoscope and get data
        self.main_controller.sig_start_pico.connect(self.pico_controller.ps_start)
        self.pico_controller.plot_signals.connect(self.main_controller.fit)

        # connections to enable settings saving
        self.saver.auto_save_changed.connect(self.pico_controller.set_auto_save)
        self.saver.claim_settings.connect(self.pico_controller.provide_settings)
        self.pico_controller.save_settings.connect(self.saver.save_settings)
        self.saver.load_settings.connect(self.pico_controller.load_settings)

        self.saver.load_defaults()

        layout = QMyVBoxLayout()
        layout.addWidget(self.pico_controller)
        lt = QMyHBoxLayout()
        lt.addLayout(QMyVBoxLayout(self.saver, self.rigol_controller))
        lt.addWidget(self.main_controller)
        layout.addLayout(lt)
        self.setLayout(layout)


class MainWindow(MyStandardWindow):

    def __init__(self):
        super().__init__()
        font = self.font()
        font.setPointSize(14)
        self.widget = MainWidget()
        self.widget.setFont(font)
        self.widget.font_size = 14
        self.appear_with_central_widget('Bode Plot Meter', self.widget)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MainWindow()
    sys.exit(app.exec())
