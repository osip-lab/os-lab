import os
import time
import numpy as np
from PyQt5.QtWidgets import QWidget, QMainWindow, QAction, qApp, QLabel, QDesktopWidget, QPushButton, QDoubleSpinBox,\
    QRadioButton, QGridLayout, QVBoxLayout, QHBoxLayout, QComboBox, QSizePolicy, QLineEdit, QGroupBox, QCheckBox
from PyQt5.QtCore import Qt, pyqtSignal, QThread, QObject, QTimer
from PyQt5.QtGui import QFont
from PyQt5.QtMultimedia import QSound


class QMergedRadioButton(QWidget):

    def __init__(self, parent=None, layout='v', option_list=('option 1', 'option 2')):
        super(QMergedRadioButton, self).__init__(parent)

        self.button_list = []
        for option in option_list:
            self.button_list.append(QRadioButton(option))
        self.button_list[0].setChecked(True)
        for button in self.button_list:
            button.clicked.connect(self.emit_signal)

        if layout == 'v':
            main_layout = QVBoxLayout()
            for button in self.button_list:
                main_layout.addWidget(button)
        elif layout == 'h':
            main_layout = QHBoxLayout()
            for button in self.button_list:
                main_layout.addWidget(button)
        elif 'x' in layout:
            rows, cols = list(map(int, layout.split('x')))
            position_list = []
            for i in range(rows):
                for j in range(rows):
                    position_list.append((i, j))
            main_layout = QGridLayout()
            for button, position in zip(self.button_list, position_list):
                main_layout.addWidget(button, *position)
        else:
            print(f'Wrong directions for QMergedRadioButton with options {option_list}!\n'
                  f'Using vertical direction!')
            main_layout = QVBoxLayout()
            for button in self.button_list:
                main_layout.addWidget(button)
        self.setLayout(main_layout)

    option_changed = pyqtSignal(str)

    def emit_signal(self):
        self.option_changed.emit(self.get_option())

    def get_option(self):
        for button in self.button_list:
            if button.isChecked():
                return button.text()

    def set_option(self, option_text):
        for button in self.button_list:
            if button.text() == option_text:
                button.setChecked(True)


class QCenteredLabel(QLabel):
    def __init__(self, text):
        super().__init__()
        self.setText(text)
        self.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
        self.setAlignment(Qt.AlignCenter)


class QMySpinBox(QDoubleSpinBox):
    def __init__(self, v_min=-1000000, v_max=1000000, v_ini=0, decimals=2, step=1, prefix='', suffix=' units'):
        super().__init__()
        self.setMaximum(v_max)
        self.setMinimum(v_min)
        self.setDecimals(decimals)
        self.setValue(v_ini)
        self.setSingleStep(step)
        self.setPrefix(prefix)
        self.setSuffix(suffix)


class QMyStandardButton(QPushButton):
    def __init__(self, *args, font_size=8):
        super().__init__(*args)
        self.setFont(QFont('Arial', font_size))
        self.setMaximumWidth(self.fontMetrics().boundingRect(self.text()).width() + 10)
        self.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)


class QMyComboBox(QComboBox):
    def __init__(self, items_list):
        super(QMyComboBox, self).__init__()
        self.addItems(items_list)
        self.adjustSize()
        self.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)


class QMyVBoxLayout(QVBoxLayout):
    def __init__(self, *args, **kwargs):
        super().__init__()
        for arg in args:
            self.addWidget(arg, **kwargs)


class QMyHBoxLayout(QHBoxLayout):
    def __init__(self, *args, **kwargs):
        super().__init__()
        for arg in args:
            self.addWidget(arg, **kwargs)


class MyStandardWindow(QMainWindow):

    def __init__(self):
        super().__init__()

        # Create exit shortcut
        self.exit_action = QAction('Exit', self)
        self.exit_action.setShortcut('Esc')
        self.exit_action.triggered.connect(qApp.quit)
        self.addAction(self.exit_action)

    def appear_with_layout(self, desired_name, desired_layout):
        # Set main widget
        central_window = QWidget()
        central_window.setLayout(desired_layout)
        self.setCentralWidget(central_window)

        # Set name and show window
        self.setWindowTitle(desired_name)
        self.show()

        # Move window to center
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def appear_with_central_widget(self, desired_name, desired_widget):
        # Set main widget
        self.setCentralWidget(desired_widget)

        # Set name and show window
        self.setWindowTitle(desired_name)
        self.show()

        # Move window to center
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())


class ThreadedWorker(QObject):
    finished = pyqtSignal()

    def __init__(self, thread: QThread):
        super(ThreadedWorker, self).__init__()
        self.main_thread = thread

    def finish(self, signal: pyqtSignal, *args):
        self.moveToThread(self.main_thread)
        signal.emit(*args)
        self.finished.emit()


class ThreadedWidget(QGroupBox):
    def __init__(self, font_size=14):
        super(ThreadedWidget, self).__init__()
        self.font_size = font_size
        font = self.font()
        font.setPointSize(font_size)
        self.setFont(font)
        self.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)

    @staticmethod
    def start_branch(worker: ThreadedWorker, branch: QThread, signal: pyqtSignal, *args):
        worker.moveToThread(branch)
        worker.finished.connect(branch.quit)
        branch.finished.connect(branch.deleteLater)
        branch.start()
        signal.emit(*args)


class TimerWidget(QGroupBox):
    def __init__(self, font_size=14, duration=60.0, warning=None):
        super(TimerWidget, self).__init__()
        self.font_size = font_size
        font = self.font()
        font.setPointSize(self.font_size)
        self.setFont(font)

        self.setTitle('Timer')
        self.btn_start = QMyStandardButton('start', font_size=self.font_size)
        self.btn_start.clicked.connect(self.start)
        self.btn_stop = QMyStandardButton('stop', font_size=self.font_size)
        self.btn_stop.clicked.connect(self.stop)
        self.btn_reset = QMyStandardButton('reset', font_size=self.font_size)
        self.btn_reset.clicked.connect(self.reset)
        self.switch = QCheckBox('report')

        self.duration = duration
        self.warning = warning
        self.counter = self.duration

        self.time_pace = int(10.0)  # us
        self.timer = QTimer()
        self.timer.setInterval(self.time_pace)
        self.timer.timeout.connect(self.change)
        self.tic = time.time()

        self.sound_end = QSound(os.path.join(os.getcwd(), 'sounds', 'end.wav'))
        self.sound_warning = QSound(os.path.join(os.getcwd(), 'sounds', 'warning.wav'))
        self.lbl = QCenteredLabel(f'{self.counter:.2f}')
        self.lbl.setFont((QFont('Arial', self.font_size)))

        self.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
        main_layout = QMyHBoxLayout(self.btn_start, self.btn_stop, self.btn_reset, self.lbl, self.switch)
        main_layout.addStretch(0)
        self.setLayout(main_layout)

    def start(self):
        self.tic = time.time()
        self.timer.start()

    def stop(self):
        self.timer.stop()
        self.counter = self.duration
        self.lbl.setText(f'{self.counter:.2f}')

    def reset(self):
        self.stop()
        self.start()

    def change(self):
        if self.counter == 0:
            self.sound_end.play()
            self.counter = self.duration
        else:
            if self.warning is not None:
                if self.counter == self.warning:
                    self.sound_warning.play()
            if (self.counter - self.counter // 1.0) < (self.time_pace / 1000 / 2):
                if self.switch.isChecked():
                    print(f'{self.duration - self.counter:.2f}', f'{time.time() - self.tic:.2f}',
                          f'{(time.time() - self.tic) - (self.duration - self.counter):.2f}')
            self.counter = np.round(self.counter - self.time_pace / 1000, decimals=2)
        self.lbl.setText(f'{self.counter:.2f}')


class QMyLineEdit(QLineEdit):
    def __init__(self, font_size=14):
        super(QMyLineEdit, self).__init__()
        self.setFont((QFont('Arial', font_size)))
        self.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)