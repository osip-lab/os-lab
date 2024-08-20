from PyQt6.QtWidgets import QWidget, QMainWindow, QApplication, QLabel, QPushButton, QDoubleSpinBox, \
    QRadioButton, QGridLayout, QVBoxLayout, QHBoxLayout, QComboBox, QLineEdit, QGroupBox
from PyQt6.QtCore import Qt, pyqtSignal, QThread, QObject
from PyQt6.QtGui import QFont, QAction, QGuiApplication


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
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)


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


class QMyComboBox(QComboBox):
    def __init__(self, items_list):
        super(QMyComboBox, self).__init__()
        self.addItems(items_list)
        self.adjustSize()


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
        self.exit_action.triggered.connect(QApplication.instance().quit)
        self.addAction(self.exit_action)

    def appear_with_layout(self, desired_name, desired_layout):
        # Set main widget
        central_widget = QWidget()
        central_widget.setLayout(desired_layout)
        self.appear_with_central_widget(desired_name, central_widget)

    def appear_with_central_widget(self, desired_name, desired_widget):
        # Set main widget
        self.setCentralWidget(desired_widget)

        # Set name and show window
        self.setWindowTitle(desired_name)
        self.show()

        # Move window to center
        qr = self.frameGeometry()
        cp = QGuiApplication.primaryScreen().availableGeometry().center()
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

    @staticmethod
    def start_branch(worker: ThreadedWorker, branch: QThread, signal: pyqtSignal, *args):
        worker.moveToThread(branch)
        worker.finished.connect(branch.quit)
        branch.finished.connect(branch.deleteLater)
        branch.start()
        signal.emit(*args)


class QMyLineEdit(QLineEdit):
    def __init__(self, font_size=14):
        super(QMyLineEdit, self).__init__()
        self.setFont((QFont('Arial', font_size)))
