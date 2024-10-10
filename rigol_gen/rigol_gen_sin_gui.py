import sys

from PyQt6.QtWidgets import QApplication, QWidget
from PyQt6.QtCore import Qt, pyqtSlot

from qt_gui.qt_ext import MyStandardWindow, QCenteredLabel, QMyHBoxLayout, QMyVBoxLayout, QMySpinBox
from rigol_gen.rigol_gen_gui import RigolGenWidget


class RigolGenSinWidget(RigolGenWidget):
    def __init__(self, font_size=14):
        super(RigolGenSinWidget, self).__init__(font_size=font_size)

        self.setTitle('Rigol Gen Sin')

        self.active = {'ch1': True, 'ch2': False}

        self.sin_freq = QMySpinBox(v_min=1.0, v_max=99999.0, v_ini=100.0, decimals=1, step=50.0, suffix=' Hz')
        self.sin_ampl = QMySpinBox(v_min=0.1, v_max=1.0, v_ini=0.5, decimals=1, step=1.0, suffix=' V')

        lt = QMyHBoxLayout(QCenteredLabel('Ch1 Sin'), self.sin_freq, self.sin_ampl)
        lt.addStretch(0)
        self.layout().addLayout(lt)

    def get_settings(self):
        self.settings = super().get_settings()
        self.settings['ch1'] = {'type': 'sin', 'freq': self.sin_freq.value(), 'ampl': self.sin_ampl.value(), 'status': self.statuses['ch1']}
        return self.settings

    @pyqtSlot(float, name='SetSinFreq')
    def set_sin_freq(self, f: float):
        self.sin_freq.setValue(f)

    @pyqtSlot(float, name='SetSinAmpl')
    def set_sin_ampl(self, v: float):
        self.sin_ampl.setValue(v)


class RigolGenMainWidget(QWidget):
    def __init__(self):
        super(RigolGenMainWidget, self).__init__()
        self.font_size = 14

        self.controller = RigolGenSinWidget(font_size=self.font_size)

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
        self.appear_with_central_widget('Rigol Gen Sin', self.rigol_gen_widget)


if __name__ == '__main__':
    # QLocale.setDefault(QLocale(QLocale.C))
    app = QApplication(sys.argv)
    ex = RigolGenWindow()
    sys.exit(app.exec())
