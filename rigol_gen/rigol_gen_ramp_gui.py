import sys

from PyQt6.QtWidgets import QApplication, QWidget
from PyQt6.QtCore import Qt, pyqtSlot

from qt_gui.qt_ext import MyStandardWindow, QCenteredLabel, QMyHBoxLayout, QMyVBoxLayout, QMySpinBox
from rigol_gen.rigol_gen_gui import RigolGenWidget


class RigolGenRampBiasWidget(RigolGenWidget):
    def __init__(self, font_size=14):
        super(RigolGenRampBiasWidget, self).__init__(font_size=font_size)

        self.setTitle('Rigol Gen Ramp and Bias')

        self.ramp_ampl = QMySpinBox(v_min=1.0, v_max=20.0, v_ini=20.0, decimals=1, step=1.0, suffix=' V')
        self.ramp_freq = QMySpinBox(v_min=5.0, v_max=50.0, v_ini=10.0, decimals=1, step=5.0, suffix=' Hz')
        self.temp_bias = QMySpinBox(v_min=-200.0, v_max=200.0, v_ini=0.0, decimals=1, step=1.0, suffix=' mV')
        self.temp_bias.valueChanged.connect(self.load)

        self.layout().addLayout(QMyHBoxLayout(QCenteredLabel('Ch1 Ramp'), self.ramp_ampl, self.ramp_freq))
        lt = QMyHBoxLayout(QCenteredLabel('Ch2 DC'), self.temp_bias)
        lt.addStretch(0)
        self.layout().addLayout(lt)

    def get_settings(self):
        self.settings = super().get_settings()
        self.settings['ch1'] = {'type': 'ramp', 'freq': self.ramp_freq.value(), 'ampl': self.ramp_ampl.value(), 'status': self.statuses['ch1']}
        self.settings['ch2'] = {'type': 'dc', 'ampl': self.temp_bias.value() * 1e-3, 'status': self.statuses['ch2']}
        return self.settings

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

        self.controller = RigolGenRampBiasWidget(font_size=self.font_size)

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
        self.appear_with_central_widget('Rigol Gen Ramp and Bias', self.rigol_gen_widget)


if __name__ == '__main__':
    # QLocale.setDefault(QLocale(QLocale.C))
    app = QApplication(sys.argv)
    ex = RigolGenWindow()
    sys.exit(app.exec())
