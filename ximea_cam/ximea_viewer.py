import os
import sys
import time
import numpy as np
import logging
from collections import deque

from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot
from PyQt6.QtWidgets import QApplication, QWidget, QLabel
from PyQt6.QtGui import QImage, QPixmap

from qt_gui.qt_ext import MyStandardWindow, QMyVBoxLayout
from ximea_cam import XimeaCamControlWidget

from local_config import PATH_DATA_LOCAL

fps_counter_flag = False


class FPSCounter(QLabel):
    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.times = deque(maxlen=10)
        self.setText('Effective FPS:')

    def update_fps(self):
        if fps_counter_flag:
            self.times.append(time.time())
            if len(self.times) == self.times.maxlen:
                fps = (len(self.times) - 1) / (self.times[-1] - self.times[0])
                self.setText(f'Effective FPS: {fps:.1f}')


# ---------------- Plotter ---------------- #
class Plotter(QLabel):
    sig_image_size_changed = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(512, 512)

    def update_image(self, frame: np.ndarray):
        # Convert NumPy array to QImage without copying
        h, w = frame.shape
        if self.width() != w or self.height() != h:
            self.setMinimumSize(w, h)
            self.resize(w, h)
            self.sig_image_size_changed.emit()
        img = QImage(frame.data, w, h, w * 2, QImage.Format.Format_Grayscale16)
        self.setPixmap(QPixmap.fromImage(img))


class MainWidget(QWidget):

    def __init__(self):
        super(MainWidget, self).__init__()
        self.font_size = 14

        self.controller = XimeaCamControlWidget(font_size=self.font_size)
        if fps_counter_flag:
            self.fps_counter = FPSCounter()
        self.plotter = Plotter()

        self.controller.new_frame.connect(self.plotter.update_image)
        if fps_counter_flag:
            self.controller.new_frame.connect(self.fps_counter.update_fps)
        self.plotter.sig_image_size_changed.connect(self.adjust_window)

        layout = QMyVBoxLayout(self.controller, alignment=Qt.AlignmentFlag.AlignCenter)
        if fps_counter_flag:
            layout.addWidget(self.fps_counter)
        layout.addWidget(self.plotter)
        self.setLayout(layout)

    @pyqtSlot(name='AdjustWindow')
    def adjust_window(self):
        self.window().adjustSize()


class MainWindow(MyStandardWindow):

    def __init__(self):
        super().__init__()
        # font = self.font()
        # font.setPointSize(14)
        self.widget = MainWidget()
        self.widget.setFont(font)
        # self.widget.font_size = 14
        self.appear_with_central_widget('Ximea Viewer', self.widget)


if __name__ == '__main__':

    dir_name = 'camera_viewer'
    dir_name = os.path.join(PATH_DATA_LOCAL, dir_name)
    os.makedirs(dir_name, exist_ok=True)
    start_time = time.strftime('%Y-%m-%d %H-%M-%S')
    logging.basicConfig(filename=os.path.join(dir_name, f'{start_time} log.txt'),
                        level=logging.INFO, format='%(asctime)s.%(msecs)03d: %(message)s', datefmt='%Y.%m.%d %H:%M:%S')

    app = QApplication(sys.argv)
    font = app.font()
    font.setPointSize(14)
    app.setFont(font)
    ex = MainWindow()
    sys.exit(app.exec())
