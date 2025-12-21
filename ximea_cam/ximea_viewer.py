import os
import sys
import time
import numpy as np
import logging

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QWidget, QLabel
from PyQt6.QtGui import QImage, QPixmap

from qt_gui.qt_ext import MyStandardWindow, QMyVBoxLayout
from ximea_cam import XimeaCamControlWidget

from local_config import PATH_DATA_LOCAL


# ---------------- Plotter ---------------- #
class Plotter(QLabel):
    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(512, 512)
        # self.tic = None
        # self.i = -3

    def update_image(self, frame: np.ndarray):
        # Convert NumPy array to QImage without copying
        h, w = frame.shape
        img = QImage(frame.data, w, h, w * 2, QImage.Format.Format_Grayscale16)
        self.setPixmap(QPixmap.fromImage(img))
        # if self.i <= 0:
        #     self.tic = time.time()
        # else:
        #     print(f'Effective FPS: {self.i / (time.time() - self.tic):.1f}')
        # self.i += 1


class MainWidget(QWidget):

    def __init__(self):
        super(MainWidget, self).__init__()
        self.font_size = 14

        self.controller = XimeaCamControlWidget(font_size=self.font_size)
        self.plotter = Plotter()

        self.controller.new_frame.connect(self.plotter.update_image)

        layout = QMyVBoxLayout(self.controller, self.plotter, alignment=Qt.AlignmentFlag.AlignCenter)
        self.setLayout(layout)


class MainWindow(MyStandardWindow):

    def __init__(self):
        super().__init__()
        font = self.font()
        font.setPointSize(14)
        self.widget = MainWidget()
        self.widget.setFont(font)
        self.widget.font_size = 14
        self.appear_with_central_widget('Image', self.widget)


if __name__ == '__main__':

    dir_name = 'camera_viewer'
    dir_name = os.path.join(PATH_DATA_LOCAL, dir_name)
    os.makedirs(dir_name, exist_ok=True)
    start_time = time.strftime('%Y-%m-%d %H-%M-%S')
    logging.basicConfig(filename=os.path.join(dir_name, f'{start_time} log.txt'),
                        level=logging.INFO, format='%(asctime)s.%(msecs)03d: %(message)s', datefmt='%Y.%m.%d %H:%M:%S')

    app = QApplication(sys.argv)
    ex = MainWindow()
    sys.exit(app.exec())
