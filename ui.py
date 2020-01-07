# coding=utf-8

import sys, os
import numpy as np
import cv2

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import QImage,QIcon,QPixmap
from PyQt5.QtWidgets import QApplication, QFileDialog, QMessageBox, QWidget

from utils import FaceDetector, Face


class UIMainWindow(object):
    def __init__(self, window):
        self.window = window
        self.ops = []
        self._setup_ui()

        self.predictor_path = os.path.abspath("./data/shape_predictor_68_face_landmarks.dat")
        self.face_detector = FaceDetector(self.predictor_path)
        self._set_connect()

    def _setup_ui(self):
        '''
            ui设定
        '''
        #! 添加新的按钮的时候注意别和原打开按钮重合了！
        
        self.window.setObjectName("MainWindow")
        self.window.resize(600, 480)
        self.central_widget = QtWidgets.QWidget(self.window)
        self.central_widget.setObjectName("centralWidget")
        self.vertical_layout = QtWidgets.QVBoxLayout(self.central_widget)
        self.vertical_layout.setObjectName("verticalLayout")
        self.sa = QtWidgets.QScrollArea(self.central_widget)
        size_policy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        size_policy.setHorizontalStretch(0)
        size_policy.setVerticalStretch(0)
        size_policy.setHeightForWidth(self.sa.sizePolicy().hasHeightForWidth())
        self.sa.setSizePolicy(size_policy)
        self.sa.setWidgetResizable(True)
        self.sa.setObjectName("sa")
        self.scroll_area_widget_contents = QtWidgets.QWidget()
        self.scroll_area_widget_contents.setGeometry(QtCore.QRect(0, 0, 813, 532))
        self.scroll_area_widget_contents.setObjectName("scrollAreaWidgetContents")
        self.sa.setWidget(self.scroll_area_widget_contents)
        self.vertical_layout.addWidget(self.sa)

        self.label = QtWidgets.QLabel(self.window)
        self.label2 = QtWidgets.QLabel(self.window)
        self.label_layout = QtWidgets.QHBoxLayout(self.sa)
        self.label_layout.addWidget(self.label)
        self.label_layout.addWidget(self.label2)

        self.grid_layout = QtWidgets.QGridLayout()
        self.grid_layout.setObjectName("gridLayout")

        self.bt_whitening = QtWidgets.QPushButton(self.central_widget)
        self.bt_whitening.setObjectName("btWhitening")
        self.grid_layout.addWidget(self.bt_whitening, 0, 0, 1, 1)
        self.sl_whitening = QtWidgets.QSlider(self.central_widget)
        self.sl_whitening.setOrientation(QtCore.Qt.Horizontal)
        self.sl_whitening.setObjectName("slWhitening")
        self.grid_layout.addWidget(self.sl_whitening, 0, 1, 1, 1)
        #spacer_item = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        #self.grid_layout.addItem(spacer_item, 0, 2, 1, 1)
        self.ops.append('whitening')

        self.bt_brightening = QtWidgets.QPushButton(self.central_widget)
        self.bt_brightening.setObjectName("btBrightening")
        self.grid_layout.addWidget(self.bt_brightening, 1, 0, 1, 1)
        self.sl_brightening = QtWidgets.QSlider(self.central_widget)
        self.sl_brightening.setOrientation(QtCore.Qt.Horizontal)
        self.sl_brightening.setObjectName("slBrightening")
        self.grid_layout.addWidget(self.sl_brightening, 1, 1, 1, 1)
        #spacer_item = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        #self.grid_layout.addItem(spacer_item, 1, 2, 1, 1)
        self.ops.append('brightening')

        self.bt_largeeye = QtWidgets.QPushButton(self.central_widget)
        self.bt_largeeye.setObjectName("btLargeeye")
        self.grid_layout.addWidget(self.bt_largeeye, 2, 0, 1, 1)
        self.sl_largeeye = QtWidgets.QSlider(self.central_widget)
        self.sl_largeeye.setOrientation(QtCore.Qt.Horizontal)
        self.sl_largeeye.setObjectName("slLargeeye")
        self.grid_layout.addWidget(self.sl_largeeye, 2, 1, 1, 1)
        #spacer_item = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        #self.grid_layout.addItem(spacer_item, 2, 2, 1, 1)
        self.ops.append('largeeye')

        self.bt_slimface = QtWidgets.QPushButton(self.central_widget)
        self.bt_slimface.setObjectName("btSlimface")
        self.grid_layout.addWidget(self.bt_slimface, 3, 0, 1, 1)
        self.sl_slimface = QtWidgets.QSlider(self.central_widget)
        self.sl_slimface.setOrientation(QtCore.Qt.Horizontal)
        self.sl_slimface.setObjectName("slSlimface")
        self.grid_layout.addWidget(self.sl_slimface, 3, 1, 1, 1)
        #spacer_item = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        #self.grid_layout.addItem(spacer_item, 3, 2, 1, 1)
        self.ops.append('slimface')

        self.bt_open = QtWidgets.QPushButton(self.central_widget)
        self.bt_open.setObjectName("bt_open")
        self.grid_layout.addWidget(self.bt_open, 4, 0, 1, 1)

        self.vertical_layout.addLayout(self.grid_layout)
        self.window.setCentralWidget(self.central_widget)

        self._retranslate()
        QtCore.QMetaObject.connectSlotsByName(self.window)

    def _retranslate(self):
        _translate = QtCore.QCoreApplication.translate
        self.window.setWindowTitle(_translate("MainWindow", "人像美颜"))
        self.bt_whitening.setText(_translate("MainWindow", "美白"))
        self.bt_brightening.setText(_translate("MainWindow", "红唇"))
        self.bt_largeeye.setText(_translate("MainWindow", "大眼"))
        self.bt_slimface.setText(_translate("MainWindow", "瘦脸"))

        self.bt_open.setText(_translate("MainWindow", "打开文件"))

    def _set_connect(self):
        self.bt_open.clicked.connect(self._open_img)
        for op in self.ops:
            self.__getattribute__('bt_' + op).clicked.connect(self.__getattribute__('_' + op))

    def _open_img(self):
        # self.img_path, _ = QFileDialog.getOpenFileName(self.central_widget, '打开图片文件', './',
        #                                               'Image Files(*.png *.jpg *.bmp)')
        self.img_path = './images/1.jpg'
        if self.img_path is None or self.img_path == '':
            return

        self.img_bgr = self.face_detector.get_bgr(self.img_path)
        self.img_hsv = self.face_detector.get_hsv(self.img_path)
        self.tmp_bgr = self.img_bgr.copy()
        self.tmp_hsv = self.img_hsv.copy()
        _, self.landmarks = self.face_detector.get_face_rect_and_landmarks(self.img_path)

        self.face = Face(self.tmp_bgr, self.landmarks)
        self._set_original_img()
        self._set_img()

    def _set_original_img(self):
        height, width, channel = self.img_bgr.shape
        bytesPerLine = 3 * width
        qimage = QImage(cv2.cvtColor(self.img_bgr, cv2.COLOR_BGR2RGB).data, width, height, bytesPerLine, QImage.Format_RGB888)
        self.label.setPixmap(QPixmap.fromImage(qimage))

    def _set_img(self):
        height, width, channel = self.tmp_bgr.shape
        bytesPerLine = 3 * width
        qimage = QImage(cv2.cvtColor(self.tmp_bgr, cv2.COLOR_BGR2RGB).data, width, height, bytesPerLine, QImage.Format_RGB888)
        self.label2.setPixmap(QPixmap.fromImage(qimage))

    def _whitening(self):
        value = min(1, max(self.sl_whitening.value() / 300., 0))
        self.face.whitening(value)
        self._set_img()

    def _brightening(self):
        value = min(1, max(self.sl_brightening.value() / 100, 0))
        self.face.organs['mouth'].brightening(value)
        self._set_img()

    def _largeeye(self):
        # 1.0为推荐value，但允许超过1.0，设置上限
        value = 1.8*min(1, max(self.sl_largeeye.value() / 100, 0))
        self.face.largeeye(value)
        self._set_img()

    def _slimface(self):
        # 1.0为推荐value，但允许超过1.0，设置上限
        value = 1.8*min(1, max(self.sl_slimface.value() / 100, 0))
        self.face.slimface(value)
        self._set_img()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = QtWidgets.QMainWindow()
    ui = UIMainWindow(main_window)
    ui.window.show()

    sys.exit(app.exec_())
