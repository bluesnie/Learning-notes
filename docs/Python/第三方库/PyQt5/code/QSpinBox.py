# _*_ encoding:utf-8 _*_
__author__ = 'nzb'
__datetime__ = '2019/5/21 16:36'


import sys
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QGroupBox, \
    QHBoxLayout, QPushButton, QVBoxLayout, QSpinBox
from PyQt5.QtWidgets import QDial
from PyQt5 import QtGui, QtCore
from PyQt5.QtCore import Qt


class UI_demo(QWidget):
    """用户界面"""
    def __init__(self):
        super().__init__()

        # 窗口信息
        self.title = 'PyQt5 QSpinbox'
        self.left = 600
        self.top = 200
        self.width = 500
        self.height = 500
        self.iconName = '../img/home.ico'

        self.initWindow()

    def initWindow(self):

        # 窗口信息
        self.setWindowIcon(QtGui.QIcon(self.iconName))  # 图标设置
        self.setGeometry(self.left, self.top, self.width, self.height)  # 大小位置设置
        self.setWindowTitle(self.title)  # 窗口标题

        # 转动盒子
        vbox = QVBoxLayout()

        self.spinbox = QSpinBox()
        self.spinbox.valueChanged.connect(self.spin_changed)
        vbox.addWidget(self.spinbox)

        self.label = QLabel()
        self.label.setFont(QtGui.QFont('Sanserif', 15))
        self.label.setAlignment(Qt.AlignCenter)
        vbox.addWidget(self.label)

        self.setLayout(vbox)

        # 展示窗口
        self.show()

    def spin_changed(self):
        spinValue = self.spinbox.value()
        self.label.setText("current value is :" + str(spinValue))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = UI_demo()
    sys.exit(app.exec_())