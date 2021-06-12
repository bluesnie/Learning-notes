# _*_ encoding:utf-8 _*_
__author__ = 'nzb'
__datetime__ = '2019/5/20 16:53'

import sys
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QGroupBox, QCheckBox, QHBoxLayout, QPushButton, QButtonGroup
from PyQt5.QtWidgets import QVBoxLayout, QRadioButton
from PyQt5 import QtGui, QtCore


class UI_demo(QWidget):
    """用户界面"""
    def __init__(self):
        super().__init__()

        # 窗口信息
        self.title = 'PyQt5 Groupbox'
        self.left = 600
        self.top = 200
        self.width = 500
        self.height = 200
        self.iconName = '../img/home.ico'

        self.initWindow()

    def initWindow(self):

        # 窗口信息
        self.setWindowIcon(QtGui.QIcon(self.iconName))  # 图标设置
        self.setGeometry(self.left, self.top, self.width, self.height)  # 大小位置设置
        self.setWindowTitle(self.title)  # 窗口标题

        # 布局组
        # 水平布局
        hbox = QHBoxLayout()

        groupbox = QGroupBox('select  you favorite sport')
        groupbox.setFont(QtGui.QFont('Sanserif', 15))
        hbox.addWidget(groupbox)

        # 垂直布局
        vbox = QVBoxLayout()

        rad1 = QRadioButton('soccer')
        vbox.addWidget(rad1)

        rad2 = QRadioButton('tennis')
        vbox.addWidget(rad2)

        groupbox.setLayout(vbox)
        self.setLayout(hbox)

        # 展示窗口
        self.show()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = UI_demo()
    sys.exit(app.exec_())


