# _*_ encoding:utf-8 _*_
__author__ = 'nzb'
__datetime__ = '2019/5/20 14:11'

import sys
from PyQt5.QtWidgets import QApplication, QDesktopWidget, QDialog, QPushButton, QVBoxLayout, QGroupBox, QHBoxLayout
from PyQt5 import QtGui
from PyQt5 import QtCore


class UI_demo(QDialog):
    """用户界面"""
    def __init__(self):
        super().__init__()

        # 窗口信息
        self.title = 'PyQt5 Layout Managment'
        self.left = 600
        self.top = 200
        self.width = 300
        self.height = 100

        self.initWindow()

    def initWindow(self):

        # 窗口信息
        self.setWindowIcon(QtGui.QIcon('../img/home.ico'))  # 图标设置
        self.setGeometry(self.left, self.top, self.width, self.height)  # 大小位置设置
        self.setWindowTitle(self.title)  # 窗口标题

        # 布局
        self.createLayout()
        vbox = QVBoxLayout()
        vbox.addWidget(self.groupBox)
        self.setLayout(vbox)

        # 展示窗口
        self.show()

    def createLayout(self):
        """垂直布局和水平布局"""
        self.groupBox = QGroupBox('What is your favorite sport?')
        hboxlayout = QHBoxLayout()

        btn = QPushButton('Soccer', self)
        btn.setIcon(QtGui.QIcon('../img/Soccer.ico'))
        btn.setIconSize(QtCore.QSize(40, 40))
        btn.setMinimumHeight((40))
        hboxlayout.addWidget(btn)

        btn1 = QPushButton('Tennis', self)
        btn1.setIcon(QtGui.QIcon('../img/Tennis.ico'))
        btn1.setIconSize(QtCore.QSize(40, 40))
        btn1.setMinimumHeight((40))
        hboxlayout.addWidget(btn1)

        btn2 = QPushButton('Basketball', self)
        btn2.setIcon(QtGui.QIcon('../img/Basketball.ico'))
        btn2.setIconSize(QtCore.QSize(40, 40))
        btn2.setMinimumHeight((40))
        hboxlayout.addWidget(btn2)
        self.groupBox.setLayout(hboxlayout)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = UI_demo()
    sys.exit(app.exec_())
