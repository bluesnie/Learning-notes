# _*_ encoding:utf-8 _*_
__author__ = 'nzb'
__datetime__ = '2019/5/21 17:53'

import sys
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QGroupBox, \
    QHBoxLayout, QPushButton, QVBoxLayout, QSpinBox, QLCDNumber
from PyQt5.QtWidgets import QDial, QToolBox, QMenuBar, QAction, qApp
from PyQt5 import QtGui, QtCore
from PyQt5.QtCore import Qt, QThread, pyqtSignal


class UI_demo(QWidget):
    """用户界面"""
    def __init__(self):
        super().__init__()

        # 窗口信息
        self.title = 'PyQt5 MenuBar'
        self.left = 600
        self.top = 200
        self.width = 440
        self.height = 400
        self.iconName = '../img/home.ico'

        self.initWindow()

    def initWindow(self):

        # 窗口信息
        self.setWindowIcon(QtGui.QIcon(self.iconName))  # 图标设置
        self.setGeometry(self.left, self.top, self.width, self.height)  # 大小位置设置
        self.setWindowTitle(self.title)  # 窗口标题

        # 生成菜单栏
        self.CreateMenu()

        # 展示窗口
        self.show()

    def CreateMenu(self):
        """菜单栏"""
        mainMenu = self.menuBar()
        fileMenu = mainMenu.addMenu('File')
        editMenu = mainMenu.addMenu('Edit')
        viewMenu = mainMenu.addMenu('View')
        helpMenu = mainMenu.addMenu('Help')



if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = UI_demo()
    sys.exit(app.exec_())
