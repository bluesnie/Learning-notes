# _*_ encoding:utf-8 _*_
__author__ = 'nzb'
__datetime__ = '2019/5/20 13:49'

import sys

from PyQt5.QtWidgets import QMainWindow, QApplication, QDesktopWidget
from PyQt5 import QtGui


class UI_demo(QMainWindow):
    """用户界面"""
    def __init__(self):
        super().__init__()

        # 窗口信息
        self.title = 'PyQt5 demo'
        self.left = 600
        self.top = 200
        self.width = 800
        self.height = 600

        self.initWindow()

    def initWindow(self):

        # 窗口信息
        self.setWindowIcon(QtGui.QIcon('../img/home.ico'))  # 图标设置

        # self.setGeometry(self.left, self.top, self.width, self.height)  # 大小位置设置
        self.setWindowTitle(self.title)  # 窗口标题

        self.resize(800, 600)  # 窗口大小
        self.center()  # 窗口居中

        # 展示窗口
        self.show()

    def center(self):
        """窗口居中"""
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())


if __name__ == "__main__":

    app = QApplication(sys.argv)
    ex = UI_demo()
    sys.exit(app.exec_())
