# _*_ encoding:utf-8 _*_
__author__ = 'nzb'
__datetime__ = '2019/5/20 13:59'

import sys

from PyQt5.QtWidgets import QMainWindow, QApplication, QDesktopWidget, QPushButton
from PyQt5 import QtGui
from PyQt5 import QtCore


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
        self.setGeometry(self.left, self.top, self.width, self.height)  # 大小位置设置
        self.setWindowTitle(self.title)  # 窗口标题

        # 按钮
        self.button()

        # 展示窗口
        self.show()

    def button(self):
        """按钮"""
        btn = QPushButton('click me', self)
        # btn.resize(100, 34) # 按钮大小
        # btn.move(290, 550)  # 移动按钮
        # 合并
        btn.setGeometry(QtCore.QRect(300, 250, 150, 34))
        # 按钮图标
        btn.setIcon(QtGui.QIcon('../img/Agt Stop.ico'))
        btn.setIconSize(QtCore.QSize(40, 40))  # 设置图标大小
        # 设置按钮提示
        btn.setToolTip('按钮提示')
        # 触发事件
        btn.clicked.connect(self.ClickMe)

    def ClickMe(self):
        print('Hello World')
        # 退出
        sys.exit()


if __name__ == "__main__":

    app = QApplication(sys.argv)
    ex = UI_demo()
    sys.exit(app.exec_())
