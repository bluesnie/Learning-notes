# _*_ encoding:utf-8 _*_
__author__ = 'nzb'
__datetime__ = '2019/5/20 14:37'

import sys
from PyQt5.QtWidgets import QApplication, QDialog, QVBoxLayout, QLabel
from PyQt5 import QtGui


class UI_demo(QDialog):
    """用户界面"""
    def __init__(self):
        super().__init__()

        # 窗口信息
        self.title = 'PyQt5 Layout Managment'
        self.left = 600
        self.top = 200
        self.width = 500
        self.height = 200

        self.initWindow()

    def initWindow(self):

        # 窗口信息
        self.setWindowIcon(QtGui.QIcon('../img/home.ico'))  # 图标设置
        self.setGeometry(self.left, self.top, self.width, self.height)  # 大小位置设置
        self.setWindowTitle(self.title)  # 窗口标题

        vbox = QVBoxLayout()

        # 背景图
        labelImage = QLabel(self)
        pixmap = QtGui.QPixmap('../img/default.jpg')
        labelImage.setPixmap(pixmap)
        vbox.addWidget(labelImage)

        self.setLayout(vbox)

        # 展示窗口
        self.show()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = UI_demo()
    sys.exit(app.exec_())

