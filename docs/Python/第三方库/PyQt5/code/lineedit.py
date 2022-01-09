# _*_ encoding:utf-8 _*_
__author__ = 'nzb'
__datetime__ = '2019/5/20 16:00'

import sys
from PyQt5.QtWidgets import QWidget, QApplication, QDialog, QVBoxLayout, QLabel, QGroupBox, QCheckBox, QHBoxLayout, QPushButton, QLineEdit
from PyQt5 import QtGui, QtCore


class UI_demo(QWidget):
    """用户界面"""
    def __init__(self):
        super().__init__()

        # 窗口信息
        self.title = 'PyQt5 Lineedit'
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

        # 行编辑
        hbox = QHBoxLayout()

        self.lineedit = QLineEdit(self)
        self.lineedit.setFont(QtGui.QFont('Sanserif', 15))
        self.lineedit.returnPressed.connect(self.onPressed)
        hbox.addWidget(self.lineedit)

        self.lable = QLabel(self)
        self.lable.setFont(QtGui.QFont('Sanserif', 15))
        hbox.addWidget(self.lable)

        self.setLayout(hbox)

        # 展示窗口
        self.show()

    def onPressed(self):
        """输入绑定事件"""
        self.lable.setText(self.lineedit.text())


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = UI_demo()
    sys.exit(app.exec_())
