# _*_ encoding:utf-8 _*_
__author__ = 'nzb'
__datetime__ = '2019/5/20 14:42'

import sys
from PyQt5.QtWidgets import QApplication, QDialog, QVBoxLayout, QLabel, QGroupBox, QRadioButton, QHBoxLayout
from PyQt5 import QtGui, QtCore


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

        # 单选框
        self.radioButton()
        vbox.addWidget(self.groupBox)

        self.label = QLabel(self)
        self.label.setFont(QtGui.QFont("Sanserif", 20))
        vbox.addWidget(self.label)

        self.setLayout(vbox)

        # 展示窗口
        self.show()

    def radioButton(self):
        """单选框"""
        self.groupBox = QGroupBox("What is your favorite sport?")
        self.groupBox.setFont(QtGui.QFont("Sanserif", 12))
        hboxlayout = QHBoxLayout()

        self.radiobtn1 = QRadioButton('Soccer')
        self.radiobtn1.setChecked(True)  # 选中状态
        self.radiobtn1.setIcon(QtGui.QIcon('../img/Soccer.ico'))
        self.radiobtn1.setIconSize(QtCore.QSize(40, 40))
        self.radiobtn1.setFont(QtGui.QFont('Sanserif', 13))
        self.radiobtn1.toggled.connect(self.OnRadioBtn)  # 选中事件
        hboxlayout.addWidget(self.radiobtn1)

        self.radiobtn2 = QRadioButton('Tennis')
        self.radiobtn2.setIcon(QtGui.QIcon('../img/Tennis.ico'))
        self.radiobtn2.setIconSize(QtCore.QSize(40, 40))
        self.radiobtn2.setFont(QtGui.QFont('Sanserif', 13))
        self.radiobtn2.toggled.connect(self.OnRadioBtn)
        hboxlayout.addWidget(self.radiobtn2)

        self.radiobtn3 = QRadioButton('Basketball')
        self.radiobtn3.setIcon(QtGui.QIcon('../img/Basketball.ico'))
        self.radiobtn3.setIconSize(QtCore.QSize(40, 40))
        self.radiobtn3.setFont(QtGui.QFont('Sanserif', 13))
        self.radiobtn3.toggled.connect(self.OnRadioBtn)
        hboxlayout.addWidget(self.radiobtn3)

        self.groupBox.setLayout(hboxlayout)

    def OnRadioBtn(self):
        """单选框选中事件"""
        radioBtn = self.sender()

        if radioBtn.isChecked():
            self.label.setText("You have selected " + radioBtn.text())


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = UI_demo()
    sys.exit(app.exec_())
