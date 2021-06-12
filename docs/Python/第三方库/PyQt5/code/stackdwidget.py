# _*_ encoding:utf-8 _*_
__author__ = 'nzb'
__datetime__ = '2019/5/23 11:29'

from PyQt5 import QtGui
from PyQt5.QtWidgets import QApplication, QPushButton, QStackedWidget, QGroupBox, QComboBox, QDialog, QTabWidget, QWidget, QVBoxLayout, QDialogButtonBox, QTabWidget, QLabel, QLineEdit
import sys
from PyQt5.QtGui import QIcon


class StackWidget(QDialog):
    """堆叠小部件"""
    def __init__(self):
        super().__init__()

        # 窗口信息
        self.title = 'PyQt5 StackedWidget'
        self.left = 600
        self.top = 200
        self.width = 500
        self.height = 400
        self.iconName = '../img/home.ico'

        self.initWindow()

    def initWindow(self):

        # 窗口信息
        self.setWindowIcon(QtGui.QIcon(self.iconName))  # 图标设置
        self.setGeometry(self.left, self.top, self.width, self.height)  # 大小位置设置
        self.setWindowTitle(self.title)  # 窗口标题

        #
        self.StackdWidget()

        # 展示窗口
        self.show()

    def StackdWidget(self):

        vbox = QVBoxLayout()

        self.stackedWidget = QStackedWidget()
        vbox.addWidget(self.stackedWidget)


        for i in range(0, 8):
            label = QLabel("Stacked Child" + str(i))
            label.setFont(QtGui.QFont("Sanserif", 15))
            label.setStyleSheet('color:red')

            self.stackedWidget.addWidget(label)

            self.button = QPushButton("Stack" + str(i))
            self.button.setStyleSheet('background-color:green')

            self.button.page = i

            self.button.clicked.connect(self.btn_clicked)

            vbox.addWidget(self.button)

        self.setLayout(vbox)

    def btn_clicked(self):
        self.button = self.sender()
        self.stackedWidget.setCurrentIndex(self.button.page - 1)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = StackWidget()
    sys.exit(app.exec_())
