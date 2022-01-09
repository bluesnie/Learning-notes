# _*_ encoding:utf-8 _*_
__author__ = 'nzb'
__datetime__ = '2019/5/23 14:30'

from PyQt5 import QtGui
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QComboBox, QVBoxLayout,  QDialog, QMainWindow, QCalendarWidget, QVBoxLayout, QLabel
import sys
from PyQt5.QtWidgets import QCompleter, QLineEdit, QTimeEdit
from PyQt5.QtCore import QTime


class Window(QWidget):
    """时间编辑"""
    def __init__(self):
        super().__init__()

        self.title = "PyQt5 QDialog"
        self.top = 200
        self.left = 500
        self.width = 400
        self.height = 300

        self.InitWindow()

    def InitWindow(self):
        self.setWindowIcon(QtGui.QIcon("../img/home.ico"))
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        self.MyTime()
        self.show()

    def MyTime(self):

        vbox = QVBoxLayout()
        time = QTime()
        time.setHMS(13, 15, 40)

        timeedit = QTimeEdit()

        timeedit.setFont(QtGui.QFont('Sanserif', 15))
        timeedit.setTime(time)

        vbox.addWidget(timeedit)

        self.setLayout(vbox)





App = QApplication(sys.argv)
window = Window()
sys.exit(App.exec())