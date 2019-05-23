# _*_ encoding:utf-8 _*_
__author__ = 'nzb'
__datetime__ = '2019/5/23 14:39'


from PyQt5 import QtGui
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QComboBox, QVBoxLayout,  QDialog, QMainWindow, QCalendarWidget, QVBoxLayout, QLabel
import sys
from PyQt5.QtWidgets import QCompleter, QLineEdit, QTimeEdit, QListWidget
from PyQt5.QtCore import QTime


class Window(QWidget):
    """列表部件"""
    def __init__(self):
        super().__init__()

        self.title = "PyQt5 QListWidget"
        self.top = 200
        self.left = 500
        self.width = 400
        self.height = 300

        self.InitWindow()

    def InitWindow(self):
        self.setWindowIcon(QtGui.QIcon("../img/home.ico"))
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        vbox = QVBoxLayout()

        self.list = QListWidget()
        self.list.insertItem(0, "Python")
        self.list.insertItem(1, "PHP")
        self.list.insertItem(2, "Java")
        self.list.insertItem(3, "C++")

        self.list.clicked.connect(self.listview_clicked)

        self.label = QLabel()
        self.setFont(QtGui.QFont("Sanserif", 15))

        vbox.addWidget(self.label)
        vbox.addWidget(self.list)

        self.setLayout(vbox)

        self.show()

    def listview_clicked(self):
        item = self.list.currentItem()
        self.label.setText(str(item.text()))


App = QApplication(sys.argv)
window = Window()
sys.exit(App.exec())