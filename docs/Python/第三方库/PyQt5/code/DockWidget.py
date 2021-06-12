# _*_ encoding:utf-8 _*_
__author__ = 'nzb'
__datetime__ = '2019/5/23 13:23'

from PyQt5 import QtGui
from PyQt5.QtWidgets import QApplication, QPushButton, QStackedWidget, QGroupBox, QComboBox, QDialog, QTabWidget, QWidget, QVBoxLayout, QLabel
from PyQt5.QtWidgets import QTextEdit, QDockWidget, QMainWindow, QListWidget
import sys
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt


class DockDialog(QMainWindow):
    """可停靠的窗口小部件"""
    def __init__(self):
        super().__init__()

        # 窗口信息
        self.title = 'PyQt5 DockDialog'
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
        self.createDockWidget()

        # 展示窗口
        self.show()

    def createDockWidget(self):
        menubar = self.menuBar()
        file = menubar.addMenu("File")
        file.addAction("New")
        file.addAction("Save")
        file.addAction("Close")

        self.dock = QDockWidget("Dockable", self)
        self.listwidget = QListWidget()

        list1 = ['Python', 'C++', 'Java']

        self.listwidget.addItems(list1)

        self.dock.setWidget(self.listwidget)

        self.setCentralWidget(QTextEdit())

        self.addDockWidget(Qt.RightDockWidgetArea, self.dock)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = DockDialog()
    sys.exit(app.exec_())