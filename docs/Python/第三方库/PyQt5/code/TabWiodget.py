# _*_ encoding:utf-8 _*_
__author__ = 'nzb'
__datetime__ = '2019/5/23 10:56'

from PyQt5 import QtGui
from PyQt5.QtWidgets import QApplication, QGroupBox, QComboBox, QCheckBox, QDialog, QTabWidget, QWidget, QVBoxLayout, QDialogButtonBox, QTabWidget, QLabel, QLineEdit
import sys
from PyQt5.QtGui import QIcon


class Tab(QDialog):
    """选项卡"""
    def __init__(self):
        super().__init__()

        self.setWindowTitle("PyQt5 Tab Widget")
        self.setWindowIcon(QIcon('../img/home.ico'))

        vbox = QVBoxLayout()
        tabWidget = QTabWidget()

        # 按钮
        buttonbox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)

        buttonbox.accepted.connect(self.accept)
        buttonbox.accepted.connect(self.reject)
        # 选项卡
        tabWidget.addTab(TabContact(), "Contact Details")
        tabWidget.addTab(TabPersonDetails(), 'Personal Details')

        vbox.addWidget(tabWidget)
        vbox.addWidget(buttonbox)

        self.setLayout(vbox)


class TabContact(QWidget):
    def __init__(self):
        super().__init__()

        nameLabel = QLabel("Name: ")
        nameEdit = QLineEdit()

        phoneLabel = QLabel("Phone: ")
        phoneEdit = QLineEdit()

        emailLabel = QLabel("Email: ")
        emailEdit = QLineEdit()

        vbox = QVBoxLayout()

        vbox.addWidget(nameLabel)
        vbox.addWidget(nameEdit)

        vbox.addWidget(phoneLabel)
        vbox.addWidget(phoneEdit)

        vbox.addWidget(emailLabel)
        vbox.addWidget(emailEdit)

        self.setLayout(vbox)


class TabPersonDetails(QWidget):
    def __init__(self):
        super().__init__()

        # 单选下拉框
        groupbox = QGroupBox("select your gender")
        list1 = ["male", 'female']

        combo = QComboBox()
        combo.addItems(list1)

        vbox = QVBoxLayout()
        vbox.addWidget(combo)
        groupbox.setLayout(vbox)

        # 多选
        groupbox2 = QGroupBox("select your favorite programming language")
        python = QCheckBox("Python")
        cpp = QCheckBox("C++")
        java = QCheckBox("Java")

        vbox = QVBoxLayout()
        vbox.addWidget(python)
        vbox.addWidget(cpp)
        vbox.addWidget(java)

        groupbox2.setLayout(vbox)

        mainLayout = QVBoxLayout()
        mainLayout.addWidget(groupbox)
        mainLayout.addWidget(groupbox2)

        self.setLayout(mainLayout)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    tabDialog = Tab()
    tabDialog.show()
    app.exec_()