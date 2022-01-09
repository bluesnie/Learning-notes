# _*_ encoding:utf-8 _*_
__author__ = 'nzb'
__datetime__ = '2019/5/22 17:11'

import sys
from PyQt5.QtWidgets import  QApplication, QMainWindow, QAction, QTextEdit, QFontDialog
from PyQt5 import QtGui,QtCore


class UI_demo(QMainWindow):
    """用户界面"""
    def __init__(self):
        super().__init__()

        # 窗口信息
        self.title = 'PyQt5 Font Dialog'
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

        # 生成菜单栏
        self.CreateMenu()

        # 生成文档编辑

        self.createEditor()

        # 展示窗口
        self.show()

    def CreateMenu(self):
        """菜单栏"""
        mainMenu = self.menuBar()
        fileMenu = mainMenu.addMenu('File')
        editMenu = mainMenu.addMenu('Edit')
        viewMenu = mainMenu.addMenu('View')
        helpMenu = mainMenu.addMenu('Help')

        copyAction = QAction(QtGui.QIcon(self.iconName), 'Copy', self)
        copyAction.setShortcut("Ctrl+C")
        editMenu.addAction(copyAction)

        cutAction = QAction(QtGui.QIcon(self.iconName), 'Cut', self)
        cutAction.setShortcut("Ctrl+X")
        editMenu.addAction(cutAction)

        saveAction = QAction(QtGui.QIcon(self.iconName), 'Save', self)
        saveAction.setShortcut("Ctrl+S")
        editMenu.addAction(saveAction)

        exitAction = QAction(QtGui.QIcon('../img/Agt Stop.ico'), 'Exit', self)
        exitAction.setShortcut("Ctrl+E")
        exitAction.triggered.connect(self.exitWindow)
        fileMenu.addAction(exitAction)

        pasteAction = QAction(QtGui.QIcon('../img/Agt Stop.ico'), 'Paste', self)
        pasteAction.setShortcut("Ctrl+E")
        editMenu.addAction(pasteAction)

        fontAction = QAction(QtGui.QIcon(self.iconName), "Font", self)
        fontAction.setShortcut("Ctrl+F")
        fontAction.triggered.connect(self.fontDialog)
        viewMenu.addAction(fontAction)

        # 工具栏
        toolbar = self.addToolBar("Toolbar")
        toolbar.addAction(copyAction)
        toolbar.addAction(cutAction)
        toolbar.addAction(saveAction)
        toolbar.addAction(exitAction)
        toolbar.addAction(pasteAction)

    def exitWindow(self):
        """关闭窗口"""
        self.close()

    def createEditor(self):
        """文档编辑"""
        self.textEdit = QTextEdit(self)
        self.setCentralWidget(self.textEdit)

    def fontDialog(self):
        """字体对话框"""
        font, ok = QFontDialog.getFont()

        if ok:
            self.textEdit.setFont(font)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = UI_demo()
    sys.exit(app.exec_())