# _*_ encoding:utf-8 _*_
__author__ = 'nzb'
__datetime__ = '2019/5/23 9:46'

import sys
from PyQt5.QtWidgets import QFileDialog,  QApplication, QMainWindow, QAction, QTextEdit, QFontDialog, QColorDialog
from PyQt5 import QtGui,QtCore
from PyQt5.QtPrintSupport import QPrintDialog, QPrinter, QPrintPreviewDialog
from PyQt5.QtCore import QFileInfo


class UI_demo(QMainWindow):
    """用户界面"""
    def __init__(self):
        super().__init__()

        # 窗口信息
        self.title = 'PyQt5 PDF'
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

        # 复制
        copyAction = QAction(QtGui.QIcon('../img/copy.ico'), 'Copy', self)
        copyAction.setShortcut("Ctrl+C")
        editMenu.addAction(copyAction)
        # 剪切
        cutAction = QAction(QtGui.QIcon('../img/cut.png'), 'Cut', self)
        cutAction.setShortcut("Ctrl+X")
        editMenu.addAction(cutAction)
        # 保存
        saveAction = QAction(QtGui.QIcon('../img/save.png'), 'Save', self)
        saveAction.setShortcut("Ctrl+S")
        editMenu.addAction(saveAction)
        # 打印
        printAction = QAction(QtGui.QIcon('../img/print.png'), "Print", self)
        printAction.triggered.connect(self.printDialog)
        fileMenu.addAction(printAction)
        # 打印预览
        printpreviewAction = QAction(QtGui.QIcon('../img/printpreview.png'), "PrintPreview", self)
        printpreviewAction.triggered.connect(self.printPreviewDialog)
        fileMenu.addAction(printpreviewAction)
        # pdf
        pdfAction = QAction(QtGui.QIcon('../img/pdf.png'), 'PDF', self)
        pdfAction.triggered.connect(self.pdfExport)
        fileMenu.addAction(pdfAction)
        # 退出
        exitAction = QAction(QtGui.QIcon('../img/exit.png'), 'Exit', self)
        exitAction.setShortcut("Ctrl+E")
        exitAction.triggered.connect(self.exitWindow)
        fileMenu.addAction(exitAction)
        # 黏贴
        pasteAction = QAction(QtGui.QIcon('../img/paste.png'), 'Paste', self)
        pasteAction.setShortcut("Ctrl+E")
        editMenu.addAction(pasteAction)
        # 字体
        fontAction = QAction(QtGui.QIcon('../img/font.png'), "Font", self)
        fontAction.setShortcut("Ctrl+F")
        fontAction.triggered.connect(self.fontDialog)
        viewMenu.addAction(fontAction)
        # 字体颜色
        colorAction = QAction(QtGui.QIcon('../img/color.png'), "Color", self)
        colorAction.triggered.connect(self.colorDialog)
        viewMenu.addAction(colorAction)

        # 工具栏
        toolbar = self.addToolBar("Toolbar")
        toolbar.addAction(copyAction)
        toolbar.addAction(cutAction)
        toolbar.addAction(saveAction)
        toolbar.addAction(exitAction)
        toolbar.addAction(pasteAction)
        toolbar.addAction(fontAction)
        toolbar.addAction(colorAction)
        toolbar.addAction(printAction)
        toolbar.addAction(pdfAction)

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

    def colorDialog(self):
        """颜色对话框"""
        color = QColorDialog.getColor()
        self.textEdit.setTextColor(color)

    def printDialog(self):
        """打印文本框"""
        printer = QPrinter(QPrinter.HighResolution)
        dialog = QPrintDialog(printer, self)

        if dialog.exec_() == QPrintDialog.Accepted:
            self.textEdit.print_(printer)

    def printPreviewDialog(self):
        """打印预览"""
        printer = QPrinter(QPrinter.HighResolution)
        previewDialog = QPrintPreviewDialog(printer, self)
        previewDialog.paintRequested.connect(self.printPreview)
        previewDialog.exec_()

    def printPreview(self, printer):
        """打印预览"""
        self.textEdit.print_(printer)

    def pdfExport(self):
        """导出PDF"""
        fn, _ = QFileDialog.getSaveFileName(self, "Export PDF", None, "PDF files (.pdf);;All Files()")

        if fn != '':

            if QFileInfo(fn).suffix() == "":
                fn += '.pdf'
            printer = QPrinter(QPrinter.HighResolution)
            printer.setOutputFormat(QPrinter.PdfFormat)
            printer.setOutputFileName(fn)
            self.textEdit.document().print_(printer)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = UI_demo()
    sys.exit(app.exec_())
