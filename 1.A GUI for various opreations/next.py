#QMessageBox消息框使用
from PyQt5.QtWidgets import  QVBoxLayout,QWidget,QApplication ,QHBoxLayout,QDialog,QPushButton,QMessageBox

from PyQt5.QtGui import QIcon,QPixmap,QFont
from PyQt5.QtCore import  Qt

import sys

class WindowClass(QWidget):

    def __init__(self,parent=None):

        super(WindowClass, self).__init__(parent)
        layout=QVBoxLayout()
        self.btn=QPushButton()
        self.btn.setText("显示消息框")
        self.btn.clicked.connect(self.showMessageBox)
        self.resize(500,500)
        layout.addWidget(self.btn)

        self.setLayout(layout)

    def showMessageBox(self):
       res_1= QMessageBox.information(self,"标题","消息对话框正文！",QMessageBox.Yes|QMessageBox.No,QMessageBox.Yes)
       print(res_1)
       res_2 = QMessageBox.question(self, "标题", "消息对话框正文！", QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
       print(res_2)
       res_3 = QMessageBox.warning(self, "标题", "消息对话框正文！", QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
       print(res_3)
       res_4 = QMessageBox.critical(self, "标题", "消息对话框正文！", QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
       print(res_4)
       res_5 = QMessageBox.about(self, "标题", "消息对话框正文！" )
       print(res_5)

if __name__=="__main__":
    app=QApplication(sys.argv)
    win=WindowClass()
    win.show()
    sys.exit(app.exec_())