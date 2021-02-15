# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'gui_test3.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib import pyplot as plt

def plot_func(self):
    from matplotlib import pyplot as plt
    #cf.swap_pyplot_backend(plot_inline=False)
    self.figure.clear() 
   
        # create an axis 
    ax = self.figure.add_subplot(111) 
   
        # plot data 
    ax.plot([1,2,3,4,5], '*-') 
   
        # refresh canvas 
    self.widget.draw() 

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(400, 300)
        self.pushButton = QtWidgets.QPushButton(Dialog)
        self.pushButton.setGeometry(QtCore.QRect(310, 260, 75, 23))
        self.pushButton.setObjectName("pushButton")
        self.widget = QtWidgets.QWidget(Dialog)
        self.widget.setGeometry(QtCore.QRect(80, 70, 120, 80))
        self.widget.setObjectName("widget")

        self.figure = plt.figure() 
        self.widget = FigureCanvas(self.figure) 
        
        self.pushButton.clicked.connect(plot_func)
        
        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.pushButton.setText(_translate("Dialog", "Plot"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())

