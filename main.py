from parsing import x1, x2, x3, x4, x5, tau, d, translate_input
import sympy as smp
import matplotlib.pylab as plt
from compute import substitute, quadr_approx, quadratic_minimum, algorithm
from gui import Ui_MainWindow
import sys
from PyQt5 import QtWidgets

    
if __name__ == '__main__':

    app = QtWidgets.QApplication(sys.argv)

    # Open the style sheet file and read it
    with open('style.qss', 'r') as f:
        style = f.read()
    # Set the current style sheet
    app.setStyleSheet(style)

    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()

    sys.exit(app.exec_())
