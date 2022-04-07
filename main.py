from parsing import x1, x2, x3, x4, x5, tau, d, translate_input
import numpy as np
import sympy as smp
import matplotlib.pylab as plt
from compute import substitute, quadr_approx, quadratic_minimum, algorithm
from gui import Ui_MainWindow
import sys
from PyQt5 import QtWidgets

    
if __name__ == '__main__':

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()

    # func_org = translate_input("5*x1**2 - 22*x1 + 25")  # string -> sympify
    # func_org = translate_input("-(x1-3)^3+(x2-7)^2")
    # func_org = translate_input("x1^2+x2^2")  # sphere function
    # func_org = translate_input("100*(x2-x1^2)^2+(1-x1)^2")  # Rosenbrock function - something's
    # quite right with this one
    # func_org = translate_input("(x1 + 2*x2 -7)^2 + (2*x1 + x2 - 5)^2")  # Booth function
    # func_org = translate_input("(1.5-x1-x1*x2)^2 + (2.25-x1-x1*x2^2)^2 + (2.625-x1+x1*x2^3)^2")  # Beale function
    func_org = translate_input("(x1^2+x2-11)^2 + (x1+x2^2-7)^2")  # Himmelblau's function
    # func_org = translate_input("-cos(x1)*cos(x2)*exp(-((x1-pi)^2+(x2-pi)^2))")  # Easom function - the point
    # won't move because one of the while loop condition is always met (f(a)-f(c)<e)
    func_org_iso = smp.lambdify([x1, x2], func_org, 'numpy')
    x = np.array([x1, x2])
    x0 = np.array([-1, 0])
    d = np.array([3, 0.2])
    d = d / d[0]  # normalize the direction
    x_next = x0 + tau * d
    # print(f"{x_next = }")
    func_org = substitute(func_org, x_next)  # create one-variable function (of tau) in a given direction (sympified)
    # print(f"{func_org = }")
    func = smp.lambdify(tau, func_org, 'numpy')  # make one-variable function lambdified with respect to tau

    # define basic values
    t = np.linspace(-100, 100, 1000)
    a = -5
    c = 5

    # optimal_x = algorithm(func=func, iso_func=func_org_iso, a=a, c=c, x0=x0, app=ui)

    sys.exit(app.exec_())


    #######
    # for x1^2+x2^3; x0=(3,1); d=(1,2), the algorithm only stops if it reaches the max number of iterations
    #######
