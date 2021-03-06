# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'gui.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.

from PyQt5.QtCore import QObject, QThread, pyqtSignal
from PyQt5.QtWidgets import QMessageBox, QWidget, QDialog
from PyQt5 import QtCore, QtGui, QtWidgets
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from compute import algorithm, substitute, algorithm_step
import numpy as np
from parsing import x1, x2, x3, x4, x5, tau, d, translate_input
import sympy as smp
import time
import pyqtgraph as pg
import re
from matplotlib.figure import Figure


# this one blocks the GUI, even though is recommended for use
class Worker(QObject):
    finished = pyqtSignal()
    progress = pyqtSignal(int)

    def __init__(self, func=None):
        super().__init__()
        try:
            self.func = func
        except Exception:
            print("exception in thread")

    def run(self, output):
        print("worker started")#func()
        output["error_state_and_message"].append(self.func())
        self.finished.emit()

# this one works, even though it should not be used
# class Worker(QThread):
#     # finished = pyqtSignal()
#     # progress = pyqtSignal(int)
#
#     def __init__(self, func=None):  # overloaded constructor so that an argument can be passed
#         super(QThread, self).__init__()
#         self.func = func
#
#     def run(self):
#         self.func()

class Ui_MainWindow(QWidget):
    def setupUi(self, MainWindow):
        MainWindow.setWindowIcon(QtGui.QIcon('icon.ico'))
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1080, 611)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.frame_1 = QtWidgets.QFrame(self.centralwidget)
        self.frame_1.setObjectName("frame_1")
        self.frame_1.setMinimumSize(430, 400)
        self.label = QtWidgets.QLabel(self.frame_1)
        self.label.setGeometry(QtCore.QRect(10, 40, 71, 21))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.lineEdit_7 = QtWidgets.QLineEdit(self.frame_1)
        self.lineEdit_7.setGeometry(QtCore.QRect(90, 280, 201, 21))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.lineEdit_7.setFont(font)
        self.lineEdit_7.setObjectName("lineEdit_7")
        self.label_4 = QtWidgets.QLabel(self.frame_1)
        self.label_4.setGeometry(QtCore.QRect(10, 160, 71, 21))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")

        #######################################################################
        self.pushButton = QtWidgets.QPushButton(self.frame_1)
        self.pushButton.setGeometry(QtCore.QRect(90, 350, 101, 31))
        self.pushButton.setObjectName("pushButton")
        self.pushButton.clicked.connect(lambda: self.optimize_thread())
        self.next_step = False
        #######################################################################

        self.label_3 = QtWidgets.QLabel(self.frame_1)
        self.label_3.setGeometry(QtCore.QRect(10, 120, 71, 21))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.label_5 = QtWidgets.QLabel(self.frame_1)
        self.label_5.setGeometry(QtCore.QRect(10, 200, 71, 21))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")
        self.label_6 = QtWidgets.QLabel(self.frame_1)
        self.label_6.setGeometry(QtCore.QRect(10, 240, 71, 21))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_6.setFont(font)
        self.label_6.setObjectName("label_6")
        self.label_7 = QtWidgets.QLabel(self.frame_1)
        self.label_7.setGeometry(QtCore.QRect(10, 280, 71, 21))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_7.setFont(font)
        self.label_7.setObjectName("label_7")
        self.lineEdit_5 = QtWidgets.QLineEdit(self.frame_1)
        self.lineEdit_5.setGeometry(QtCore.QRect(90, 200, 201, 21))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.lineEdit_5.setFont(font)
        self.lineEdit_5.setObjectName("lineEdit_5")

        #######################################################################
        self.function_combobox = QtWidgets.QComboBox(self.frame_1)
        self.function_combobox.setGeometry(QtCore.QRect(90, 40, 340, 22))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.function_combobox.setEditable(True)
        self.function_combobox.setFont(font)
        self.function_combobox.setObjectName("function_combobox")

        function_list = ["x1^2+x2^2", "100*(x2-x1^2)^2+(1-x1)^2", "(x1 + 2*x2 -7)^2 + (2*x1 + x2 - 5)^2",
                         "(1.5-x1+x1*x2)^2 + (2.25-x1-x1*x2^2)^2 + (2.625-x1+x1*x2^3)^2",
                         "(x1^2+x2-11)^2 + (x1+x2^2-7)^2", "-cos(x1)*cos(x2)*exp(-((x1-pi)^2+(x2-pi)^2))",
                         "x1^4+x2^4-x1^2-x2^2", "(x1+2*x2-7)^2+(2*x1+x2-5)^2"]
        self.function_combobox.addItems(function_list)
        #######################################################################

        # self.lineEdit = QtWidgets.QLineEdit(self.frame_1)
        # self.lineEdit.setGeometry(QtCore.QRect(90, 40, 301, 22))
        # font = QtGui.QFont()
        # font.setPointSize(10)
        # self.lineEdit.setFont(font)
        # self.lineEdit.setObjectName("lineEdit")


        self.label_2 = QtWidgets.QLabel(self.frame_1)
        self.label_2.setGeometry(QtCore.QRect(10, 80, 71, 21))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.lineEdit_3 = QtWidgets.QLineEdit(self.frame_1)
        self.lineEdit_3.setGeometry(QtCore.QRect(90, 120, 201, 21))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.lineEdit_3.setFont(font)
        self.lineEdit_3.setObjectName("lineEdit_3")
        self.lineEdit_2 = QtWidgets.QLineEdit(self.frame_1)
        self.lineEdit_2.setGeometry(QtCore.QRect(90, 80, 201, 21))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.lineEdit_2.setFont(font)
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.lineEdit_4 = QtWidgets.QLineEdit(self.frame_1)
        self.lineEdit_4.setGeometry(QtCore.QRect(90, 160, 201, 21))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.lineEdit_4.setFont(font)
        self.lineEdit_4.setObjectName("lineEdit_4")
        self.lineEdit_6 = QtWidgets.QLineEdit(self.frame_1)
        self.lineEdit_6.setGeometry(QtCore.QRect(90, 240, 201, 21))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.lineEdit_6.setFont(font)
        self.lineEdit_6.setObjectName("lineEdit_6")
        self.gridLayout.addWidget(self.frame_1, 0, 0, 1, 1)
        self.frame = QtWidgets.QFrame(self.centralwidget)
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.frame.setMinimumWidth(500)

        #######################################################################
        # step mode checkbox
        self.step_mode = False
        self.algorithm_thread_started = False
        self.mode_checkbox = QtWidgets.QCheckBox("Tryb krokowy", self.frame_1)
        self.mode_checkbox.setGeometry(QtCore.QRect(90, 320, 201, 21))
        self.mode_checkbox.toggled.connect(self.change_step_mode)
        #######################################################################

        #######################################################################
        self.progress_label = QtWidgets.QLabel(self.frame_1)
        self.progress_label.setGeometry(QtCore.QRect(90, 390, 201, 21))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.progress_label.setFont(font)
        self.progress_label.setObjectName("progress_label")
        self.progress_label.hide()
        #######################################################################

        #######################################################################
        # add a horizontal layout
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.frame)
        self.horizontalLayout.setObjectName("horizontalLayout")
        # add a frame to horizontal layout and add a vertical layout to that frame
        self.frame_2 = QtWidgets.QFrame(self)
        self.horizontalLayout.addWidget(self.frame_2)
        self.verticalLayout = QtWidgets.QVBoxLayout(self.frame_2)
        self.verticalLayout.setObjectName('verticalLayout')
        # create figure and canvas
        self.figure = plt.figure()
        self.figure.patch.set_facecolor("None")
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setStyleSheet("background-color:transparent;")
        self.toolbar = NavigationToolbar(self.canvas, self)
        # add canvas to vertical layout
        self.verticalLayout.addWidget(self.canvas)
        self.verticalLayout.addWidget(self.toolbar)
        # add popup attribute
        self.popup = None
        #######################################################################

        #######################################################################
        # add a text editor output
        self.output_text_window = QtWidgets.QTextEdit(self.frame_1)
        self.output_text_window.setGeometry(QtCore.QRect(50, 430, 300, 140))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.output_text_window.setReadOnly(True)
        self.output_text_window.setFont(font)
        self.output_text_window.setObjectName("output_text_window")
        self.output_text_window.setAccessibleName("output_text_window")
        #######################################################################

        self.gridLayout.addWidget(self.frame, 0, 1, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 884, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Optymalizacja w kierunku metod?? aproksymacji kwadratowej"))
        self.label.setText(_translate("MainWindow", "F(x)"))
        self.lineEdit_7.setText(_translate("MainWindow", "20"))
        self.label_4.setText(_translate("MainWindow", "??1"))
        self.pushButton.setText(_translate("MainWindow", "Szukaj rozwi??zania"))
        self.label_3.setText(_translate("MainWindow", "d"))
        self.label_5.setText(_translate("MainWindow", "??2"))
        self.label_6.setText(_translate("MainWindow", "iteracje L"))
        self.label_7.setText(_translate("MainWindow", "d??ugo???? w"))
        self.lineEdit_5.setText(_translate("MainWindow", "0.001"))
        # self.lineEdit.setText(_translate("MainWindow", "x1^2+x2^2"))
        self.label_2.setText(_translate("MainWindow", "x0"))
        self.lineEdit_3.setText(_translate("MainWindow", "1, 0"))
        self.lineEdit_2.setText(_translate("MainWindow", "0, 0"))
        self.lineEdit_4.setText(_translate("MainWindow", "0.001"))
        self.lineEdit_6.setText(_translate("MainWindow", "20"))
        self.progress_label.setText(_translate("MainWindow", "Trwaj?? obliczenia..."))

    def optimize(self):
        try:
            data = self.get_data()
            processed_data = self.process_the_data(data)
            # if step mode
            if self.step_mode:
                # change label on the button
                self.pushButton.setText("Nast??pny krok")
                self.pushButton.setEnabled(True)
                # algorithm
                # first step with data from input
                results = self.run_algorithm(processed_data, step_mode=True)
                self.show_approx_func(results)
                # report the progress
                self.write_solution_to_textedit(results)
                if processed_data["order"] == 2:
                    self.plot_results(results, processed_data)
                self.next_step = False
                while results["end"] == None:
                    # wait for user to press the button
                    while not self.next_step:
                        QtWidgets.QApplication.processEvents()
                        time.sleep(0.05)
                    # reset the variable
                    self.next_step = False
                    results = self.run_algorithm(results, step_mode=True)
                    # report the progress
                    self.write_solution_to_textedit(results)
                    # show normal function and quadratic approximation function with new tau
                    self.show_approx_func(results)
                    # show isohypse
                    if processed_data["order"] == 2:
                        self.plot_results(results, processed_data)
                # show popup with info
                msg = QMessageBox()
                msg.setWindowTitle("Rozwi??zanie")
                msg.setIcon(QMessageBox.Information)
                msg.setStandardButtons(QMessageBox.Ok)
                msg.setText(f"Znaleziono rozwi??zanie.")
                x = msg.exec_()
                # change the button text
                self.pushButton.setText("Szukaj rozwi??zania")
            # if normal mode
            elif not self.step_mode:
                results = self.run_algorithm(processed_data)
            if processed_data["order"] == 2:
                self.plot_results(results, processed_data)
            self.write_solution_to_textedit(results)
        except Exception as e:
            print(e)
            self.pushButton.setText("Szukaj rozwi??zania")
            return -1, e
        else:
            return 0, ""

    def optimize_thread(self):
        if self.step_mode:
            self.next_step = True
        if not self.algorithm_thread_started:
            self.algorithm_thread_started = True
            # self.optimize_thr = Worker(self.optimize)
            # self.optimize_thr.finished.connect(self.optimize)
            # self.optimize_thr.start()
            output_state = {
                "error_state_and_message": [],
            }
            # create QThread object
            self.algorithm_thread = QThread()
            # create a worker object
            self.algorithm_worker = Worker(func=self.optimize)
            # move worker object to the thread
            self.algorithm_worker.moveToThread(self.algorithm_thread)
            # connect signals and slots
            self.algorithm_thread.started.connect(lambda: self.algorithm_worker.run(output=output_state))
            self.algorithm_worker.finished.connect(self.algorithm_thread.quit)
            self.algorithm_worker.finished.connect(self.algorithm_worker.deleteLater)
            self.algorithm_thread.finished.connect(self.algorithm_thread.deleteLater)
            # start the thread
            self.algorithm_thread.start()
            self.pushButton.setEnabled(False)
            self.progress_label.show()

            # final results
            self.algorithm_thread.finished.connect(lambda: self.pushButton.setEnabled(True))
            self.algorithm_thread.finished.connect(lambda: self.progress_label.hide())
            # self.algorithm_thread.finished.connect(lambda: print(f"{output_state = }"))
            self.algorithm_thread.finished.connect(lambda: self.show_warning(output_state))
            self.algorithm_thread.finished.connect(self.algorithm_thread_finished)

    def show_warning(self, error_info):
        """ Either shows a warning or not, based on the state_flag value """
        # print("show_warning")
        # print(f"{error_info['error_state_and_message'] = }")
        if error_info["error_state_and_message"][0][0] == 0:
            return
        elif error_info["error_state_and_message"][0][0] == -1:
            msg = QMessageBox()
            msg.setWindowTitle("B????d")
            msg.setIcon(QMessageBox.Warning)
            msg.setStandardButtons(QMessageBox.Ok)
            msg.setText(f"Obliczenia nie zosta??y wykonane.\nWprowadzone dane mog?? by?? nieprawid??owe.")
            msg.setDetailedText(f"\nTre???? b????du:\n{error_info['error_state_and_message'][0][1]}")
            x = msg.exec_()

    def get_data(self):
        func = self.function_combobox.currentText()
        x0_str = self.lineEdit_2.text()
        d_str = self.lineEdit_3.text()
        e1 = float(self.lineEdit_4.text())
        e2 = float(self.lineEdit_5.text())
        l = float(self.lineEdit_6.text())
        w = float(self.lineEdit_7.text())
        x0 = re.findall("[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", x0_str)
        x0 = np.array([float(x) for x in x0])
        d = re.findall("[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", d_str)
        d = np.array([float(d_) for d_ in d])
        # x0 = np.array(nums_from_string.get_nums(x0))
        # d = np.array(nums_from_string.get_nums(d))
        # zabezpieczy?? przed z??ym podaniem danych, np. 0.5/3 daje dwie liczby
        data = {
            "func": func,
            "x0": x0,
            "d": d,
            "e1": e1,
            "e2": e2,
            "l": l,
            "w": w
        }
        if len(x0) != len(d):
            e = "Wektor warto??ci pocz??tkowej musi by?? tego samego wymiaru, co wektor kierunku poszukiwa??."
            raise Exception(e)
        return data

    def process_the_data(self, data):
        func_org = translate_input(data["func"])
        if func_org == None:
            raise Exception("Wprowadzone r??wnanie nie jest prawid??owe.")
        func_order = len(func_org.atoms(smp.Symbol))
        iso_func = smp.lambdify([x1, x2], func_org, 'numpy')
        x = np.array([x1, x2, x3, x4, x5])
        x0 = np.array([x0_value for x0_value in data["x0"]])  # data["x0"][0], data["x0"][1]])
        d = np.array([d_value for d_value in data["d"]])  # data["d"][0], data["d"][1]])
        if d[0] != 0:
            d = d / abs(max(abs(d)))  # normalize the direction
        else:
            d = d / abs(d[1])
        x_next = x0 + tau * d
        print(f"{x_next = }")  # ###############
        a = 0  # np.sqrt(np.sum(np.power(x0[0], 2))) # was x[0]

        w = data["w"]  # / np.sqrt(np.sum(np.power(d, 2)))
        c = a + w

        func_org = substitute(func=func_org, xes=x,
                              values=x_next)  # create one-variable function (of tau) in a given direction (sympified)
        func = smp.lambdify(tau, func_org, 'numpy')  # make one-variable function lambdified with respect to tau

        processed_data = {
            "order": func_order,
            "func": func,
            "iso_func": iso_func,
            "range": (a, c),
            "x0": x0,
            "d": d,
            "e1": data["e2"],
            "e2": data["e1"],
            "l": data["l"],
            "w": w,
            # for step_mode
            "iterations": 0,
            "steps_x": [[x] for x in list(x0)],
            "optimized_x": x0,
            "b": (a+c)/2,
            "tau": 0,
            "approx_func": None
        }
        return processed_data

    def run_algorithm(self, processed_data, step_mode=False):
        # define basic values
        # t = np.linspace(-100, 100, 1000)
        a = processed_data["range"][0]
        c = processed_data["range"][1]
        func = processed_data["func"]
        # iso_func = processed_data["iso_func"]
        x0 = processed_data["x0"]
        d = processed_data["d"]
        l = processed_data["l"]
        # w = processed_data["w"]
        e1 = processed_data["e1"]
        e2 = processed_data["e2"]

        # for step mode
        if step_mode:
            iteration = processed_data["iterations"]
            consequent_x = processed_data["steps_x"]
            b = processed_data["b"]
            tau = processed_data["tau"]
            approx_func = processed_data["approx_func"]
            opt_x = processed_data["optimized_x"]
            return algorithm_step(func=func, a=a, c=c, d=d, x0=x0, epsilon1=e1, epsilon2=e2, num_of_iterations=l,
                                  iteration=iteration, consequent_x=consequent_x, b=b, prev_tau=tau,
                                  prev_approx_func=approx_func, prev_optimized_x=opt_x)
        else:
            return algorithm(func=func, a=a, c=c, d=d, x0=x0, epsilon1=e1, epsilon2=e2, num_of_iterations=l)

    def plot_results(self, results, processed_data):
        func = processed_data["func"]
        iso_func = processed_data["iso_func"]
        x0 = processed_data["x0"]
        consequent_x = results["steps_x"]
        optimized_x = results["optimized_x"]
        iteration = results["iterations"]

        x_min = [min(consequent_x[0]), min(consequent_x[1])]
        x_max = [max(consequent_x[0]), max(consequent_x[1])]
        iso_range = [[x_min[0]-2, x_max[0]+2], [x_min[1]-2, x_max[1]+2]]

        self.plot_isohypse([iso_range[0], iso_range[1]], iso_func)  # plot isohypse
        plt.plot(consequent_x[0], consequent_x[1], linewidth=0.4)  # plot a line
        plt.scatter(consequent_x[0], consequent_x[1], c='b', s=20)  # plot every iteration's point
        plt.scatter(x0[0], x0[1], c='r', marker='x', s=80, alpha=1)  # plot the starting point
        plt.scatter(optimized_x[0], optimized_x[1], c='#00D100', marker='x', s=80)  # plot the result
        plt.title(f"Rozwi??zanie optymalne kierunku wynosi ({round(optimized_x[0], 3)}, {round(optimized_x[1], 3)}).\n"
                  f"Ilo???? iteracji: {iteration}.")
        self.canvas.draw()

    def plot_isohypse(self, xy_range, z_func):
        x = np.linspace(xy_range[0][0], xy_range[0][1], 100)
        y = np.linspace(xy_range[1][0], xy_range[1][1], 100)
        X, Y = np.meshgrid(x, y)
        Z = z_func(X, Y)
        self.show_plot(x_val=X, y_val=Y, z_val=Z)
        # plt.contour(X, Y, Z, 50, cmap="RdGy")

    def show_plot(self, x_val, y_val, z_val):
        # clear the canvas
        self.figure.clear()

        # create bar plot
        plt.contourf(x_val, y_val, z_val, 50, cmap="RdGy") #cmap="RdGy"
        plt.colorbar()

        # refresh canvas
        self.canvas.draw()

    def write_solution_to_textedit(self, results):
        round_value = 3
        opt_x = results["optimized_x"]
        solution_output_text = f"Liczba iteracji: {results['iterations']}\n"
        for i in range(opt_x.size):
            solution_output_text += f"x{i+1}: {round(opt_x[i], round_value)}\n"
        if results["end"] is not None:
            solution_output_text += f"\nWarunek zatrzymania: {results['end']}"
        solution_output_text += f"\nWarto???? funkcji w pkt. optymalnym: {round(results['func_val_at_min'], 5)}"

        self.output_text_window.setText(solution_output_text)
        # self.output_text_window.setText("Rozwi??zanie wynosi: " + str(results["optimized_x"]))

    def closeEvent(self, event):  # not needed
        self.popup = None

    def change_step_mode(self):
        self.step_mode = not self.step_mode

    def algorithm_thread_finished(self):
        self.algorithm_thread_started = False

    def show_approx_func(self, data):
        if self.popup is None:
            self.show_popup()

        resolution = 100
        a, c, b = data["range"][0], data["range"][1], data["b"]
        tau = data["tau"]
        func = data["org_func"]
        approx_func = data["approx_func"]
        x = np.linspace(a-abs(2*a), c+abs(2*c), resolution)
        y = func(x)
        y_approx = approx_func(x)

        pen_org = pg.mkPen(color='blue')
        pen_approx = pg.mkPen(color='red')

        self.approx_plot.clear()

        self.approx_plot.addLegend()
        org_plot = self.approx_plot.plot(x, y, pen=pen_org, name='funkcja celu')
        approx_plot = self.approx_plot.plot(x, y_approx, pen=pen_approx, name='aproksymacja kwadratowa')
        self.approx_plot.plot([a, c, b], [approx_func(a), approx_func(c), approx_func(b)], pen=None, symbol='o')
        self.approx_plot.plot([tau], [approx_func(tau)], pen=None, symbol='o',
                              symbolPen=pg.mkPen(color=(255, 0, 255), width=0),
                              symbolBrush=pg.mkBrush(0, 255, 255, 255), symbolSize=7)

    def close_event(self, event):
        self.popup = None

    def show_popup(self):
        self.popup = PopUp()
        self.popup.setWindowTitle("Kwadratowa aproksymacja funkcji")
        self.popup.resize(600, 700)
        self.popup.show()

        self.popup.closeEvent = lambda event: self.close_event(event)

        # add a layout
        self.horizontalLayout_approx = QtWidgets.QHBoxLayout()
        self.horizontalLayout_approx.setObjectName("horizontalLayoutApprox")

        # create and add the plot
        self.approx_plot = pg.PlotWidget()
        self.horizontalLayout_approx.addWidget(self.approx_plot)

        # styling
        self.approx_plot.setBackground('transparent')

        self.popup.setLayout(self.horizontalLayout_approx)


class PopUp(QWidget):
    def __init__(self):
        super().__init__()
    def closeEvent(self, event):
        self.close()

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
