# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'Visu.ui'
##
## Created by: Qt User Interface Compiler version 6.8.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QAction, QBrush, QColor, QConicalGradient,
    QCursor, QFont, QFontDatabase, QGradient,
    QIcon, QImage, QKeySequence, QLinearGradient,
    QPainter, QPalette, QPixmap, QRadialGradient,
    QTransform)
from PySide6.QtWidgets import (QApplication, QCheckBox, QDoubleSpinBox, QFrame,
    QGraphicsView, QLabel, QMainWindow, QMenu,
    QMenuBar, QPushButton, QRadioButton, QSizePolicy,
    QStatusBar, QWidget)

from xypad import XYPad

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(974, 721)
        self.actionGeometry = QAction(MainWindow)
        self.actionGeometry.setObjectName(u"actionGeometry")
        self.actionModel = QAction(MainWindow)
        self.actionModel.setObjectName(u"actionModel")
        self.actionBoundary_Conditions = QAction(MainWindow)
        self.actionBoundary_Conditions.setObjectName(u"actionBoundary_Conditions")
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.graphicsView = QGraphicsView(self.centralwidget)
        self.graphicsView.setObjectName(u"graphicsView")
        self.graphicsView.setGeometry(QRect(520, 20, 331, 331))
        self.figureWidget = QWidget(self.centralwidget)
        self.figureWidget.setObjectName(u"figureWidget")
        self.figureWidget.setGeometry(QRect(20, 20, 471, 441))
        self.stress_checkbox = QCheckBox(self.centralwidget)
        self.stress_checkbox.setObjectName(u"stress_checkbox")
        self.stress_checkbox.setGeometry(QRect(70, 480, 92, 24))
        self.vmin_spin = QDoubleSpinBox(self.centralwidget)
        self.vmin_spin.setObjectName(u"vmin_spin")
        self.vmin_spin.setGeometry(QRect(170, 500, 111, 27))
        self.vmin_spin.setMinimum(-10.000000000000000)
        self.vmin_spin.setMaximum(0.000000000000000)
        self.vmin_spin.setSingleStep(0.250000000000000)
        self.vmax_spin = QDoubleSpinBox(self.centralwidget)
        self.vmax_spin.setObjectName(u"vmax_spin")
        self.vmax_spin.setGeometry(QRect(310, 500, 111, 27))
        self.vmax_spin.setMinimum(0.000000000000000)
        self.vmax_spin.setMaximum(10.000000000000000)
        self.vmax_spin.setSingleStep(0.250000000000000)
        self.vmax_spin.setValue(1.000000000000000)
        self.spin_x = QDoubleSpinBox(self.centralwidget)
        self.spin_x.setObjectName(u"spin_x")
        self.spin_x.setGeometry(QRect(540, 530, 121, 27))
        self.spin_x.setMinimum(-2.000000000000000)
        self.spin_x.setMaximum(2.000000000000000)
        self.spin_x.setSingleStep(0.100000000000000)
        self.spin_y = QDoubleSpinBox(self.centralwidget)
        self.spin_y.setObjectName(u"spin_y")
        self.spin_y.setGeometry(QRect(700, 430, 121, 27))
        self.spin_y.setMinimum(-1.000000000000000)
        self.spin_y.setMaximum(4.000000000000000)
        self.spin_y.setSingleStep(0.100000000000000)
        self.xy_pad = XYPad(self.centralwidget)
        self.xy_pad.setObjectName(u"xy_pad")
        self.xy_pad.setGeometry(QRect(520, 370, 151, 141))
        self.bnd_a = QRadioButton(self.centralwidget)
        self.bnd_a.setObjectName(u"bnd_a")
        self.bnd_a.setGeometry(QRect(540, 620, 61, 24))
        self.line_2 = QFrame(self.centralwidget)
        self.line_2.setObjectName(u"line_2")
        self.line_2.setGeometry(QRect(520, 600, 118, 3))
        self.line_2.setFrameShape(QFrame.Shape.HLine)
        self.line_2.setFrameShadow(QFrame.Shadow.Sunken)
        self.label = QLabel(self.centralwidget)
        self.label.setObjectName(u"label")
        self.label.setGeometry(QRect(660, 590, 131, 18))
        self.line_3 = QFrame(self.centralwidget)
        self.line_3.setObjectName(u"line_3")
        self.line_3.setGeometry(QRect(800, 590, 161, 21))
        self.line_3.setFrameShape(QFrame.Shape.HLine)
        self.line_3.setFrameShadow(QFrame.Shadow.Sunken)
        self.bnd_b = QRadioButton(self.centralwidget)
        self.bnd_b.setObjectName(u"bnd_b")
        self.bnd_b.setGeometry(QRect(610, 620, 61, 24))
        self.bnd_c = QRadioButton(self.centralwidget)
        self.bnd_c.setObjectName(u"bnd_c")
        self.bnd_c.setGeometry(QRect(690, 620, 61, 24))
        self.bnd_d = QRadioButton(self.centralwidget)
        self.bnd_d.setObjectName(u"bnd_d")
        self.bnd_d.setGeometry(QRect(760, 620, 61, 24))

        self.check_solution_button = QPushButton(self.centralwidget)
        self.check_solution_button.setObjectName(u"check_solution_button")
        self.check_solution_button.setGeometry(QRect(70, 590, 131, 26))
        self.spin_poisson = QDoubleSpinBox(self.centralwidget)
        self.spin_poisson.setObjectName(u"spin_poisson")
        self.spin_poisson.setGeometry(QRect(790, 530, 121, 27))
        self.spin_poisson.setMinimum(0.150000000000000)
        self.spin_poisson.setMaximum(0.450000000000000)
        self.spin_poisson.setSingleStep(0.010000000000000)
        self.spin_poisson.setValue(0.300000000000000)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 974, 23))
        self.menuModel = QMenu(self.menubar)
        self.menuModel.setObjectName(u"menuModel")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.menubar.addAction(self.menuModel.menuAction())
        self.menuModel.addAction(self.actionGeometry)
        self.menuModel.addAction(self.actionModel)
        self.menuModel.addAction(self.actionBoundary_Conditions)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.actionGeometry.setText(QCoreApplication.translate("MainWindow", u"Geometry", None))
        self.actionModel.setText(QCoreApplication.translate("MainWindow", u"Model", None))
        self.actionBoundary_Conditions.setText(QCoreApplication.translate("MainWindow", u"Boundary Conditions", None))
        self.stress_checkbox.setText(QCoreApplication.translate("MainWindow", u"Stresses", None))
        self.vmin_spin.setPrefix(QCoreApplication.translate("MainWindow", u"vmin = ", None))
        self.vmax_spin.setPrefix(QCoreApplication.translate("MainWindow", u"vmax = ", None))
        self.spin_x.setPrefix(QCoreApplication.translate("MainWindow", u"x displ. = ", None))
        self.spin_y.setPrefix(QCoreApplication.translate("MainWindow", u"y displ. = ", None))
        self.bnd_a.setText(QCoreApplication.translate("MainWindow", u"a", None))
        self.label.setText(QCoreApplication.translate("MainWindow", u"Active Boundaries", None))
        self.bnd_b.setText(QCoreApplication.translate("MainWindow", u"b", None))
        self.bnd_c.setText(QCoreApplication.translate("MainWindow", u"c", None))
        self.bnd_d.setText(QCoreApplication.translate("MainWindow", u"d", None))
        self.check_solution_button.setText(QCoreApplication.translate("MainWindow", u"Check Solution", None))
        self.spin_poisson.setPrefix(QCoreApplication.translate("MainWindow", u"poisson = ", None))
        self.menuModel.setTitle(QCoreApplication.translate("MainWindow", u"File", None))
    # retranslateUi

