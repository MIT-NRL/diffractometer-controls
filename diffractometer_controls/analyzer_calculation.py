import sys
import numpy as np
from pydm.display import Display
from qtpy import QtCore, QtGui
from qtpy.QtWidgets import (QVBoxLayout, QHBoxLayout, QGroupBox,
    QLabel, QLineEdit, QPushButton, QScrollArea, QFrame,
    QApplication, QWidget, QLabel, QMessageBox)
from bluesky_widgets.qt.run_engine_client import (
    QtReConsoleMonitor,
    QtReEnvironmentControls,
    QtReExecutionControls,
    QtReManagerConnection,
    QtRePlanEditor,
    QtRePlanHistory,
    QtRePlanQueue,
    QtReQueueControls,
    QtReRunningPlan,
    QtReStatusMonitor,
)

from bluesky_widgets.models.run_engine_client import RunEngineClient
import display

class AnalyzerCalcuation(display.MITRDisplay):
    def __init__(self, parent=None, args=None, macros=None, ui_filename='analyzer_calculation.ui'):
        super().__init__(parent, args, macros, ui_filename)
        # print("REScreen here")
        # self.customize_ui()

    def ui_filename(self):
        return 'analyzer_calculation.ui'

    def ui_filepath(self):
        return super().ui_filepath()

    def customize_ui(self):
        ##================================================================##
        #Add calculations to second tab for diffraction and focusing
        cutting511 = np.arccos(np.vdot([4,0,0],[5,1,1])/(np.linalg.norm([4,0,0])*np.linalg.norm([5,1,1])))*180/np.pi
        cutting311 = np.arccos(np.vdot([4,0,0],[3,1,1])/(np.linalg.norm([4,0,0])*np.linalg.norm([3,1,1])))*180/np.pi
        self.AnalyzerOptions = {'400':{'d_spacing':1.35765,'cutting_angle':0},'311':{'d_spacing':1.637433,'cutting_angle':cutting311},'511':{'d_spacing':1.0450421715709526,'cutting_angle':cutting511}}
        # self.AnalyzerOptions = {'hkl':['400','311','511'],'d_spacing':[1.35765,1.637433,1.0450421715709526]}
        self.comboBox_analyzer.addItems(self.AnalyzerOptions.keys())
        self.updateLineEdit_analyzer()

        #Connect analyzer comboBox selection with updating the connected actions
        self.comboBox_analyzer.activated.connect(self.updateLineEdit_analyzer)
        self.comboBox_sample.activated.connect(self.updateComboBox_sample)
        self.comboBox_sample_hkl.activated.connect(self.updateLineEdit_sample)

        #Add the samples to the comboBox
        self.SampleOptions = {'PG':{'006':1.1183},
                              'Fe':{'211':1.16939,'220':1.01273,'311':0.90581},
                              'Ni':{'311':1.06253,'220':1.24592,'222':1.01729},
                              'Al':{'111':2.33188,'200':2.01946,'220':1.42798,'311':1.23715,'222':1.1815,'400':1.00973},
                              'Graphite':{'002':3.348,'100':2.12696,'101':2.02715,'102':1.7953,'004':1.674,'110':1.2280,'112':1.1529,'006':1.116,'200':1.06348},
                              }
        self.comboBox_sample.addItem('PG',self.SampleOptions['PG'].keys())
        self.comboBox_sample.addItem('Fe',self.SampleOptions['Fe'].keys())
        self.comboBox_sample.addItem('Ni',self.SampleOptions['Ni'].keys())
        self.comboBox_sample.addItem('Al',self.SampleOptions['Al'].keys())
        self.comboBox_sample.addItem('Graphite',self.SampleOptions['Graphite'].keys())
        self.comboBox_sample_hkl.addItems(self.comboBox_sample.itemData(0))

 
        self.updateLineEdit_sample()

        #Include calculations of the Bragg angles
        self.lineEdit_2thetaS.setText('90')
        self.updateThetaA()
        self.updateLambda()
        self.lineEdit_2thetaS.textChanged.connect(self.updateThetaA)

        ##===============================================================##
        #Add calculations for focusing
        self.lineEdit_RV.setText('1.0')
        self.lineEdit_LSA.setText('1.5')
        self.updateVertCurv()
        self.updateHorizontalCur()
        self.updateHorizontalCurInv()
        self.updateAnalyzerTurns()
        # self.updateMagnfication()
      

        #Update the focusing conditions when values change
        self.lineEdit_RV.textChanged.connect(self.updateVertCurv)
        self.lineEdit_LSA.textChanged.connect(self.updateVertCurv)

    def updateHorizontalCur(self):
        #Updates the horizontal curvature of the analyzer
        if (self.lineEdit_RV.text() != '') and (self.lineEdit_LSA.text() != '') and (self.lineEdit_2thetaS.text() != '') and (self.lineEdit_RV.text() != ''):
            self.lineEdit_RH.clear()
            self.lineEdit_RH.setText(str(round(self.horizontal_focusing(
                d_sample=self.convertToFloat(self.lineEdit_sample_hkl_d),
                d_analyzer=self.convertToFloat(self.lineEdit_analyzer_d),
                theta_S=self.convertToFloat(self.lineEdit_2thetaS),
                theta_A=self.convertToFloat(self.lineEdit_2thetaA),
                theta_cutting=self.AnalyzerOptions[self.comboBox_analyzer.currentText()]['cutting_angle'],
                LSA=self.convertToFloat(self.lineEdit_LSA),
                LAD=self.convertToFloat(self.lineEdit_LAD),
            ),2)))
            self.updateHorizontalCurInv()
            self.updateAnalyzerTurns()
        else: 
            self.lineEdit_RH.setText('')

    def updateMagnfication(self):
        #calculates the magnification from the distances
        if (self.lineEdit_LSA.text() != '') and (self.lineEdit_LAD.text() != ''):
            self.lineEdit_mag.clear()
            self.lineEdit_mag.setText(str(round(
                self.convertToFloat(self.lineEdit_LAD) / self.convertToFloat(self.lineEdit_LSA)
                ,2)))
        else:
            self.lineEdit_mag.setText('')

    def updateHorizontalCurInv(self):
        #Calculates the inverse horizontal curvature
        if self.lineEdit_RH != '':
            self.lineEdit_RH_inv.clear()
            self.lineEdit_RH_inv.setText(str(round(
                1 / self.convertToFloat(self.lineEdit_RH)
            ,3)))
        else:
            self.lineEdit_RH_inv.setText('')
    
    def updateAnalyzerTurns(self):
        #Calculates the number of turns on the analyzer to create the horizontal curvature determined through focusing
        if self.lineEdit_RH_inv != '':
            self.lineEdit_analyzer_turns.clear()
            self.lineEdit_analyzer_turns.setText(str(round(
                1814.74772 * self.convertToFloat(self.lineEdit_RH_inv),1
            )))
        else: 
            self.lineEdit_analyzer_turns.setText('')

    def updateVertCurv(self):
        #Updates the calculation for the vertical curvature
        if (self.lineEdit_RV.text() != '') and (self.lineEdit_LSA.text() != '') and (self.lineEdit_2thetaS.text() != ''):
            self.lineEdit_LAD.clear()
            self.lineEdit_LAD.setText(str(round(self.vertical_focusing(
                d_sample=self.convertToFloat(self.lineEdit_sample_hkl_d),
                d_analyzer=self.convertToFloat(self.lineEdit_analyzer_d),
                theta_S=self.convertToFloat(self.lineEdit_2thetaS)/2,
                theta_A=self.convertToFloat(self.lineEdit_2thetaA)/2,
                theta_cutting=self.AnalyzerOptions[self.comboBox_analyzer.currentText()]['cutting_angle'],
                RV=self.convertToFloat(self.lineEdit_RV),
                LSA=self.convertToFloat(self.lineEdit_LSA)
            ),2)
            ))
        else:
            self.lineEdit_LAD.setText('')
        self.updateHorizontalCur()
        self.updateMagnfication()

    def updateThetaA(self):
        #Updates thetaA when thetaS is set
        if self.lineEdit_2thetaS.text() != '':
            self.lineEdit_2thetaA.clear()
            angle = self.BraggLaw(
                d_sample = self.convertToFloat(self.lineEdit_sample_hkl_d),
                d_analyzer = self.convertToFloat(self.lineEdit_analyzer_d),
                theta_s = self.convertToFloat(self.lineEdit_2thetaS)/2
                )
            self.lineEdit_2thetaA.setText(str(round(2*angle,3)))
        else:
            self.lineEdit_2thetaA.setText('')
        self.updateLambda()
        self.update180ThetaS()
        self.updateGonAngle()
        self.updateVertCurv()

    def update180ThetaS(self):
        if self.lineEdit_2thetaS.text() != '':
            self.lineEdit_180_2thetaa.clear()
            self.lineEdit_180_2thetaa.setText(str(round(180 - 
            self.convertToFloat(self.lineEdit_2thetaA),3)))
        else:
            self.lineEdit_180_2thetaa.clear()

    def updateGonAngle(self):
        if self.lineEdit_2thetaS.text() != '':
            self.lineEdit_GonA.clear()
            self.lineEdit_GonA.setText(str(round(
                self.convertToFloat(self.lineEdit_180_2thetaa)/2,2
            )))
        else:
            self.lineEdit_GonA.clear()

    def convertToFloat(self,input):
        # Converts the input to float and gives error message
        # Included string cleaning to remove any non-number symbols
        cleanInput = ''.join(c for i,c in enumerate(input.text()) if (c in '1234567890.') or ((c in '+-') and (i == 0)))
        # print(input.text(),cleanInput)
        if (cleanInput not in '+-') and (cleanInput != ''):
            try:
                return np.float64(cleanInput)
            except Exception:
                self.msg = QMessageBox()
                self.msg.about(self,'Error','Input can only be a number')
                return np.nan
        else:
            return np.nan
        


    def updateLambda(self):
        #calculates lambda with the given conditions
        if self.lineEdit_2thetaS.text() != '':
            self.lineEdit_lambda.clear()
            lambdaValue = (2 * self.convertToFloat(self.lineEdit_sample_hkl_d)
                * np.sin(self.convertToFloat(self.lineEdit_2thetaS)/2 * np.pi/180)
                )
            self.lineEdit_lambda.setText(str(round(lambdaValue,5)) + ' Å')
        else:
            self.lineEdit_lambda.clear()

    def BraggLaw(self,d_sample=1,theta_s=None,d_analyzer=None,theta_a=None):
        if theta_s is not None:
            return 180/np.pi * np.arcsin(d_sample/d_analyzer * np.sin(theta_s * np.pi/180))
        if theta_a is not None:
            return 180/np.pi * np.arcsin(d_analyzer/d_sample * np.sin(theta_a * np.pi/180))

    def vertical_focusing(self,d_sample=1.0,d_analyzer=1.0,theta_S=45.0,theta_A=45.0,theta_cutting=0.0,RV=1.0,LSA=1.5):
        #Vertical focusing condition
        #Calculates LAD to focus the beam
        return ((2 * np.cos(np.pi/180 * theta_cutting) * np.sin(np.pi/180 * theta_A)) / RV - 1/LSA) **-1

    def horizontal_focusing(self,d_sample=1.0,d_analyzer=1.0,theta_S=45.0,theta_A=45.0,theta_cutting=0.0,LSA=1.5,LAD=1.5):
        theta_A /= 2
        theta_S /= 2
        kappa = 1 - np.tan(np.pi/180 * theta_A)/np.tan(np.pi/180 * theta_S)
        xi = np.cos(np.pi/180*(theta_A-theta_cutting)) / np.cos(np.pi/180*(theta_A+theta_cutting))
    
        return 2*np.cos(np.pi/180 * (theta_A + theta_cutting)) / ( (1+kappa)*np.sin(np.pi/180*(2*theta_A))) * (kappa * LAD + xi * LSA)

    def updateLineEdit_analyzer(self):
        #Updates the d-spacing for the analyzer
        self.lineEdit_analyzer_d.clear()
        # comboIndex = self.comboBox_analyzer.currentIndex()
        self.lineEdit_analyzer_d.setText(str(round(self.AnalyzerOptions[self.comboBox_analyzer.currentText()]['d_spacing'],5))+' Å')
        self.updateThetaA()

    def updateComboBox_sample(self,index):
        #Updates the sample hkl items
        self.comboBox_sample_hkl.clear()
        self.comboBox_sample_hkl.addItems(self.comboBox_sample.itemData(index))
        self.updateLineEdit_sample()

    def updateLineEdit_sample(self):
        #Updates the d-spacing of the sample
        self.lineEdit_sample_hkl_d.clear()
        self.lineEdit_sample_hkl_d.setText(str(round(self.SampleOptions[self.comboBox_sample.currentText()][self.comboBox_sample_hkl.currentText()],6))+' Å')
        self.updateThetaA()