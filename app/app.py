'''
ECG application (classification and interpretation).
Author: Yadong Zhang
E-mail: zhangyadong@stu.xjtu.edu.cn

Demo:
$ python app.py
'''
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='Times New Roman')
import matplotlib.patches as patches
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

import time
import pickle
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

import sys
sys.path.append('../')
from lib.util import TMF_image, get_GCAM, get_sym_GCAM
from lib.util import build_fullnet as build

StyleSheet = '''
#RedProgressBar {
    text-align: center;
    min-height: 12px;
    max-height: 12px;
}
#RedProgressBar::chunk {
    background-color: #F44336;
}
'''

DETAIL = {0: 'Load data', 10: 'Load model', 20: 'Get TMF image', 30: 'Get Grad-CAM of AF', 55: 'Get symmetrized Grad-CAM of AF', 60: 'Get Grad-CAM of non-AF', 85: 'Get symmetrized Grad-CAM of non-AF', 90: 'Predict', 100: 'Finished'}  # detail of the process status

class TSCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        '''
        ECG time series
        '''
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(TSCanvas, self).__init__(fig)

class HMCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        '''
        Symmetrized Grad-CAM of TMF images
        '''
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(HMCanvas, self).__init__(fig)

class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, ts, gcam_AF, gcam_nAF, proba_AF, proba_nAF, label):

        super().__init__()
        self.gcam_AF = gcam_AF
        self.gcam_nAF = gcam_nAF
        self.ts = ts
        self.setFixedSize(640, 800)
        win = len(self.ts)  # time series length
        D = 3  # triad
        self.overlap = win-(D-1)*self.gcam_AF.shape[0]  # overlap of the heatmap
        
        # ECG time series, initial plot
        self.sc1 = HMCanvas(self, width=4.0, height=5, dpi=100)
        self.sc1.axes.plot(np.arange(len(self.ts)), self.ts, 'k', linewidth=1.0)
        self.sc1.axes.set_xlim([0, len(self.ts)])
        self.sc1.axes.set_yticks([])
        # Heatmap, initial plot
        self.sc2 = HMCanvas(self, width=4.0, height=5, dpi=100)
        self.sc2.axes.imshow(self.gcam_AF, cmap=plt.cm.jet)
        self.sc2.axes.set_xticks([])
        self.sc2.axes.set_yticks([])
        # Heatmap, initial plot
        self.sc3 = HMCanvas(self, width=4.0, height=5, dpi=100)
        self.sc3.axes.imshow(self.gcam_nAF, cmap=plt.cm.jet)
        self.sc3.axes.set_xticks([])
        self.sc3.axes.set_yticks([])
        
        # Slider for gap
        self.s=QSlider(Qt.Horizontal)
        self.s.setMinimum(1)
        self.s.setMaximum(self.gcam_AF.shape[0])
        self.s.setSingleStep(1)
        self.s.setValue(20)
        self.s.setTickPosition(QSlider.TicksBelow)
        self.s.setTickInterval(100)
        self.s.valueChanged.connect(self.triadchange)  # connect with the triadchange function to redraw the plots
        
        # Slider for initial time index of triad
        self.idx=QSlider(Qt.Horizontal)
        self.idx.setMinimum(0)
        self.idx.setMaximum(self.gcam_AF.shape[1]-1)
        self.idx.setSingleStep(1)
        self.idx.setValue(20)
        self.idx.setTickPosition(QSlider.TicksBelow)
        self.idx.setTickInterval(100)
        self.idx.valueChanged.connect(self.triadchange)
        
        # Labels
        self.l0 = QLabel('Label: %s'%('AF' if label == 1 else 'non-AF'))
        self.l0.setAlignment(Qt.AlignCenter)
        self.imp1 = QLabel('Probability of AF:%.3f'%proba_AF)
        self.imp1.setAlignment(Qt.AlignCenter)
        self.imp2 = QLabel('Probability of non-AF:%.3f'%proba_nAF)
        self.imp2.setAlignment(Qt.AlignCenter)
        if proba_AF > proba_nAF:
            self.imp1.setStyleSheet("background-color: red") 
        elif proba_AF < proba_nAF:
            self.imp2.setStyleSheet("background-color: red") 
        self.l1 = QLabel('Time index:')
        self.l1.setAlignment(Qt.AlignCenter)
        self.l2 = QLabel('Delay:')
        self.l2.setAlignment(Qt.AlignCenter)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.l0)  # label
        layout.addWidget(self.sc1)  # Time series
        layout.addWidget(self.imp1)  # Label: probability of AF
        layout.addWidget(self.sc2)  # Heatmap
        layout.addWidget(self.imp2)  # Label: probability of non-AF
        layout.addWidget(self.sc3)  # Heatmap
        layout.addWidget(self.l1)  # Label: initial time index
        layout.addWidget(self.idx)  # Time index slider
        layout.addWidget(self.l2)  #  LabeL: gap
        layout.addWidget(self.s)  # Gap slider
        # Create a placeholder widget to hold our toolbar and canvas.
        widget = QtWidgets.QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)
        self.show()
        self.setLayout(layout)

    def triadchange(self):
        '''
        Re-draw the triad, time series and heatmap
        '''
        start = self.idx.value()
        gap = self.s.value()
        
        # Re-draw the heatmap
        self.sc2.axes.cla()  # Clear the canvas.
        self.sc2.axes.imshow(self.gcam_AF, cmap=plt.cm.jet)
        # Change color of + in heatmap
        self.sc2.axes.scatter(x=[start], y=[gap-1], s=100, color='w', marker='+')
        self.sc2.axes.set_xlim([0, self.gcam_AF.shape[1]])
        self.sc2.axes.set_ylim([self.gcam_AF.shape[0], 0])
        self.sc2.axes.set_xticks([])
        self.sc2.axes.set_yticks([])
        # Trigger the canvas to update and redraw.
        self.sc2.draw()
        
        # Re-draw the heatmap
        self.sc3.axes.cla()  # Clear the canvas.
        self.sc3.axes.imshow(self.gcam_nAF, cmap=plt.cm.jet)
        # Change color of + in heatmap
        self.sc3.axes.scatter(x=[start], y=[gap-1], s=100, color='w', marker='+')
        self.sc3.axes.set_xlim([0, self.gcam_nAF.shape[1]])
        self.sc3.axes.set_ylim([self.gcam_nAF.shape[0], 0])
        self.sc3.axes.set_xticks([])
        self.sc3.axes.set_yticks([])
        # Trigger the canvas to update and redraw.
        self.sc3.draw()
        
        # Fix the symmetrized triad
        right_bound = len(self.ts)-(3-1)*gap
        if start + 1> right_bound:
            start = self.gcam_AF.shape[1] - start - 1
            gap = self.gcam_AF.shape[0] - gap + 1
        
        # Re-draw time series
        self.sc1.axes.cla()  # Clear the canvas.
        self.sc1.axes.plot(np.arange(len(self.ts)), self.ts, 'k', linewidth=1.0)
        self.sc1.axes.plot(np.arange(start, start+gap*3, gap), self.ts[np.arange(start, start+gap*3, gap)], 'ro-', markersize=5)
        self.sc1.axes.set_xlim([0, len(self.ts)])
        self.sc1.axes.set_yticks([])
        self.sc1.draw()
        
        # Set new text for labels
        self.l1.setText('Time index:'+str(start))
        self.l2.setText('Delay:'+str(gap))

    def closeEvent(self, event):
        
        reply = QMessageBox.question(self, 'Warning',
            "Sure to exit?", QMessageBox.Yes | 
            QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.hide()
            self.dialog = StartWindow()
        else:
            event.ignore()   

class Thread(QThread):
    _signal = pyqtSignal(int)
    def __init__(self):
        super(Thread, self).__init__()

    def __del__(self):
        self.wait()

    def run(self):
        pnums = list(DETAIL.keys())
        pnums.sort()
        
        self._signal.emit(pnums[0])
        net = build()
        with open(ecg_path, 'rb') as f:
            data = np.load(f)
        with open(label_path, 'rb') as f:
            label = pickle.load(f)
        self._signal.emit(pnums[1])
        net = build()
        self._signal.emit(pnums[2])
        # AF signal
        ts = data[IDX]
        val_y = label['Y_test'][IDX]
        
        D = 3
        shape = np.array([len(range(1, (len(ts)-1)//(D-1) + 1)), len(range(0, len(ts)-(D-1)*1)), D])
        overlap = len(ts)-(D-1)*shape[0]
        # TMF image: [1, W, H, 3]
        img = np.zeros(shape)
        img = TMF_image(ts, overlap, img, D)
        img = np.expand_dims(img, axis=0)
        self._signal.emit(pnums[3])
        # SG-CAM image of non-AF
        gcam = get_GCAM(net, img, [1,0], layers=-3)
        self._signal.emit(pnums[4])
        gcam_norm = get_sym_GCAM(overlap, gcam, D)
        nAF_gcam = gcam_norm.copy()
        self._signal.emit(pnums[5])
        # SG-CAM image of AF
        gcam = get_GCAM(net, img, [0,1], layers=-3)
        self._signal.emit(pnums[6])
        gcam_norm = get_sym_GCAM(overlap, gcam, D)
        AF_gcam = gcam_norm.copy()
        self._signal.emit(pnums[7])
        proba = net.predict(img)
        print('Predicted probabilities:', proba)
        nAF_proba, AF_proba = proba[0]
        global RES
        RES = [ts, AF_gcam, nAF_gcam, AF_proba, nAF_proba, val_y]
        self._signal.emit(pnums[8])

class StartWindow(QWidget):
    
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):               
        
        pixmap = QPixmap("img.jpg")

        # Start menu image
        lbl = QLabel(self)
        lbl.setPixmap(pixmap)
        self.setFixedSize(pixmap.width(),pixmap.height())
        self.parentSize = [pixmap.width(),pixmap.height()]
        self.center()
        
        # Process bar
        self.progress = QProgressBar(self, objectName="RedProgressBar")
        self.progress.move(pixmap.width()//2-50, pixmap.height()-150)
        self.progress.resize(100, 20)
        # Start button
        self.startBtn = QPushButton("Start", self)
        self.startBtn.move(pixmap.width()//2-50, pixmap.height()-100)
        self.startBtn.clicked.connect(self.on_pushButton_clicked)
        self.startBtn.resize(100, 50)
        # Title
        self.label = QLabel(self)
        self.label.setFixedWidth(800)
        self.label.setFixedHeight(100)
        self.label.move((pixmap.width()-self.label.width())/2, (pixmap.height()-self.label.height())/2)
        self.label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setText(u"Triadic Motif Field")
        self.label.setAutoFillBackground(False)
        self.label.setFont(QFont("Roman times", 30, QFont.Bold))

        self.setWindowTitle('TMF: Interpretable Classification Model')    
        self.show()

    def on_pushButton_clicked(self):
        self.thread = Thread()
        self.thread._signal.connect(self.signal_accept)
        self.thread.start()
        self.startBtn.setEnabled(False)

    def signal_accept(self, msg):
        self.progress.setValue(int(msg))
        self.label.setText(DETAIL[int(msg)])
        print(DETAIL[int(msg)])
        if self.progress.value() == 100:
            self.hide()
            self.progress.setValue(0)
            global RES
            self.dialog = MainWindow(*RES)
            
    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())
        
    def closeEvent(self, event):
        
        reply = QMessageBox.question(self, 'Warning',
            "Sure to exit?", QMessageBox.Yes | 
            QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()        

if __name__ == '__main__':

    if len(sys.argv) > 1:
        IDX = int(sys.argv[1])
    else: # default value 
        IDX = 3736  # AF signal
    
    ecg_path = '../data/ECG_X_test.bin'  # 2-d array, [N, dim]
    label_path = '../data/ECG_info.pkl'  # dict, key='Y_test', 0 and 1 indicate non-AF and AF
    app = QApplication(sys.argv)
    app.setStyleSheet(StyleSheet)
    ex = StartWindow()
    sys.exit(app.exec_())