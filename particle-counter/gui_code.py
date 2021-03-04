from PyQt5 import QtWidgets, uic, QtGui                                                       # GUI functions
import sys                                                                                    # For interacting with computer OS
from os import walk                                                                           # To get filepaths automatically
import numpy as np                                                                            # Maths
import matplotlib.pyplot as plt                                                               # Plotting
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas              # 'Canvas' widget for inserting pyplot into pyqt
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar      # Toolbar widget to accompany pyplot figure
import gui_counting_functions as cf                                                               # Custom functions for shape this program

# Set fake Windows App ID to trick taskbar into displaying icon
import ctypes
myappid = u'mycompany.myproduct.subproduct.version' # arbitrary string
ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)

# set image background colour to match grey of GUI
plt.rcParams.update({"figure.facecolor": '#f0f0f0'})

def zoom_factory(self,ax,base_scale = .8):
    """ Allow zooming with the scroll wheel on pyplot figures 
        Pass figure axes and a scrolling scale. 
        
        Based on https://stackoverflow.com/questions/11551049/matplotlib-plot-zooming-with-scroll-wheel """
    def zoom_fun(event):
        # get the current x and y limits
        cur_xlim = ax.get_xlim()
        cur_ylim = ax.get_ylim()
        cur_xrange = (cur_xlim[1] - cur_xlim[0])*.5
        cur_yrange = (cur_ylim[1] - cur_ylim[0])*.5
        xdata = event.xdata # get event x location
        ydata = event.ydata # get event y location
        
        if event.button == 'up':
            # deal with zoom in
            scale_factor = 1/base_scale
        elif event.button == 'down':
            # deal with zoom out
            scale_factor = base_scale
            
        # set new limits
        # zoom such that the cursor stays in the same position
        ax.figure.canvas.toolbar.push_current() # Ensure toolbar home stays the same
        ax.set_xlim([xdata - (xdata-cur_xlim[0]) / scale_factor, xdata + (cur_xlim[1]-xdata) / scale_factor]) 
        ax.set_ylim([ydata - (ydata-cur_ylim[0]) / scale_factor, ydata + (cur_ylim[1]-ydata) / scale_factor])

        self.canvas.draw()  # force re-draw

    fig = ax.get_figure() # get the figure of interest
    # attach the call back
    self.canvas.mpl_connect('scroll_event',zoom_fun) #QWheelEvent

    #return the function
    return zoom_fun

class Ui(QtWidgets.QMainWindow):
    """ User interface class """
    def __init__(self):
        """ Initialise attributes of the GUI """
        # Load GUI layout for QT designer .ui file
        super(Ui, self).__init__() # Call the inherited classes __init__ method
        uic.loadUi('gui_draft_2.ui', self) # Load the .ui file from QT Designer
        
        # Set logo
        self.setWindowIcon(QtGui.QIcon('logo.png'))
        
        # Create figure widget
        self.figure = plt.figure(figsize=(20,20)) 
        self.canvas = FigureCanvas(self.figure) # pass figure into canvas widget
        self.toolbar = NavigationToolbar(self.canvas, self) # add toolbar to canvas widget
   
        # Add figure widget into the GUI
        layout = self.verticalLayout_plotwindow # connect to the plot layout
        layout.addWidget(self.toolbar) # add toolbar to the layout 
        layout.addWidget(self.canvas) # add canvas to the layout 
        
        # Plot initial figure
        im = plt.imread('initial_pic.png')
        self.ax = self.figure.add_subplot(111)
        self.ax.imshow(im)
        self.plot()
            
        # Connect buttons to functions
        self.pushButton_fileinput.clicked.connect(self.file_input)
        self.pushButton_nextimage.clicked.connect(self.plot) 
        self.pushButton_updatescalenm.clicked.connect(self.update_calibration)
        self.pushButton_autoidentify.clicked.connect(self.auto_identify)
        
    def file_input(self):
        """ Connects to file input button
        Opens file explorer for user to select image then plots it """
        self.label_statusbar.setText(r'Status: Opening image')
        
        # Get filepath from file explorer
        self.fpath, _ = QtWidgets.QFileDialog.getOpenFileName(None, 'Open File', '*.tif')
        self.label_inputname.setText(self.fpath)
        
        # Clear current figure and plot selected image
        self.figure.clear() # clear old figure
        self.ax = self.figure.add_subplot(111) # create an axis
        self.im,self.raw_im,self.px,self.w,self.h = cf.read_tif(self.fpath) # load image
        self.ax.imshow(self.raw_im,cmap='Greys_r') # add image to axis
        self.plot() # plot axis
        
        # prompt user input if metadata is not present
        if np.isnan(self.px):
            self.lineEdit_scalenm.setEnabled(True)
            self.pushButton_updatescalenm.setEnabled(True)
            self.label_pxwidth.setText('Pixel width: Metadata not found')
            self.label_imdim.setText('Image dimensions: Metadata not found')
            self.label_statusbar.setText(r'Status: Metadata not found. Please enter the scalebar size in nm.')
        
        else:
            # Update labels
            self.lineEdit_scalenm.setEnabled(False)
            self.lineEdit_scalenm.setText(' ')
            self.pushButton_updatescalenm.setEnabled(False)
            self.label_pxwidth.setText('Pixel width: %.3f px per nm' % (self.px*1e9))
            self.label_imdim.setText('Image dimensions: %i nm x %i nm' % (self.w*1e9,self.h*1e9))
            self.label_statusbar.setText(r'Status: Idle')
        
    def plot(self):
        """ Plots the image that is currently assigned to self.ax 
        Image will be added to interactive canvas with scrollwheel zoom enabled """
        # image settings
        self.ax.axis('off')
        plt.tight_layout()
        
        # Plot with zoom functionality
        f = zoom_factory(self,self.ax,base_scale = .8)
        
        # refresh canvas 
        self.canvas.draw()
        
    def update_calibration(self):
        """ Runs automatic calibration based on scalebar size input by user """
        # Get user value
        scale_bar_val = int(self.lineEdit_scalenm.text())
        
        # Run calibration
        self.im,self.raw_im,self.px,self.w,self.h = cf.alternate_calibration(self.fpath,scale_bar_val = scale_bar_val)
        
        # Update labels
        self.label_pxwidth.setText('Pixel width: %.3f px per nm' % (self.px*1e9))
        self.label_imdim.setText('Image dimensions: %i nm x %i nm' % (self.w*1e9,self.h*1e9))
        self.label_statusbar.setText(r'Status: Idle')
        
    def auto_identify(self):
        print('Starting analysis')
        self.label_statusbar.setText(r'Status: Analysing image')
        self.label_statusbar.repaint()
        filters = []
        if self.checkBox_f1.isChecked():
            filters.append('blur_fill')
        if self.checkBox_f2.isChecked():
            filters.append('sobel')
        minsize = self.spinBox_minsize.value()
        maxsize = self.spinBox_maxsize.value()
        
        self.figure.clear() # clear old figure
        self.ax = self.figure.add_subplot(111) # create an axis
        print('Using ' + str(filters) + 'with min %i nm and max %i nm diameters' % (minsize,maxsize))
        ds,errs = cf.full_count_process(self.im,self.raw_im,self.px,self.w,self.h, self.ax, filters=filters, min_avg_length = minsize, max_avg_length = maxsize)
        self.plot()
        print('Finished')
        self.label_statusbar.setText(r'Status: Idle')
        
app = QtWidgets.QApplication(sys.argv) # Create an instance of QtWidgets.QApplication
window = Ui() # Create an instance of our class

# Show in fullscreen
window.resize(800,800) # workaround for FS
window.showMaximized()

app.exec_() # Start the application