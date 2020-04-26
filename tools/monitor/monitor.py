import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QHBoxLayout, QVBoxLayout, QPushButton, QLayout, QTabWidget, QSizePolicy, QTextEdit, QRadioButton, QButtonGroup, QSlider, QListWidget
from PyQt5.QtGui import QIcon, QImage
from PyQt5.QtCore import Qt
from PyQt5 import QtGui
from PyQt5.QtCore import pyqtSlot, QTimer

from PIL import Image
import os
import numpy as np
import json
from functools import partial

import load_normals

from matplotlib.pyplot import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvas
import matplotlib.pyplot as plt

from classes.bsdf_training_graph import BSDFTrainingGraphPlotter, MultiSelector, elementToString
from classes.crosssection import CrossSectionDisplayLayout
from classes.radial_overlay import RadialErrorView

import optparse

import cv2
#import ...awsmanager as awsmanager
#exec(open(os.path.dirname(__file__) +  "/../awsmanager.py", "r").read())

import importlib.util
spec = importlib.util.spec_from_file_location("module.name", os.path.dirname(__file__) +  "/../awsmanager.py")
awsmanager = importlib.util.module_from_spec(spec)
spec.loader.exec_module(awsmanager)


ENVMAPS = ["doge", "field", "uffizi"]

def parseBSDFFile(path, maxsearch=100):
    fullbsdf = {}
    for i in range(maxsearch):
        print("Searching: " + path + "-" + format(i) + ".txt")
        if os.path.exists(path + "-" + format(i) + ".txt"):
            print("Found " + path + "-" + format(i) + ".txt")
            lines = open(path + "-" + format(i) + ".txt", "r").readlines()
            theta_in = [line for line in lines if line.startswith("#theta_in")]
            theta_in = float(theta_in[0][9:])
            thetaless = [line for line in lines if not line.startswith("#")]
            allvals = [[ float(val) for val in line.split(" ")] for line in thetaless]
            theta_out = [vals[0] for vals in allvals if vals[1] == -180]
            values = [vals[2] * np.cos(theta_in * (3.1415/180.0)) for vals in allvals if vals[1] == -180]
            bsdfslice = {}
            for to, val in zip(theta_out, values):
                bsdfslice[to] = val
            fullbsdf[theta_in] = bsdfslice
        else:
            break

    # No entries, bsdf files don't exist.
    if len(fullbsdf) == 0:
        return None

    return fullbsdf

def parseNDFFile(path, maxsearch=100):
    ndffile = path + "-ndf.txt"
    if not os.path.exists(ndffile):
        return None
    lines = open(ndffile, "r").readlines()
    allvals = [[ float(val) for val in line.split(" ")] for line in lines]
    theta_out = [vals[0] for vals in allvals]
    values = [vals[1] * np.cos(t_out * (3.1415/180.0)) for vals, t_out in zip(allvals, theta_out)]

    fullndf = {}
    for to, val in zip(theta_out, values):
        fullndf[to] = val
    #print(fullndf)
    return fullndf

    """
    fullbsdf = {}
    for i in range(maxsearch):
        print("Searching: " + path + "-" + format(i) + ".txt")
        
            print("Found " + path + "-" + format(i) + ".txt")
            lines = open(path + "-" + format(i) + ".txt", "r").readlines()
            theta_in = [line for line in lines if line.startswith("#theta_in")]
            theta_in = float(theta_in[0][9:])
            thetaless = [line for line in lines if not line.startswith("#")]
            allvals = [[ float(val) for val in line.split(" ")] for line in thetaless]
            theta_out = [vals[0] for vals in allvals if vals[1] == -180]
            values = [vals[2] * np.cos(theta_in * (3.1415/180.0)) for vals in allvals if vals[1] == -180]
            bsdfslice = {}
            for to, val in zip(theta_out, values):
                bsdfslice[to] = val
            fullbsdf[theta_in] = bsdfslice
        else:
            break

    # No entries, bsdf files don't exist.
    if len(fullbsdf) == 0:
        return None
    return fullbsdf
    """

class DeltaImage:
    def __init__(self, label, initdata, factor = 1.0):
        self.label = label
        self.shape = initdata.shape

        self.data = initdata

        self.imageWidget = QLabel()
        self.imageWidget.setAlignment(Qt.AlignCenter)

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.imageWidget)

        self.labelWidget = QLabel(self.label)
        self.labelWidget.setAlignment(Qt.AlignCenter)

        self.layout.addWidget(self.labelWidget)

        self.factor = factor

        self.updateData()


    def updateData(self):
        self.labelWidget.setText(self.label)

        self.fdata = self.data * self.factor

        if len(self.fdata.shape) == 2:
            self._imdat = np.tile(self.fdata.reshape((self.fdata.shape[0], self.fdata.shape[1], 1)), [1,1,4]).astype('uint8')
        elif len(self.fdata.shape) == 3:
            self._imdat = np.concatenate((self.fdata, np.ones((self.fdata.shape[0], self.fdata.shape[1], 1))), axis=2).astype('uint8')
        else:
            print("Invalid numpy image shape: " + format(self.fdata.shape))
            assert(0)

        self._qimg = QtGui.QImage(
            self._imdat.data,
            self.fdata.shape[0], self.fdata.shape[1],
            QtGui.QImage.Format_RGB32)
        self._pix = QtGui.QPixmap.fromImage(self._qimg)
        self.imageWidget.setPixmap(self._pix)

    def setData(self, data):
        self.data = data
        self.updateData()

    def setLabel(self, label):
        self.label = label
        self.updateData()

    def getLayout(self):
        return self.layout
    
    def showLayout(self):
        self.imageWidget.show()
        self.labelWidget.show()
        self.layout.show()

    def setFactor(self, factor):
        self.factor = factor
        self.updateData()

class StandardImage:
    def __init__(self, label, filename, factor = 1.0):
        self.label = label

        self.filename = filename

        self.imageWidget = QLabel()
        self.imageWidget.setAlignment(Qt.AlignCenter)

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.imageWidget)

        self.labelWidget = QLabel(self.label)
        self.labelWidget.setAlignment(Qt.AlignCenter)

        self.layout.addWidget(self.labelWidget)

        self.factor = factor

        self.updateData()

    def updateData(self):
        print(self.filename)
        self.labelWidget.setText(self.label)
        self.imageWidget.setPixmap(QtGui.QPixmap(self.filename))

    def setFilename(self, filename):
        self.filename = filename
        self.updateData()

    def setLabel(self, label):
        self.label = label
        self.updateData()

    def getLayout(self):
        return self.layout
    
    def showLayout(self):
        self.imageWidget.show()
        self.labelWidget.show()
        self.layout.show()
    
    def setFactor(self, factor):
        self.factor = factor
        self.updateData()


class DeltaView(QLayout):
    def __init__(self, title):
        self.title = title
        self.layout = QVBoxLayout()
        
        self.imagesetLayout = QHBoxLayout()
        self.images = []

        self.titleLabel = QLabel(self.title)
        self.titleLabel.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.titleLabel)

        self.layout.addLayout(self.imagesetLayout)

        self.layout.addStretch(1)

        self.widget = QWidget()
        self.widget.setLayout(self.layout)

    def setImageData(self, index, data):
        self.images[index].setData(data)

    def setFactor(self, factor):
        for image in self.images:
            image.setFactor(factor)

    def addImage(self, imageview):
        self.images.append(imageview)
        self.imagesetLayout.addLayout(imageview.getLayout())

    def getLayout(self):
        return self.layout

    def getWidget(self):
        return self.widget


class ScrubbingControls:
    def __init__(self, title, label, maxindex, parent=None, scrubTimeout=100):
        self.title = title
        self.maxindex = maxindex
        self.index = 0
        self.labelText = label

        self.nextButton = QPushButton("Next", parent)
        #self.nextButton.clicked.connect(self._nextCallback)
        self.nextButton.pressed.connect(self._nextCallbackRepeatPressed)
        self.nextButton.released.connect(self._nextCallbackRepeatReleased)

        self.lastButton = QPushButton("Last", parent)
        self.lastButton.clicked.connect(self._lastCallback)
        #self.lastButton.pressed.connect(self._lastCallbackRepeatPressed)
        #self.lastButton.released.connect(self._lastCallbackRepeatReleased)

        self.prevButton = QPushButton("Prev", parent)
        #self.prevButton.clicked.connect(self._prevCallback)
        self.prevButton.pressed.connect(self._prevCallbackRepeatPressed)
        self.prevButton.released.connect(self._prevCallbackRepeatReleased)

        self.centerLabel = QLabel(self.labelText + " 0/" + format(maxindex))
        self.centerLabel.setAlignment(Qt.AlignCenter)

        self.layout = QHBoxLayout()
        self.layout.addWidget(self.prevButton)
        self.layout.addWidget(self.centerLabel)
        self.layout.addWidget(self.nextButton)
        self.layout.addWidget(self.lastButton)

        self.nextTimer = QTimer()
        self.nextTimer.timeout.connect(self.__nextTimerFired)
        self.prevTimer = QTimer()
        self.prevTimer.timeout.connect(self.__prevTimerFired)

        self.scrubTimeout = scrubTimeout
        self.callback = None

    def getLayout(self):
        return self.layout

    def setIndexChangedCallback(self, callback):
        self.callback = callback

    def setMaxIndex(self, index):
        self.maxindex = index
        if self.index >= self.maxindex:
            self.index = self.maxindex
            self.callback(self.index)
        self.updateLabel()

    def updateLabel(self):
        self.centerLabel.setText(self.labelText + " " + format(self.index) + "/" + format(self.maxindex))

    def __prevTimerFired(self):
        self._prevCallback()

    def __nextTimerFired(self):
        self._nextCallback()

    def _nextCallbackRepeatPressed(self):
        self._nextCallback()
        self.nextTimer.start(self.scrubTimeout)

    def _nextCallbackRepeatReleased(self):
        self.nextTimer.stop()

    def _prevCallbackRepeatPressed(self):
        self._prevCallback()
        self.prevTimer.start(self.scrubTimeout)

    def _prevCallbackRepeatReleased(self):
        self.prevTimer.stop()

    def _nextCallback(self):
        print("CLICKED!")
        if self.index < self.maxindex:
            self.index += 1

        self.updateLabel()
        if self.callback:
            self.callback(self.index)

    def _lastCallback(self):
        print("CLICKED!")
        #if self.index < self.maxindex:
        self.index = self.maxindex

        self.updateLabel()
        if self.callback:
            self.callback(self.index)

    def _prevCallback(self):
        print("CLICKED!")
        if self.index > 0:
            self.index -= 1

        self.updateLabel()
        if self.callback:
            self.callback(self.index)


class ExposureControl:
    def __init__(self, title, label, maxval=3, minval=-3, initial=0, interval=0.1, parent=None, scrubTimeout=100):
        self.title = title
        self.minval = minval
        self.maxval = maxval
        self.value = initial
        self.labelText = label
        self.interval = interval

        """
        self.nextButton = QPushButton("Next", parent)
        #self.nextButton.clicked.connect(self._nextCallback)
        self.nextButton.pressed.connect(self._nextCallbackRepeatPressed)
        self.nextButton.released.connect(self._nextCallbackRepeatReleased)

        self.lastButton = QPushButton("Last", parent)
        self.lastButton.clicked.connect(self._lastCallback)
        #self.lastButton.pressed.connect(self._lastCallbackRepeatPressed)
        #self.lastButton.released.connect(self._lastCallbackRepeatReleased)

        self.prevButton = QPushButton("Prev", parent)
        #self.prevButton.clicked.connect(self._prevCallback)
        self.prevButton.pressed.connect(self._prevCallbackRepeatPressed)
        self.prevButton.released.connect(self._prevCallbackRepeatReleased)

        self.centerLabel = QLabel(self.labelText + " 0/" + format(maxindex))
        self.centerLabel.setAlignment(Qt.AlignCenter)

        self.layout = QHBoxLayout()
        self.layout.addWidget(self.prevButton)
        self.layout.addWidget(self.centerLabel)
        self.layout.addWidget(self.nextButton)
        self.layout.addWidget(self.lastButton)

        self.nextTimer = QTimer()
        self.nextTimer.timeout.connect(self.__nextTimerFired)
        self.prevTimer = QTimer()
        self.prevTimer.timeout.connect(self.__prevTimerFired)

        self.scrubTimeout = scrubTimeout
        self.callback = None
        """
        
        
        self.centerLabel = QLabel(self.labelText + ": " + format(np.exp2(self.value/50)))
        self.centerLabel.setAlignment(Qt.AlignCenter)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMaximum(self.maxval)
        self.slider.setMinimum(self.minval)
        self.slider.setValue(self.value)
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.setTickInterval(self.interval)
        self.slider.valueChanged.connect(self.__valueChanged)

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.centerLabel)
        self.layout.addWidget(self.slider)

    def getLayout(self):
        return self.layout

    def setValueChangedCallback(self, callback):
        self.callback = callback

    def updateLabel(self):
        self.centerLabel.setText(self.labelText + ": " + format(np.exp2(self.value/10)))

    def __valueChanged(self):
        self.value = self.slider.value()
        self.updateLabel()
        self.callback(self.slider.value())

class ServerSelectView:
    def __init__(self, title, label, awsKeyFile, awsKeyName, parent=None):
        self.title = title
        self.labelText = label

        self.refreshButton = QPushButton("Refresh", parent)
        self.refreshButton.clicked.connect(self._refreshCallback)

        self.awsKeyFile = awsKeyFile
        self.awsKeyName = awsKeyName

        self.centerLabel = QLabel(self.labelText)
        self.centerLabel.setAlignment(Qt.AlignCenter)
        self.listWidget = QListWidget()
        self.listWidget.itemClicked.connect(self._listCallback)

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.centerLabel)
        self.layout.addWidget(self.listWidget)
        self.layout.addWidget(self.refreshButton)

        self.widget = QWidget()
        self.widget.setLayout(self.layout)

    def _refreshCallback(self):
        self.refreshServers()

    def refreshServers(self):
        self.centerLabel.setText("Refreshing...")
        self.listWidget.clear()
        for server in awsmanager.getAllServers(self.awsKeyName):
            datasets = awsmanager.getServerDatasets(server, self.awsKeyFile)
            for dataset in datasets:
                if dataset["datetime"] is not None:
                    self.listWidget.addItem(server + " " + dataset["name"] + " " + dataset["datetime"].strftime("%I:%M%p %A %d. %B %Y"))
                else:
                    self.listWidget.addItem(server + " " + dataset["name"] + " Time Unavailable")
            if len(datasets) == 0:
                self.listWidget.addItem(server + " FREE")
        self.centerLabel.setText(self.labelText)

    def getLayout(self):
        return self.layout

    def setServerChangedCallback(self, callback):
        self.callback = callback

    def updateLabel(self):
        self.centerLabel.setText(self.labelText + ": " + format(np.exp2(self.value/10)))

    def _listCallback(self, item):
        self.listWidget.setCurrentItem(item)
        if len(item.text().split(" ")) > 2:
            #localpath = os.path.dirname(item.text().split(" ")[2].split(" ")[-1])
            remotepath = "/home/ubuntu/outputs/" + item.text().split(" ")[1]
            self.callback({"server": item.text().split(" ")[0], "directory": remotepath, "valid": True})
        else:
            self.callback({"server": item.text().split(" ")[0], "directory": "", "valid": False})

    def getWidget(self):
        return self.widget

class ImageLoader:
    def __init__(self, directory):

        self.directory = directory
        self.cache = {}
        self.filters = {}

        self.initialize()

    def initialize(self, resetCache=True, resetFilters=True):
        self._analyzeVersion(self.directory)

        if not self.vSinglePartConfig:
            params = json.load(open(self.directory + "/inputs/normals.json", "r"))
            self.lightsFile = self.directory + "/inputs/" + params["lights"]["file"]
        else:
            params = json.load(open(self.directory + "/inputs/config.json", "r"))
            self.lightsFile = self.directory + "/inputs/" + params["lights"]["file"]

        cparams = json.load(open(self.directory + "/inputs/config.json", "r"))
        meshname = "notfound"
        if "mesh" in cparams["target"]:
            meshname = cparams["target"]["mesh"]
        elif "refmesh" in cparams["target"]:
            meshname = cparams["target"]["refmesh"]

        self.referenceMesh = self.directory + "/inputs/" + meshname

        if resetCache:
            self.cache = {}

        if resetFilters:
            self.filters = {}

    def changeDirectory(self, directory):
        print("Changing source directory: " + directory)
        self.directory = directory
        self.initialize(resetCache=True, resetFilters=False) # Reinitialize all variables, 
                                                             # and clear caches, but retain filters.

    def _identityFilter(self, data):
        return data

    def cachedLoad(self, filename):
        if filename not in self.cache:
            self.cache[filename] = np.load(filename)
            return self.cache[filename]
        else:
            return self.cache[filename]

    def _analyzeVersion(self, directory):
        self.vSinglePartConfig = not os.path.exists(directory + "/inputs/normals.json")
        self.vDifferenceErrorsExist = os.path.exists(directory + "/images/difference-errors")
        self.vSamplersExist = os.path.exists(directory + "/images/samplers") and os.path.exists(directory + "/images/samplers/npy/00/0000-img-00.npy")
        self.vRendersExist = os.path.exists(directory + "/renders") and os.path.exists(directory + "/renders/gradients/00/0000-img00.p.npy")
        self.vBSDFErrorsExist = os.path.exists(directory + "/bsdfs") and (os.path.exists(directory + "/bsdfs/bsdfs-00-uffizi-diff.npy") or os.path.exists(directory + "/bsdfs/bsdfs-00-00-uffizi-diff.npy"))
        self.vCurrentsExist = os.path.exists(directory + "/images/current") and os.path.exists(directory + "/images/current/npy/00/0000-img-00.npy")
        self.vSingleBounceExist = os.path.exists(directory + "/images/single-bounce-currents") and os.path.exists(directory + "/images/single-bounce-currents/00/0000-00.npy")
        self.vSingleBounceRendersExist = os.path.exists(directory + "/renders/single-bounce-gradients") and os.path.exists(directory + "/renders/single-bounce-gradients/00/0000-00.npy")

        print("vRendersExist: ", self.vRendersExist)
        print("vDifferenceErrorsExist: ", self.vDifferenceErrorsExist)
        print("vSinglePartConfig: ", self.vSinglePartConfig)
        print("vSingleBounceExist: ", self.vSingleBounceExist)
        print("vSingleBounceRendersExist: ", self.vSingleBounceRendersExist)

    def _getFilter(self, st):
        if st in self.filters:
            return self.filters[st]
        else:
            return self._identityFilter

    def setFilter(self, st, filter):
        self.filters[st] = filter

    def getLightCount(self):
        lightlines = open(self.lightsFile, "r").readlines()
        return len(lightlines)

    def getLight(self, index):
        lightlines = open(self.lightsFile, "r").readlines()
        return lightlines[index]

    def getIterationCount(self, superindex=0):
        if os.path.exists(self.directory + "/errors/errors-" + format(superindex).zfill(2) + ".json"):
            edata = json.load(open(self.directory + "/errors/errors-" + format(superindex).zfill(2) + ".json", "r"))
            return len(edata["ierrors"]) + self.getBSDFIterationCount()
        else:
            return json.load(open(self.directory + "/inputs/config.json", "r"))["estimator"]["iterations"] + self.getBSDFIterationCount()

    def getBSDFIterationCount(self, superindex=0):
        if os.path.exists(self.directory + "/errors/bsdf-errors-" + format(superindex).zfill(2) + ".json"):
            edata = json.load(open(self.directory + "/errors/bsdf-errors-" + format(superindex).zfill(2) + ".json", "r"))
            return len(edata["ierrors"])
        else:
            return json.load(open(self.directory + "/inputs/config.json", "r"))["bsdf-estimator"]["iterations"]

    def getSuperIterationCount(self):
        nsi = len([f for f in os.listdir(self.directory + "/errors") if f.endswith(".json") and f.startswith("errors-")])
        bsi = len([f for f in os.listdir(self.directory + "/errors") if f.endswith(".json") and f.startswith("bsdf-errors-")])
        return np.max([nsi, bsi])

    def getFullConfigText(self):
        if not self.vSinglePartConfig:
            return open(self.directory + "/inputs/normals.json", "r").read() + "\n\n" + open(self.directory + "/inputs/config.json", "r").read()
        else:
            return open(self.directory + "/inputs/config.json", "r").read()

    def loadImageErrorImage(self, lindex, iterindex, superindex=0):
        # Load an error image based on light index and iteration index
        _filter = self._getFilter("light-error-image")
        imdata = None
        if self.vDifferenceErrorsExist:
            imdata = self.cachedLoad(
                    self.directory + "/images/difference-errors/npy/" + format(superindex).zfill(2) + "/" + format(iterindex).zfill(4) +
                    "-img-" + format(lindex).zfill(2) + ".npy")
        else:
            imdata = self.cachedLoad(
                    self.directory + "/images/normalized-errors/npy/" + format(superindex).zfill(2) + "/" + format(iterindex).zfill(4) +
                    "-img-" + format(lindex).zfill(2) + ".npy")

        imdata = np.array(Image.fromarray(imdata).resize((256, 256), Image.ANTIALIAS))
        return _filter(imdata)

    def loadSamplerImage(self, lindex, iterindex, superindex=0):
        # Load an error image based on light index and iteration index
        _filter = self._getFilter("sampler-image")
        imdata = None
        if self.vSamplersExist:
            imdata = self.cachedLoad(
                    self.directory + "/images/samplers/npy/" + format(superindex).zfill(2) + "/" + format(iterindex).zfill(4) +
                    "-img-" + format(lindex).zfill(2) + ".npy")
        else:
            imdata = np.zeros((256, 256))

        imdata = np.array(Image.fromarray(imdata).resize((256, 256), Image.ANTIALIAS))
        return _filter(imdata)

    def loadBSDFErrorImage(self, superindex=0, iterindex=0, envmap="uffizi"):
        # Load an error image based on light index and iteration index
        _filter = self._getFilter("bsdf-error-image")
        imdata = None
        if self.vBSDFErrorsExist:
            imdata = self.cachedLoad(
                    self.directory + "/bsdfs/bsdfs-" + format(superindex).zfill(2) + "-" + format(iterindex).zfill(2) + "-" + envmap + "-diff.npy")
        else:
            imdata = np.zeros((256, 256))

        return _filter(imdata)

    def getDimensions(self, lindex, iterindex, superindex=0):
        imdata = self.cachedLoad(
                    self.directory + "/images/current/npy/" + format(superindex).zfill(2) + "/" + format(iterindex).zfill(4) +
                    "-img-" + format(lindex).zfill(2) + ".npy")
        return imdata.shape

    def loadCurrentImage(self, lindex, iterindex, superindex=0):
        # Load an error image based on light index and iteration index
        _filter = self._getFilter("current-image")
        #imdata = None
        if self.vCurrentsExist:
            imdata = self.cachedLoad(
                    self.directory + "/images/current/npy/" + format(superindex).zfill(2) + "/" + format(iterindex).zfill(4) +
                    "-img-" + format(lindex).zfill(2) + ".npy")
        else:
            imdata = np.zeros((256, 256))

        imdata = np.array(Image.fromarray(imdata).resize((256, 256), Image.ANTIALIAS))
        return _filter(imdata)
    
    def loadRawCurrentImage(self, lindex, iterindex, superindex=0):
        if self.vCurrentsExist:
            return self.cachedLoad(
                self.directory + "/images/current/npy/" + format(superindex).zfill(2) + "/" + format(iterindex).zfill(4) +
                "-img-" + format(lindex).zfill(2) + ".npy")
        else:
            return np.zeros((256, 256))
    
    def loadRawImageErrorImage(self, lindex, iterindex, superindex=0):
        # Load an error image based on light index and iteration index
        imdata = None
        if self.vDifferenceErrorsExist:
            imdata = self.cachedLoad(
                    self.directory + "/images/difference-errors/npy/" + format(superindex).zfill(2) + "/" + format(iterindex).zfill(4) +
                    "-img-" + format(lindex).zfill(2) + ".npy")
        else:
            imdata = self.cachedLoad(
                    self.directory + "/images/normalized-errors/npy/" + format(superindex).zfill(2) + "/" + format(iterindex).zfill(4) +
                    "-img-" + format(lindex).zfill(2) + ".npy")

        return imdata

    def loadRawSingleBounceCurrentImage(self, lindex, iterindex, superindex=0):
        if self.vSingleBounceExist:
            return self.cachedLoad(
                self.directory + "/images/single-bounce-currents/" + format(superindex).zfill(2) + "/" + format(iterindex).zfill(4) +
                "-" + format(lindex).zfill(2) + ".npy")
        else:
            return np.zeros((256, 256))

    def loadFinalHighSPImage(self, lindex):
        # Load an error image based on light index and iteration index
        _filter = self._getFilter("current-image")
        #imdata = None
        if self.vCurrentsExist:
            imdata = self.cachedLoad(
                    self.directory + "/images/current-highsp/npy/final-" + format(lindex).zfill(2) + ".npy")
            imdata = imdata[:,:,0]
        else:
            imdata = np.zeros((256, 256))

        imdata = np.array(Image.fromarray(imdata).resize((256, 256), Image.ANTIALIAS))
        return _filter(imdata)

    def loadReferenceImage(self, lindex, superindex=0):
        # Load a reference image.
        _filter = self._getFilter("reference-image")
        if os.path.exists(self.directory + "/targets/npy/target-image-" + format(lindex).zfill(2)+ ".npy"):
            imdata = self.cachedLoad(
                self.directory + "/targets/npy/target-image-" + format(lindex).zfill(2)+ ".npy")
            return _filter(imdata)
        else:
            #return None
            return _filter(self.cachedLoad(self.directory + "/inputs/target.npy")[:,:,lindex])
    
    def loadRawReferenceImage(self, lindex, superindex=0):
        # Load a reference image.
        if os.path.exists(self.directory + "/targets/npy/target-image-" + format(lindex).zfill(2)+ ".npy"):
            return self.cachedLoad(
                self.directory + "/targets/npy/target-image-" + format(lindex).zfill(2)+ ".npy")
        else:
            return self.cachedLoad(self.directory + "/inputs/target.npy")[:,:,lindex]

    def loadNormalImage(self, iterindex, superindex=0):
        # Load a normal error image based on light index and iteration index.
        print(self.getNormalImagePath(iterindex, superindex, filetype="png"))
        if not self.vRendersExist or (not os.path.exists(self.getNormalImagePath(iterindex, superindex, filetype="png"))):
            return np.zeros((256, 256, 3))
        print("Found:", self.getNormalImagePath(iterindex, superindex, filetype="png"))
        _filter = self._getFilter("normal-image")
        #imdata = self.cachedLoad(
        #        self.getNormalImagePath(iterindex, superindex))
        imdata = cv2.imread(self.getNormalImagePath(iterindex, superindex, filetype="png"))
        #plt.imshow(_filter(imdata))
        #plt.show()
        return _filter(imdata)

    def loadGradientImage(self, lindex, iterindex, superindex=0):
        # Load a gradient map image.
        if not self.vRendersExist or (not os.path.exists(self.getGradientImagePath(lindex, iterindex, superindex, polarity="p"))):
            return np.zeros((256, 256, 3))

        _filter = self._getFilter("gradient-image")
        imdata = self.cachedLoad(
                    self.getGradientImagePath(lindex, iterindex, superindex, polarity="p")) - self.cachedLoad(
                    self.getGradientImagePath(lindex, iterindex, superindex, polarity="n"))
        return _filter(imdata)

    def loadRawGradientImage(self, lindex, iterindex, superindex=0):
        return np.zeros((256, 256, 3))
        # Load a gradient map image.
        if not self.vRendersExist or (not os.path.exists(self.getGradientImagePath(lindex, iterindex, superindex, polarity="p"))):
            return np.zeros((256, 256, 3))

        _filter = self._getFilter("raw-gradient-image")
        gdata = self.cachedLoad(
                    self.getGradientImagePath(lindex, iterindex, superindex, polarity="p")) - self.cachedLoad(
                    self.getGradientImagePath(lindex, iterindex, superindex, polarity="n"))

        errdata = self.cachedLoad(
                    self.directory + "/images/difference-errors/npy/" + format(superindex).zfill(2) + "/" + format(iterindex).zfill(4) +
                    "-img-" + format(lindex).zfill(2) + ".npy")

        return _filter(np.divide(gdata, np.tile(errdata.reshape([errdata.shape[0], errdata.shape[1], 1]), [1,1,3])))

    def loadNormalDeltaImage(self, iterindex, superindex=0):
        # Load a normal delta image.
        if (not self.vRendersExist) or (not os.path.exists(self.getNormalDeltasPath(iterindex, superindex, polarity="p"))):
            return np.zeros((256, 256, 3))
        _filter = self._getFilter("normal-delta-image")

        imdata = self.cachedLoad(
                    self.getNormalDeltasPath(iterindex, superindex, polarity="p")) - self.cachedLoad(
                    self.getNormalDeltasPath(iterindex, superindex, polarity="n"))
        
        return _filter(imdata)

    def loadTotalGradientImage(self, iterindex, superindex=0):
        # Load a normal delta image.
        if (not self.vRendersExist) or (not os.path.exists(self.getTotalGradientImagePath(iterindex, superindex, polarity="p"))):
            print("couldn't find total gradients..")
            return np.zeros((256, 256, 3))
        _filter = self._getFilter("total-gradient-image")

        imdata = self.cachedLoad(
                    self.getTotalGradientImagePath(iterindex, superindex, polarity="p")) - self.cachedLoad(
                    self.getTotalGradientImagePath(iterindex, superindex, polarity="n"))

        return _filter(imdata)

    def loadSingleBounceGradientImage(self, lindex, iterindex, superindex=0):
        if (not self.vRendersExist) or (not os.path.exists(self.getSingleBounceGradientImagePath(lindex, iterindex, superindex))):
            print("couldn't find total gradients..")
            return np.zeros((256, 256, 3))
        _filter = self._getFilter("total-gradient-image")

        imdata = self.cachedLoad(self.getSingleBounceGradientImagePath(lindex, iterindex, superindex))

        return _filter(imdata)
    
    def loadTargetNormals(self):
        path = self.directory + "/inputs/target-normals.npy"
        if os.path.exists(path):
            return np.load(path)


    def isTotalGradientsAvailable(self):
        if (not self.vRendersExist) or (not os.path.exists(self.getTotalGradientImagePath(0, 0, polarity="p"))):
            return False
        return True

    def getRemeshedMesh(self, superindex):
        #return self.directory + "/meshes/remeshed/" + format(superindex).zfill(2) + ".ply"
        return self.directory + "/meshes/normals/" + format(superindex).zfill(2) + "/0000.ply"

    def getNormalsMesh(self, iterindex, superindex=0):
        #return self.directory + "/meshes/remeshed/" + format(superindex).zfill(2) + ".ply"
        return self.directory + "/meshes/normals/" + format(superindex).zfill(2) + "/" + format(iterindex).zfill(4) + ".ply"

    def getTotalGradientMesh(self, iterindex, superindex):
        #return self.directory + "/meshes/remeshed/" + format(superindex).zfill(2) + ".ply"
        return self.directory + "/meshes/totalgradients/" + format(superindex).zfill(2) + "/" + format(iterindex).zfill(4) + ".ply"

    def getGradientMesh(self, lindex, iterindex, superindex):
        #return self.directory + "/meshes/remeshed/" + format(superindex).zfill(2) + ".ply"
        return self.directory + "/meshes/gradients/" + format(superindex).zfill(2) + "/" + format(iterindex).zfill(4) + "-img" + format(lindex).zfill(2) + ".ply"

    def getSingleBounceGradientMesh(self, lindex, iterindex, superindex):
        #return self.directory + "/meshes/remeshed/" + format(superindex).zfill(2) + ".ply"
        return self.directory + "/meshes/single-bounce-gradients/" + format(superindex).zfill(2) + "/" + format(iterindex).zfill(4) + "-" + format(lindex).zfill(2) + ".ply"

    def getNormalImagePath(self, iterindex, superindex=0, filetype="npy"):
        return self.directory + "/renders/normals/" + format(superindex).zfill(2) + "/" + format(iterindex).zfill(4) + "." + filetype

    def getNormalDeltasPath(self, iterindex, superindex=0, polarity="p"):
        if polarity not in ["p", "n"]:
            print("Error: No such polarity '" + polarity + "'")
            assert(0)
        return self.directory + "/renders/normaldeltas/" + format(superindex).zfill(2) + "/" + format(iterindex).zfill(4) + "." + polarity + ".npy"

    def getTotalGradientImagePath(self, iterindex, superindex=0, polarity="p"):
        if polarity not in ["p", "n"]:
            print("Error: No such polarity '" + polarity + "'")
            assert(0)
        return self.directory + "/renders/totalgradients/" + format(superindex).zfill(2) + "/" + format(iterindex).zfill(4) + "." + polarity + ".npy"

    def getSingleBounceGradientImagePath(self, lindex, iterindex, superindex=0):
        return self.directory + "/renders/single-bounce-gradients/" + format(superindex).zfill(2) + "/" + format(iterindex).zfill(4) + "-" + format(lindex).zfill(2) + ".npy"

    def getGradientImagePath(self, lindex, iterindex, superindex=0, polarity="p"):
        if polarity not in ["p", "n"]:
            print("Error: No such polarity '" + polarity + "'")
            assert(0)
        return self.directory + "/renders/gradients/" + format(superindex).zfill(2) + "/" + format(iterindex).zfill(4) + "-img" + format(lindex).zfill(2) + "." + polarity + ".npy"

    def getReferenceMesh(self):
        return self.referenceMesh

# uses matplotlib
class GraphPlotter:
    def __init__(self, directory, superindex=0):
        self.directory = directory
        self.canvas = FigureCanvas(Figure(figsize=(5, 3)))
        self.superIndex = superindex
        self.layout = QVBoxLayout()
        self.titleLabel = QLabel("Graph")

        self.titleLabel.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.titleLabel)
        self.layout.addWidget(self.canvas)
        self.canvas.setSizePolicy(QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding))

        #self.layout.addStretch(1)

        self.subplot = self.canvas.figure.subplots()
        self.lineindex = 0

        self.widget = QWidget()
        self.widget.setLayout(self.layout)

    def changeDirectory(self, directory):
        self.directory = directory

    def plot(self):
        errorPath = self.directory + "/errors/errors-" + format(self.superIndex).zfill(2) + ".json"

        if not os.path.exists(errorPath):
            return

        # TODO: Likely performance issues with this statement.
        self.errordata = json.load(open(errorPath, "r"))

        nerrors = self.errordata["nerrors"]
        ierrors = self.errordata["ierrors"]

        print(len(nerrors), self.lineindex)
        self.subplot.plot(range(len(nerrors)), nerrors)
        self.subplot.plot(range(len(ierrors)), ierrors)

        self.subplot.axvline(x=self.lineindex)
        # TODO: Set legend and stuff.

    def updatePlot(self, index, superindex):
        print("Updating plot")
        self.subplot.cla()
        self.lineindex = index
        self.superIndex = superindex
        self.plot()
        self.canvas.draw()

    def getLayout(self):
        return self.layout
    
    def getWidget(self):
        return self.widget



# Plots bsdf w.r.t input angle.
class BSDFPlotter:
    def __init__(self, directory, superindex=0, targets=[]):
        self.directory = directory
        self.canvas = FigureCanvas(Figure(figsize=(5, 3)))
        self.superIndex = superindex
        self.layout = QVBoxLayout()
        self.titleLabel = QLabel("BSDF Plot")

        self.titleLabel.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.titleLabel)
        self.layout.addWidget(self.canvas)
        self.canvas.setSizePolicy(QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding))

        #self.layout.addStretch(1)

        self.subplot = self.canvas.figure.subplots()
        self.lineindex = 0

        self.superindices = []

        self.widget = QWidget()
        self.widget.setLayout(self.layout)

        # Assume targets do not change between super iterations.
        self.targets = targets

        self.enabled = True
        self.weightsEnabled = True

        self.mode = "BSDF"

    def setMode(self, mode):
        if mode in ["BSDF", "NDF"]:
            self.mode = mode
        else:
            print("ERROR Invalid mode: ",  mode)

    def changeDirectory(self, directory):
        self.directory = directory

    def disable(self):
        self.enabled = False

    def enable(self):
        self.enabled = True

    def disableWeights(self):
        self.weightsEnabled = False

    def enableWeights(self):
        self.weightsEnabled = True

    def _plotBSDF(self):

        # TODO: Likely performance issues with this statement.
        config = json.load(open(self.directory + "/inputs/config.json", "r"))
        params = config
        directory = self.directory

        currentBSDF = parseBSDFFile(directory + "/bsdfs/exports/bexport-" + format(self.superIndex).zfill(2) + "-" + format(self.index).zfill(4))
        targetBSDF = parseBSDFFile(directory + "/bsdfs/exports/bexport-target")
        bestBSDF = parseBSDFFile(directory + "/bsdfs/exports/bexport-final")
        tableBSDF = parseBSDFFile(directory + "/bsdfs/exports/bexport-table")

        legend = []
        minX = 1000
        maxX = 0
        for i, ti in enumerate(list(currentBSDF.keys())[::2]):
            legend += ["Our Method $\\theta_i=" + format(ti) + "\\degree$"]
            minX = min([minX] + list(currentBSDF[ti].keys()))
            maxX = max([maxX] + list(currentBSDF[ti].keys()))
            self.subplot.plot(currentBSDF[ti].keys(), currentBSDF[ti].values(), color='C' + format(i))

            legend += ["Ground Truth $\\theta_i=" + format(ti) + "\\degree$"]
            minX = min([minX] + list(targetBSDF[ti].keys()))
            maxX = max([maxX] + list(targetBSDF[ti].keys()))
            self.subplot.plot(targetBSDF[ti].keys(), targetBSDF[ti].values(), linestyle='--', color='C' + format(i))

            if tableBSDF is not None:
                legend += ["Tabular Fit $\\theta_i=" + format(ti) + "\\degree$"]
                self.subplot.plot(tableBSDF[ti].keys(), tableBSDF[ti].values(), linestyle=':', color='C' + format(i))

        self.subplot.xaxis.set_ticks(np.arange(minX, maxX, 5.0))
        self.subplot.grid()
        self.subplot.legend(legend)
        self.subplot.set_xlabel("$\\theta_o$ (degrees)")
        self.subplot.set_ylabel(r'$f_r(\theta_i,\phi_i=0,\theta_o,\phi_o=0)$ ($sr^{-1}$)')
        # self.canvas.figure.savefig("current.png")
        # TODO: Set legend and stuff.

    def _mirrorValues(self, vec):
        return list(vec)[::-1] + list(vec)
    def _invMirrorValues(self, vec):
        return [ -v for v in list(vec)[::-1] ] + list(vec)

    def _plotNDF(self):
        print("Plotting NDF")
        # TODO: Likely performance issues with this statement.
        config = json.load(open(self.directory + "/inputs/config.json", "r"))
        params = config
        directory = self.directory

        # NDF loading.
        # TODO: FINISH
        currentNDF = parseNDFFile(directory + "/bsdfs/exports/bexport-" + format(self.superIndex).zfill(2) + "-" + format(self.lineindex).zfill(4))
        targetNDF = parseNDFFile(directory + "/bsdfs/exports/bexport-target")
        bestNDF = parseNDFFile(directory + "/bsdfs/exports/bexport-final")
        tableNDF = parseNDFFile(directory + "/bsdfs/exports/bexport-table")

        legend = []
        #minX = min([minX] + list(currentNDF.keys()))
        #maxX = max([maxX] + list(currentNDF.keys()))
        if currentNDF is not None:
            legend += ["Our Method"]
            self.subplot.plot(self._invMirrorValues(currentNDF.keys()), self._mirrorValues(currentNDF.values()))

        #minX = min([minX] + list(targetNDF.keys()))
        #maxX = max([maxX] + list(targetNDF.keys()))
        if targetNDF is not None:
            legend += ["Ground Truth"]
            self.subplot.plot(self._invMirrorValues(targetNDF.keys()), self._mirrorValues(targetNDF.values()), linestyle='--')

        if tableNDF is not None:
            legend += ["Tabular Fit"]
            self.subplot.plot(self._invMirrorValues(tableNDF.keys()), self._mirrorValues(tableNDF.values()), linestyle=':')

    def plot(self):
        print("Plotting BSDF Curves")
        if not self.enabled:
            print("BSDF plots disabled.")
            return
        
        if self.mode == "BSDF":
            self._plotBSDF()
        elif self.mode == "NDF":
            self._plotNDF()

    def updatePlot(self, index, superindex):
        print("Updating plot")
        self.subplot.cla()
        self.lineindex = index
        self.superIndex = superindex
        self.plot()
        self.canvas.draw()

    def getLayout(self):
        return self.layout
    
    def getWidget(self):
        return self.widget


# Plots bsdf parameter values against their targets.
class BSDFGraphPlotter:
    
    def __init__(self, directory, superindex=0, targets=[]):
        self.directory = directory
        self.canvas = FigureCanvas(Figure(figsize=(5, 3)))
        self.superIndex = superindex
        self.layout = QVBoxLayout()
        self.titleLabel = QLabel("BSDF Graph")

        self.titleLabel.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.titleLabel)
        self.layout.addWidget(self.canvas)
        self.canvas.setSizePolicy(QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding))

        #self.layout.addStretch(1)

        self.subplot = self.canvas.figure.subplots()
        self.lineindex = 0

        self.superindices = []

        self.widget = QWidget()
        self.widget.setLayout(self.layout)

        # Assume targets do not change between super iterations.
        self.targets = targets

        self.enabled = True
        self.weightsEnabled = True


    def changeDirectory(self, directory):
        self.directory = directory
        self.updatePlot(0, self.superIndex)

    def disable(self):
        self.enabled = False

    def enable(self):
        self.enabled = True

    def disableWeights(self):
        self.weightsEnabled = False

    def enableWeights(self):
        self.weightsEnabled = True

    def plot(self):

        if not self.enabled:
            print("BSDF plots disabled.")
            return

        # TODO: Likely performance issues with this statement.
        config = json.load(open(self.directory + "/inputs/config.json", "r"))
        params = config
        directory = self.directory

        if type(params["hyper-parameter-list"]) is list:
            runnerParameterList = params["hyper-parameter-list"]
        else:
            runnerParameterList = json.load(open(directory + "/" + params["hyper-parameter-list"], "r"))
            params["hyper-parameter-list"] = runnerParameterList

        if "hyper-parameters" in params["original"] and type(params["original"]["hyper-parameters"]) is not dict:
            runnerParameterValues = np.load(directory + "/inputs/" + params["original"]["hyper-parameters"])
            params["original"]["hyper-parameters"] = dict(zip(runnerParameterList, runnerParameterValues))

        if "hyper-parameters" in params["estimator"] and type(params["estimator"]["hyper-parameters"]) is not dict:
            runnerParameterValues = np.load(directory + "/inputs/" + params["estimator"]["hyper-parameters"])
            params["estimator"]["hyper-parameters"] = dict(zip(runnerParameterList, runnerParameterValues))

        if "hyper-parameters" in params["bsdf-estimator"] and type(params["bsdf-estimator"]["hyper-parameters"]) is not dict:
            runnerParameterValues = np.load(directory + "/inputs/" + params["bsdf-estimator"]["hyper-parameters"])
            params["bsdf-estimator"]["hyper-parameters"] = dict(zip(runnerParameterList, runnerParameterValues))

        config = params

        # Add padding parameters if necessary
        # Zero padding.
        if "zero-padding" in params and type(params["zero-padding"]) is int:
            zeroPadding = parameters["zero-padding"]
        elif "zero-padding" in params:
            zeroPadding = json.load(open(directory + "/" + params["zero-padding"], "r"))[0]
        else:
            zeroPadding = 0

        if type(config["bsdf-estimator"]["update-list"]) is not list:
            config["bsdf-estimator"]["update-list"] = list(runnerParameterList)

        for i in range(zeroPadding):
            config["hyper-parameter-list"].append("padding" + format(i))
            params["bsdf-estimator"]["hyper-parameters"]["padding" + format(i)] = 0
            params["estimator"]["hyper-parameters"]["padding" + format(i)] = 0
            if "hyper-parameters" in params["original"]:
                params["original"]["hyper-parameters"]["padding" + format(i)] = 0

        # Load targets.
        if "hyper-parameters" in params["original"]:
            tvals = []
            torder = config["hyper-parameter-list"]
            for tkey in torder:
                tvals.append(config["original"]["hyper-parameters"][tkey])
        else:
            torder = config["hyper-parameter-list"]
            tvals = [0] * len(config["hyper-parameter-list"])

        active_targets = [0] * len(tvals)
        active_torder = []
        for tkey in config["bsdf-estimator"]["update-list"]:
            active_targets[config["hyper-parameter-list"].index(tkey)] = 1
            active_torder.append(tkey)
            #print(active_targets)

        bvals = []
        bgrads = []
        ierrors = []
        nerrors = []
        lvals = []
        berrors = []

        bsefile = self.directory + "/bsdfs/errors.json"

        if not os.path.exists(bsefile):
            return

        reconerrors = json.load(open(bsefile, "r"))["sphere-errors"]
        for i in range(config["remesher"]["iterations"]):
            befile = self.directory + "/errors/bsdf-errors-" + format(i).zfill(2) + ".json"
            efile = self.directory + "/errors/errors-" + format(i).zfill(2) + ".json"
            if os.path.exists(befile) and os.path.exists(efile):
                self.berrordata = json.load(open(befile, "r"))
                self.errordata = json.load(open(efile, "r"))

                ierrors = ierrors + self.errordata["ierrors"]
                nerrors = nerrors + self.errordata["nerrors"]

                # Adjust bvals.
                lvals = lvals + self.berrordata["ierrors"]
                ierrors = ierrors + self.berrordata["ierrors"]
                nerrors = nerrors + len(self.berrordata["ierrors"]) * [nerrors[-1]]

                if i < len(reconerrors):
                    berrors = berrors + [reconerrors[i]] * len(self.berrordata["ierrors"] + self.errordata["ierrors"])

                bvals = bvals + ([self.berrordata["bvals"][0]] * len(self.errordata["ierrors"])) + self.berrordata["bvals"]
                bgrads = bgrads + ([[0]*len(bvals[0])] * len(self.errordata["ierrors"])) + self.berrordata["bmodgrads"]
                self.superindices.append(len(bvals)-1)

        #ierrors = self.errordata["ierrors"]

        if len(bvals) == 0:
            return

        bvals = np.stack(bvals, axis=1)
        bgrads = np.stack(bgrads, axis=1)
        ierrors = np.array(ierrors) / np.max(ierrors)
        nerrors = np.array(nerrors) / np.max(nerrors)
        berrors = np.array(berrors) / np.max(berrors)

        print(len(bvals), self.superindices)
        #print(bvals)
        # Disable all
        if self.weightsEnabled:
            for i, paramvals in enumerate(bvals):
                if not active_targets[i]:
                    continue
                print("C" + format(i))
                for k, grad in enumerate(bgrads[i]):
                    if grad > 0:
                        ymul = +0.1 * np.log(grad + 1)
                    else:
                        ymul = -0.1 * np.log(-(grad - 1))
                    self.subplot.arrow(k, paramvals[k], 0, -ymul * 0.01, color="C" + format(i % 10))
                self.subplot.plot(range(len(paramvals)), paramvals, color="C" + format(i % 10))

        self.subplot.plot(range(len(ierrors)), ierrors, color="C" + format((len(bvals)+1) % 10))
        self.subplot.plot(range(len(nerrors)), nerrors, color="C" + format((len(bvals)+2) % 10))
        self.subplot.plot(range(len(berrors)), berrors, color="C" + format((len(bvals)+3) % 10))

        #Eself.subplot.plot(range(len(nerrors)), nerrors)
        #self.subplot.plot(range(len(ierrors)), ierrors)

        for si in self.superindices:
            self.subplot.axvline(x=si, color="C" + format(len(bvals) % 10))

        for i, tval in enumerate(tvals):
            if not active_targets[i]:
                continue
            print("C" + format(i))
            self.subplot.axhline(y=tval, color="C" + format(i % 10))

        legends = []
        if self.weightsEnabled:
            legends += config["bsdf-estimator"]["update-list"] 
        legends += ["image-error", "normals-error", "bsdf-reconstruction-error", "super-iteration-divider"]
        self.subplot.legend(legends)
        # TODO: Set legend and stuff.

    def updatePlot(self, index, superindex):
        print("Updating plot")
        self.subplot.cla()
        self.lineindex = index
        self.superIndex = superindex
        self.plot()
        self.canvas.draw()

    def getLayout(self):
        return self.layout
    
    def getWidget(self):
        return self.widget

# uses matplotlib
class MeshPlotter:
    def __init__(self, directory, filenames, mode="normals-concentric", normalsSubsample=1, verticesSubsample=1):
        self.directory = directory
        self.canvas = FigureCanvas(Figure(figsize=(5, 3)))
        self.layout = QVBoxLayout()
        self.titleLabel = QLabel("Graph")

        self.titleLabel.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.titleLabel)
        self.layout.addWidget(self.canvas)
        self.canvas.setSizePolicy(QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding))

        self.subplot = self.canvas.figure.subplots()

        self.widget = QWidget()
        self.widget.setLayout(self.layout)
        self.filenames = filenames
        self.normalsSubsample = normalsSubsample
        self.verticesSubsample = verticesSubsample

        self.mode = mode

    def changeDirectory(self, directory):
        self.directory = directory
        self.updatePlot(self.filenames)

    def plotNormals(self, vertices, normals, color='k', multiplier=1.0):
        # Select a slice of vertices.
        shortlist = []
        for vertex, normal in zip(vertices, normals):
            if vertex[1] > -0.0005 and vertex[1] < +0.0005:
                shortlist.append((vertex, normal))

        for vertex, normal in shortlist[::self.normalsSubsample]:
            self.subplot.plot([vertex[0], vertex[0] + normal[0]* 0.04 * multiplier], [vertex[2], vertex[2] + normal[2]*0.04 * multiplier], color=color, linestyle='-', linewidth=1)

    def plot(self):
        for idx, filename in enumerate(self.filenames):
            # TODO: Likely performance issues with this statement.
            print("Mesh: ", filename)

            if not os.path.exists(filename):
                print("Couldn't find " + filename)
                continue

            self.vertices = load_normals.load_vertices(filename)
            self.normals = load_normals.load_normals(filename)
            if idx == 0:
                self.referenceVertices = self.vertices
                self.referenceNormals = self.normals
            if idx == 1:
                self.gradients = self.normals
                continue

            shortlist = []
            for vertex, normal in zip(self.vertices, self.normals):
                if vertex[1] > -0.0005 and vertex[1] < +0.0005:
                    shortlist.append((vertex, normal))

            xs = []
            ys = []

            for vertex, normal in shortlist:
                xs.append(vertex[0])
                ys.append(vertex[2])

            if self.mode == "normals":
                if idx < 2:
                    self.plotNormals(self.vertices, self.normals, color='C' + format(idx))
            elif self.mode == "normals-concentric":
                if idx == 2:
                    self.plotNormals(self.vertices, self.referenceNormals, color='C0')
                    self.plotNormals(self.vertices, self.gradients, color='C2', multiplier=50.0)
                    self.plotNormals(self.vertices, self.normals, color='C1')

            self.subplot.scatter(xs[::self.verticesSubsample], ys[::self.verticesSubsample])

    def updatePlot(self, filenames):
        print("Updating plot")
        self.subplot.cla()
        self.filenames = filenames
        self.plot()
        self.canvas.draw()

    def getLayout(self):
        return self.layout
    
    def getWidget(self):
        return self.widget

# Finish this up later..
class MeshIndexLoader:
    def __init__(self, dir):
        pass

    def preloadIteration(self, index):
        # Pre-Load a specific mesh based on the given index
        pass

    def getNormalAtIndex(self, index):
        # Normal at the given index
        pass

    def getGradientAtIndex(self, index):
        # Gradient of the particular normal
        pass

class Selector:
    def __init__(self, options):
        self.options = options
        self.radios = []
        self.layout = QHBoxLayout()
        self.buttongroup = QButtonGroup()

        for i, option in enumerate(options):
            radiobutton = QRadioButton(option)

            radiobutton.toggled.connect(partial(self._onToggled, i))
            print("Radio " + option + " registered as " + format(i))

            self.layout.addWidget(radiobutton)
            self.buttongroup.addButton(radiobutton)
            self.radios.append(radiobutton)

        self.callback = None

    def _onToggled(self, idx):
        print("IDX: " + format(idx))
        button = self.radios[idx]
        print("Toggled: " + button.text() + " " + format(idx) + " " + format(button.isChecked()))

        if button.isChecked() == True:
            if self.callback:
                self.callback(button.text())

    def setToggledCallback(self, callback):
        self.callback = callback

    def getLayout(self):
        return self.layout


def lToS(lt):
    return '%1.2f %1.2f %1.2f' % (float(lt.split(" ")[1]), float(lt.split(" ")[2]), float(lt.split(" ")[3]))

class App(QWidget):

    def __init__(self):
        super().__init__()
        self.title = 'Differential Rendering Optimizer Monitor'
        self.left = 10
        self.top = 10
        self.width = 1280
        self.height = 600

        self.errorImages = []
        self.gradientImages = []
        self.rawGradientImages = []
        self.normalImages = []
        self.currentImages = []
        self.bsdfImages = []
        self.referenceImages = []
        self.samplerImages = []

        self.imageview = None
        self.plotter = None
        self.bsdfPlotter = None
        self.loader = None
        self.controls = None

        self.superIndex = 0
        self.index = 0

        self.factor = 1.0

        self.awsConfig = None

        self.initUI()

    def _indexChanged(self, index, ignorePlots=False):
        self.index = index
        for i, image in enumerate(self.errorImages):
            image.setData(self.loader.loadImageErrorImage(i, index, self.superIndex))

        for i, image in enumerate(self.currentImages):
            image.setData(self.loader.loadCurrentImage(i, index, self.superIndex))

        for i, image in enumerate(self.gradientImages):
            image.setData(self.loader.loadGradientImage(i, index, self.superIndex))

        for i, image in enumerate(self.rawGradientImages):
            image.setData(self.loader.loadRawGradientImage(i, index, self.superIndex))

        for i, image in enumerate(self.referenceImages):
            image.setData(self.loader.loadReferenceImage(i, self.superIndex))

        for i, image in enumerate(self.samplerImages):
            image.setData(self.loader.loadSamplerImage(i, index, self.superIndex))

        self.normalImage.setData(self.loader.loadNormalImage(index, self.superIndex))
        self.normalDeltaImage.setData(self.loader.loadNormalDeltaImage(index, self.superIndex))

        self.totalGradientImage.setData(self.loader.loadTotalGradientImage(index, self.superIndex))

        for i, envmap in enumerate(ENVMAPS):
            self.bsdfImages[i].setData(self.loader.loadBSDFErrorImage(superindex=self.superIndex, iterindex=index, envmap=envmap))

        if not ignorePlots:
            self.plotter.updatePlot(index, self.superIndex)
            self.bsdfPlotter.updatePlot(index, self.superIndex)
            self.radialErrorPlotter.updatePlot(index, self.superIndex)
            self.bsdfCurvePlotter.updatePlot(index, self.superIndex)
            self.crossSectionPlotter.updatePlot(
                    [
                        self.loader.loadCurrentImage(1, index, self.superIndex),
                        #self.loader.loadFinalHighSPImage(1),
                        self.loader.loadReferenceImage(1, self.superIndex)
                    ]
                )

    def _factorChanged(self, factor):
        """self.factor = factor
        for i, image in enumerate(self.errorImages):
            image.setFactor(factor)

        for i, image in enumerate(self.currentImages):
            image.setFactor(factor)

        for i, image in enumerate(self.gradientImages):
            image.setFactor(factor)

        for i, image in enumerate(self.rawGradientImages):
            image.setFactor(factor)

        self.normalImage.setFactor(factor)
        self.normalDeltaImage.setFactor(factor)

        self.totalGradientImage.setFactor(factor)"""
        self.factor = factor
        self._indexChanged(self.index, ignorePlots=True)

    def _superIndexChanged(self, superindex):
        self.controls.setMaxIndex(self.loader.getIterationCount(superindex))
        self.superIndex = superindex

        self._indexChanged(self.index)
        self.plotter.updatePlot(self.index, superindex)
        self.bsdfPlotter.updatePlot(self.index, superindex)
        self.radialErrorPlotter.updatePlot(self.index, superindex)
        self.bsdfCurvePlotter.updatePlot(self.index, superindex)

        if self.loader.isTotalGradientsAvailable():
            self.meshPlotter.updatePlot([self.referenceMesh, self.loader.getTotalGradientMesh(0, self.superIndex), self.loader.getRemeshedMesh(self.superIndex)])
        else:
            self.meshPlotter.updatePlot([self.referenceMesh, self.loader.getRemeshedMesh(self.superIndex), self.loader.getRemeshedMesh(self.superIndex)])

        for i, envmap in enumerate(ENVMAPS):
            self.bsdfImages[i].setData(self.loader.loadBSDFErrorImage(superindex=superindex, envmap=envmap))

    def _magFilter(self, data):
        return self.radialOverlay.apply(self._polarizeData((data * 20.0 * self.factor * 256)).astype('uint8'))

    def _refFilter(self, data):
        temp = (data * 40.0 * 0.1 * self.factor * 255)
        temp = np.clip(temp, 0, 255)
        return self.radialOverlay.apply(temp.astype('uint8'))

    def _currentFilter(self, data):
        temp = (data * 40.0 * 0.1 * self.factor * 255)
        temp = np.clip(temp, 0, 255)
        return self.radialOverlay.apply(temp.astype('uint8'))

    def _normalFilter(self, data):
        return self.radialOverlay.apply(data)

    def _normalDeltaFilter(self, data, dim=0):
        singledim = (data * 36  * self.factor * 255)[:,:,dim]
        return self.radialOverlay.apply(self._polarizeData(singledim))

    def _normalDeltaXFilter(self, data):
        return self._normalDeltaFilter(data, 0)
    def _normalDeltaYFilter(self, data):
        return self._normalDeltaFilter(data, 1)
    def _normalDeltaZFilter(self, data):
        return self._normalDeltaFilter(data, 2)

    def _polarizeData(self, data):
        positives = np.clip(data, 0, 255)
        negatives = np.clip(data, -255, 0)
        return np.concatenate((positives[..., np.newaxis], np.zeros(positives.shape)[..., np.newaxis], -negatives[..., np.newaxis]), axis=2).astype('uint8')

    # Gradient filter collection

    def _gradientFilter(self, data, dim=0):
        singledim = (data * 0.5 * 20000 * self.factor * 255)[:,:,dim]
        return self.radialOverlay.apply(self.radialOverlay.apply(self._polarizeData(singledim)))

    def _gradientXFilter(self, data):
        return self._gradientFilter(data, 0)
    def _gradientYFilter(self, data):
        return self._gradientFilter(data, 1)
    def _gradientZFilter(self, data):
        return self._gradientFilter(data, 2)

    # Total gradient filter collection

    def _totalGradientFilter(self, data, dim=0): 
        singledim = (data * 0.5 * 6000 * self.factor * 255)[:,:,dim]
        return self.radialOverlay.apply(self._polarizeData(singledim))

    def _totalGradientXFilter(self, data):
        return self._totalGradientFilter(data, 0)
    def _totalGradientYFilter(self, data):
        return self._totalGradientFilter(data, 1)
    def _totalGradientZFilter(self, data):
        return self._totalGradientFilter(data, 2)

    # Raw gradient filter collection

    def _rawGradientFilter(self, data, dim=0):
        singledim = (data * 0.8 * 255)[:,:,dim]
        return self._polarizeData(singledim)

    def _rawGradientXFilter(self, data):
        return self._rawGradientFilter(data, 0)
    def _rawGradientYFilter(self, data):
        return self._rawGradientFilter(data, 1)
    def _rawGradientZFilter(self, data):
        return self._rawGradientFilter(data, 2)

    def _bsdfDiffFilter(self, data):
        return self._polarizeData(data * 200 * self.factor * 256)

    def _samplerFilter(self, data):
        temp = (data * 0.1 * self.factor * 255)
        temp = np.clip(temp, 0, 255)
        return temp.astype('uint8')

    def _gradientSelectorCallback(self, option):
        print("Gradient selector called: " + option)
        if option == "X":
            self.loader.setFilter("gradient-image", self._gradientXFilter)
        elif option == "Y":
            self.loader.setFilter("gradient-image", self._gradientYFilter)
        elif option == "Z":
            self.loader.setFilter("gradient-image", self._gradientZFilter)

        # Refresh the images.
        self._indexChanged(self.index)
    
    def _totalGradientSelectorCallback(self, option):
        print("Total Gradient selector called: " + option)
        if option == "X":
            self.loader.setFilter("total-gradient-image", self._totalGradientXFilter)
        elif option == "Y":
            self.loader.setFilter("total-gradient-image", self._totalGradientYFilter)
        elif option == "Z":
            self.loader.setFilter("total-gradient-image", self._totalGradientZFilter)

        # Refresh the images.
        self._indexChanged(self.index)

    def _rawGradientSelectorCallback(self, option):
        print("Raw Gradient selector called: " + option)
        if option == "X":
            self.loader.setFilter("raw-gradient-image", self._rawGradientXFilter)
        elif option == "Y":
            self.loader.setFilter("raw-gradient-image", self._rawGradientYFilter)
        elif option == "Z":
            self.loader.setFilter("raw-gradient-image", self._rawGradientZFilter)

        # Refresh the images.
        self._indexChanged(self.index)

    def _normalDeltaSelectorCallback(self, option):
        print("Normal delta selector called: " + option)
        if option == "X":
            self.loader.setFilter("normal-delta-image", self._normalDeltaXFilter)
        elif option == "Y":
            self.loader.setFilter("normal-delta-image", self._normalDeltaYFilter)
        elif option == "Z":
            self.loader.setFilter("normal-delta-image", self._normalDeltaZFilter)

        # Refresh the images.
        self._indexChanged(self.index)
    
    def _bsdfPlottableChangedCallback(self, option):
        if option == "First Moments":
            self.bsdfTrainingGraphPlotter.changePlottable(0)
        elif option == "Second Moments":
            self.bsdfTrainingGraphPlotter.changePlottable(1)
        elif option == "Update Factors":
            self.bsdfTrainingGraphPlotter.changePlottable(2)
        elif option == "Step Sizes":
            self.bsdfTrainingGraphPlotter.changePlottable(3)
        elif option == "SNR":
            self.bsdfTrainingGraphPlotter.changePlottable(4)
    
    def _bsdfPlottableStateChangedCallback(self, state):
        self.bsdfTrainingGraphPlotter.setState(state)

    def _factorChangedCallback(self, value):
        self.factor = np.exp2(value/10)
        self._factorChanged(self.factor)

    def _serverChangedCallback(self, server):
        # Remote mount directory to a temporary local directory.
        # Trigger directory changed.
        #if awsmanager.getServerStatus(server, self.awsConfig["keyFile"]) is not "free":
        #    (localDir, remoteDir) = awsmanager.remoteMount(server, self.awsConfig["keyFile"], remotedir="")
        #    self.directory = localDir
        #    self._directoryChanged()
        #else:
        #    print(server + " has no active dataset")
        if not server["valid"]:
            print(server["server"] + " has no active dataset")
        else:
            localDir = awsmanager.remoteMount(server["server"], self.awsConfig["keyFile"], path=server["directory"])
            self.directory = localDir
            self._directoryChanged()

    def _directoryChanged(self):
        self.loader.changeDirectory(self.directory)
        self.superControls.setMaxIndex(self.loader.getSuperIterationCount())
        self.controls.setMaxIndex(self.loader.getIterationCount())

        self.plotter.changeDirectory(self.directory)
        self.bsdfPlotter.changeDirectory(self.directory)
        self.meshPlotter.changeDirectory(self.directory)
        self.bsdfCurvePlotter.changeDirectory(self.directory)
        self.radialErrorPlotter.changeDirectory(self.directory)


    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        self.layout = QVBoxLayout()
        self.directory = DIRECTORY

        self.loader = ImageLoader(self.directory)
        self.loader.setFilter("light-error-image", self._magFilter)
        self.loader.setFilter("reference-image", self._refFilter)
        self.loader.setFilter("current-image", self._currentFilter)
        self.loader.setFilter("normal-image", self._normalFilter)
        self.loader.setFilter("normal-delta-image", self._normalDeltaXFilter)
        self.loader.setFilter("gradient-image", self._gradientXFilter)
        self.loader.setFilter("total-gradient-image", self._totalGradientXFilter)
        self.loader.setFilter("raw-gradient-image", self._rawGradientXFilter)
        self.loader.setFilter("bsdf-error-image", self._bsdfDiffFilter)
        self.loader.setFilter("sampler-image", self._samplerFilter)

        self.plotter = GraphPlotter(self.directory)
        self.bsdfPlotter = BSDFGraphPlotter(self.directory)
        self.bsdfCurvePlotter = BSDFPlotter(self.directory)
        self.referenceMesh = self.loader.getReferenceMesh()
        print("Reference mesh: ", self.referenceMesh)
        self.meshPlotter = MeshPlotter(self.directory, [self.referenceMesh, self.loader.getRemeshedMesh(0), self.loader.getRemeshedMesh(0)], normalsSubsample=4, verticesSubsample=4)
        self.errorImageView = DeltaView("Error images")
        self.currentImageView = DeltaView("Current images")
        self.referenceImageView = DeltaView("Reference images")
        self.samplerImageView = DeltaView("Sample density images")
        self.gradientImageView = DeltaView("Gradient images")
        self.bsdfDiffImageView = DeltaView("BSDF Differential images")
        self.totalGradientImageView = DeltaView("Total Gradient images")
        self.rawGradientImageView = DeltaView("Raw Gradient images")
        self.normalImageView = DeltaView("Normal maps")
        self.normalDeltaImageView = DeltaView("Normal delta maps")
        self.bsdfTrainingGraphPlotter = BSDFTrainingGraphPlotter(self.directory)
        self.crossSectionPlotter = CrossSectionDisplayLayout([np.zeros((256,256)), np.zeros((256,256))])
        self.radialErrorPlotter = RadialErrorView(np.zeros((256, 256)), self.loader)

        self.radialOverlay = self.radialErrorPlotter.overlayTool

        if AWS_REMOTE_ENABLE:
            # Find AWS settings
            awsConfig = awsmanager.loadRemoteSettings("MTSTF_SERVER_CONFIG")
            if awsConfig is None:
                print("Couldn't find AWS settings. Run without the -r flag for local datasets")
            self.awsRemoteView = ServerSelectView("Server select", "Pick a server", awsConfig["keyFile"], awsConfig["keyName"])
            self.awsConfig = awsConfig

        self.controls = ScrubbingControls("Controls", "Iteration", self.loader.getIterationCount() - 1, self)
        self.superControls = ScrubbingControls("Controls 2", "SuperIteration", self.loader.getSuperIterationCount() - 1, self)
        self.exposureSlider = ExposureControl("Exposure", "Exposure", maxval=200, minval=-200, initial=0)

        self.controls.setIndexChangedCallback(self._indexChanged)
        self.superControls.setIndexChangedCallback(self._superIndexChanged)
        self.exposureSlider.setValueChangedCallback(self._factorChangedCallback)

        if AWS_REMOTE_ENABLE:
            self.awsRemoteView.setServerChangedCallback(self._serverChangedCallback)

        # TODO: Temporary.
        self.bsdfPlotter.disableWeights()
        #self.bsdfPlotter.disable()
        self.bsdfCurvePlotter.setMode("NDF")

        for i in range(self.loader.getLightCount()):
            errorImage = DeltaImage(
                "L" + format(i).zfill(2) + ": " + lToS(self.loader.getLight(i)),
                self.loader.loadImageErrorImage(i, 0))
            refImage = DeltaImage(
                "L" + format(i).zfill(2) + ": " + lToS(self.loader.getLight(i)),
                self.loader.loadReferenceImage(i))
            gradientImage = DeltaImage(
                "L" + format(i).zfill(2) + ": " + lToS(self.loader.getLight(i)),
                self.loader.loadGradientImage(i, 0))
            rawGradientImage = DeltaImage(
                "L" + format(i).zfill(2) + ": " + lToS(self.loader.getLight(i)),
                self.loader.loadRawGradientImage(i, 0))
            currentImage = DeltaImage(
                "L" + format(i).zfill(2) + ": " + lToS(self.loader.getLight(i)),
                self.loader.loadCurrentImage(i, 0))
            samplerImage = DeltaImage(
                "L" + format(i).zfill(2) + ": " + lToS(self.loader.getLight(i)),
                self.loader.loadSamplerImage(i, 0))

            self.errorImageView.addImage(errorImage)
            self.referenceImageView.addImage(refImage)
            self.gradientImageView.addImage(gradientImage)
            self.samplerImageView.addImage(samplerImage)
            self.rawGradientImageView.addImage(rawGradientImage)
            self.currentImageView.addImage(currentImage)

            self.errorImages.append(errorImage)
            self.gradientImages.append(gradientImage)
            self.samplerImages.append(samplerImage)
            self.rawGradientImages.append(rawGradientImage)
            self.currentImages.append(currentImage)
            self.referenceImages.append(refImage)

        for envmap in ENVMAPS:
            bsdfImage = DeltaImage(
                envmap,
                self.loader.loadBSDFErrorImage(superindex=0, iterindex=0, envmap=envmap))
            self.bsdfDiffImageView.addImage(bsdfImage)
            self.bsdfImages.append(bsdfImage)

        normalImage = DeltaImage(
                "",
                self.loader.loadNormalImage(0))

        normalDeltaImage = DeltaImage(
                "",
                self.loader.loadNormalDeltaImage(0))

        totalGradientImage = DeltaImage(
                "",
                self.loader.loadTotalGradientImage(0))

        self.normalImageView.addImage(normalImage)
        self.normalDeltaImageView.addImage(normalDeltaImage)
        self.totalGradientImageView.addImage(totalGradientImage)

        self.normalImage = normalImage
        self.normalDeltaImage = normalDeltaImage
        self.totalGradientImage = totalGradientImage

        self.tabWidget = QTabWidget()

        #self.layout.addLayout(self.referenceImageView.getLayout())
        #self.layout.addLayout(self.errorImageView.getLayout())
        #self.layout.addLayout(self.gradientImageView.getLayout())
        #self.layout.addLayout(self.normalImageView.getLayout())
        #self.layout.addLayout(self.plotter.getLayout())

        self.gradientLayout = QVBoxLayout()
        self.gradientLayoutWidget = QWidget()
        self.gradientLayoutWidget.setLayout(self.gradientLayout)

        self.gradientDimSelector = Selector(["X", "Y", "Z"])
        self.gradientDimSelector.setToggledCallback(self._gradientSelectorCallback)

        self.gradientLayout.addWidget(self.gradientImageView.getWidget())
        self.gradientLayout.addLayout(self.gradientDimSelector.getLayout())

        # Total gradient layout..
        self.totalGradientLayout = QVBoxLayout()
        self.totalGradientLayoutWidget = QWidget()
        self.totalGradientLayoutWidget.setLayout(self.totalGradientLayout)

        self.totalGradientDimSelector = Selector(["X", "Y", "Z"])
        self.totalGradientDimSelector.setToggledCallback(self._totalGradientSelectorCallback)

        self.totalGradientLayout.addWidget(self.totalGradientImageView.getWidget())
        self.totalGradientLayout.addLayout(self.totalGradientDimSelector.getLayout())

        # Raw gradient layout.. (deprecated)
        self.rawGradientLayout = QVBoxLayout()
        self.rawGradientLayoutWidget = QWidget()
        self.rawGradientLayoutWidget.setLayout(self.rawGradientLayout)

        self.rawGradientDimSelector = Selector(["X", "Y", "Z"])
        self.rawGradientDimSelector.setToggledCallback(self._rawGradientSelectorCallback)

        self.rawGradientLayout.addWidget(self.rawGradientImageView.getWidget())
        self.rawGradientLayout.addLayout(self.rawGradientDimSelector.getLayout())

        # BSDF plotter layout
        self.bsdfPlottableLayout = QVBoxLayout()
        self.bsdfPlottableLayoutWidget = QWidget()
        self.bsdfPlottableLayoutWidget.setLayout(self.bsdfPlottableLayout)

        labelsPath = self.directory + "/inputs/bsdf-dictionary.json"
        dictionary = json.load(open(labelsPath, "r"))
        labelsFiltered = [ elementToString(dictionary["elements"][i]) for i in range(len(dictionary["elements"])) ]

        self.bsdfPlottableDimSelector = Selector(["First Moments", "Second Moments", "Update Factors", "Step Sizes", "SNR"])
        self.bsdfPlottableStateSelector = MultiSelector(labelsFiltered, [True] * len(labelsFiltered))
        self.bsdfPlottableDimSelector.setToggledCallback(self._bsdfPlottableChangedCallback)
        self.bsdfPlottableStateSelector.setToggledCallback(self._bsdfPlottableStateChangedCallback)

        self.bsdfPlottableLayout.addWidget(self.bsdfTrainingGraphPlotter.getWidget())
        self.bsdfPlottableLayout.addLayout(self.bsdfPlottableDimSelector.getLayout())
        self.bsdfPlottableLayout.addLayout(self.bsdfPlottableStateSelector.getLayout())

        # Normal delta layout..
        self.normalDeltaLayout = QVBoxLayout()
        self.normalDeltaLayoutWidget = QWidget()
        self.normalDeltaLayoutWidget.setLayout(self.normalDeltaLayout)

        self.normalDeltaDimSelector = Selector(["X", "Y", "Z"])
        self.normalDeltaDimSelector.setToggledCallback(self._normalDeltaSelectorCallback)

        self.normalDeltaLayout.addWidget(self.normalDeltaImageView.getWidget())
        self.normalDeltaLayout.addLayout(self.normalDeltaDimSelector.getLayout())

        if AWS_REMOTE_ENABLE:
            # Add view to select remote server.
            self.tabWidget.addTab(self.awsRemoteView.getWidget(), "Server Select")

        self.tabWidget.addTab(self.referenceImageView.getWidget(), "Reference Images")
        self.tabWidget.addTab(self.errorImageView.getWidget(), "Error Images")
        self.tabWidget.addTab(self.currentImageView.getWidget(), "Current Images")
        self.tabWidget.addTab(self.samplerImageView.getWidget(), "Sampler Images")
        self.tabWidget.addTab(self.gradientLayoutWidget, "Gradient Images")
        self.tabWidget.addTab(self.totalGradientLayoutWidget, "Total Gradient Image")
        self.tabWidget.addTab(self.rawGradientLayoutWidget, "Raw Gradient Images")
        self.tabWidget.addTab(self.normalImageView.getWidget(), "Normal Images")
        self.tabWidget.addTab(self.normalDeltaLayoutWidget, "Normal Delta Images")
        self.tabWidget.addTab(self.bsdfDiffImageView.getWidget(), "BSDF Diff Images")
        self.tabWidget.addTab(self.plotter.getWidget(), "Error Plots")
        self.tabWidget.addTab(self.bsdfPlotter.getWidget(), "BSDF Plots")
        self.tabWidget.addTab(self.bsdfPlottableLayoutWidget, "BSDF Optimization")
        self.tabWidget.addTab(self.bsdfCurvePlotter.getWidget(), "BSDF")
        self.tabWidget.addTab(self.meshPlotter.getWidget(), "Mesh Plots")
        self.tabWidget.addTab(self.crossSectionPlotter.getWidget(), "Cross-Section")
        self.tabWidget.addTab(self.radialErrorPlotter.getWidget(), "Radial Error")

        # Add parameter text display.
        self.parameterDisplay = QTextEdit()
        self.tabWidget.addTab(self.parameterDisplay, "Parameters")

        self.parameterText = self.loader.getFullConfigText()
        self.parameterDisplay.setReadOnly(True)
        self.parameterDisplay.insertPlainText(self.parameterText)

        self.layout.addWidget(self.tabWidget)
        self.layout.addLayout(self.controls.getLayout())
        self.layout.addLayout(self.superControls.getLayout())
        self.layout.addLayout(self.exposureSlider.getLayout())

        self.plotter.plot()
        self.meshPlotter.plot()
        self.bsdfTrainingGraphPlotter.plot()
        self.radialErrorPlotter.plot()

        self.setLayout(self.layout)
        self.show()

if __name__ == '__main__':
    parse = optparse.OptionParser()
    parse.add_option("-r", "--aws-remote", dest="awsRemote", action="store_true")
    (options, args) = parse.parse_args()
    AWS_REMOTE_ENABLE = options.awsRemote
    #print(args)
    DIRECTORY = args[0]

    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())