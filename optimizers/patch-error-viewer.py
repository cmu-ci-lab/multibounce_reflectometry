# Compares finite-differences with monte-carlo gradients

from mpl_toolkits.mplot3d import Axes3D

from dictionary_embedded import embedDictionary
import json
import numpy as np
import sys
import os
from shutil import copyfile
import hdsutils
import optparse

import matplotlib.pyplot as plt

from dataset_reader import Dataset, Testset, toMap, mergeMaps

import load_normals

import splitpolarity
import rendernormals

parser = optparse.OptionParser()


(options, args) = parser.parse_args()

directory = args[0]

testset = Testset(directory)

pathPlot = directory + "/errors-patch-plot.npy"

errors = np.load(pathPlot)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surface = np.array([ [errors[x,y] for x in range(errors.shape[0])] for y in range(errors.shape[1]) ])
x, y = np.meshgrid(range(errors.shape[0]), range(errors.shape[1]))
ax.plot_surface(x, y, surface)

#plt.imshow(errors)
plt.show()