# Compares finite-differences with monte-carlo gradients

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
parser.add_option("--layout-columns", dest="layoutColumns", default=5, type="int")
parser.add_option("--layout-rows", dest="layoutRows", default=5, type="int")
parser.add_option("-i","--iteration", dest="iteration", default=0, type="int")
parser.add_option("-s","--super-iteration", dest="superiteration", default=0, type="int")
parser.add_option("-l","--lindex", dest="lindex", default=0, type="int")


(options, args) = parser.parse_args()

directory = args[0]

dataset = Dataset(directory)
testset = dataset.testSet()

gdirpath = directory + "/images/raw-bsdf-gradients"

lr = options.layoutRows
lc = options.layoutColumns


gradientSlices = np.load(gdirpath + "/" + format(options.superiteration).zfill(2) + "/" + format(options.iteration).zfill(4) + "-" + format(options.lindex).zfill(2) + ".npy")
fig = plt.figure(figsize=(lr, lc))
for k in range(gradientSlices.shape[0]):
    if k >= (lr * lc):
        print "Not enough spots to display all slices. Slices remaining: ", (gradientSlices.shape[0] - k) + 1
        break
    ax = fig.add_subplot(lr, lc, k+1)
    ax.set_title(testset.bsdfDictionaryElementStrings[k])
    plt.imshow(gradientSlices[k,:,:])
plt.show()
