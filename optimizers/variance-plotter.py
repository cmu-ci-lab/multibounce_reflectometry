# Computes variance in gradient

from dictionary_embedded import embedDictionary
import json
import numpy as np
import sys
import os
from shutil import copyfile
import hdsutils
import optparse

import matplotlib.pyplot as plt

# from dataset_reader import Dataset (Alternative temporary fix)
from dataset_reader import Dataset

parser = optparse.OptionParser()
(options, args) = parser.parse_args()

directory = args[0]

dataset = Dataset(directory)

varianceData = np.load(directory + "/variance-data.npy")

legend = []
numElements = len(dataset.testSet().bsdfDictionary["elements"])
elementNames = [ format(element["alpha"]) + "," + format(element["eta"]) for element in dataset.testSet().bsdfDictionary["elements"] if element["type"] == "roughconductor"] + ["diffuse"] * len([e for e in dataset.testSet().bsdfDictionary["elements"] if e["type"] == "diffuse"])
sidxs = np.argsort([ element["alpha"] for element in dataset.testSet().bsdfDictionary["elements"] if element["type"] == "roughconductor"] + [100] * len([e for e in dataset.testSet().bsdfDictionary["elements"] if e["type"] == "diffuse"]))
for i, profile in enumerate(varianceData):
    print(len(profile["gradvars"]))
    print(sidxs)
    plt.bar(range(i, profile["gradvars"][:numElements].shape[0]*4 + i, 4), profile["gradvars"][sidxs])
    #plt.bar(range(i, numElements*4 + i, 4), profile["gradients"][1][sidxs])
    """xymax = np.max(np.stack(profile["gradients"], axis=1)[sidxs,:], axis=1)
    xymin = np.min(np.stack(profile["gradients"], axis=1)[sidxs,:], axis=1)
    xyerr = (np.array(xymax) - np.array(xymin)) / 2
    plt.errorbar(
        range(i, numElements*4 + i, 4),
        -np.mean(np.stack(profile["gradients"], axis=1)[sidxs,:], axis=1),
        xyerr,
        xerr=[0.0] * numElements,
        fmt="o"
        )
    plt.yscale('log')
    """
    plt.yscale('log')
    legend.append(profile["name"])
    plt.legend(legend)

#plt.yscale('log')
plt.xticks(range(2, numElements*4 + 2, 4), [elementNames[i] for i in sidxs])
plt.show()