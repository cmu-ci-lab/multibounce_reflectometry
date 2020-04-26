# Use mtsutil to write slice data
import json
import os
import np2exr
import hdsutils
import numpy as np
import sys
import exr2tif
import matplotlib.pyplot as plt
from shutil import copyfile
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

directory = sys.argv[1]

execfile(os.path.dirname(__file__) + "/../optimizers/dictionary_embedded.py")

def buildParameterString(weights, plist, padding):
    pstring = ""
    for weight, pname in zip(weights, plist):
        pstring += "-D" + pname + "=" + format(weight) + " "

    return pstring

def exportData(scenepath, targetpath, paramstring, title="Exported BSDF"):
    #paramstring = ""

    #for i, param in enumerate(params):
    #    paramstring += "-Dweight" + format(i).zfill(4) + "=" + format(param) + " "

    command = "mtsutil bsdfexport" +\
              " -o \"" + targetpath + "\"" +\
              " -Denvmap=doge.exr" + \
              " -Ddepth=-1" + \
              " " + paramstring + \
              " -DsampleCount=256" + \
              " -t 1024 -p 2 -k 8" + \
              " -l \"" + title + "\"" +\
              " \"" + scenepath + "\" > /dev/null"

    print("[Export Command] ", command)
    os.system(command)

    return command

def exportNDF(scenepath, targetpath, paramstring, title="Exported BSDF"):
    #paramstring = ""

    #for i, param in enumerate(params):
    #    paramstring += "-Dweight" + format(i).zfill(4) + "=" + format(param) + " "

    command = "mtsutil bsdfexport" +\
              " -o \"" + targetpath + "\"" +\
              " -Denvmap=doge.exr" + \
              " -Ddepth=-1" + \
              " " + paramstring + \
              " -DsampleCount=256" + \
              " -t 4196 -n" + \
              " -l \"" + title + "\"" +\
              " \"" + scenepath + "\" > /dev/null"

    print("[Export Command] ", command)
    os.system(command)

    return command

#scenepath = "/mnt/ufssd/thesis/current/cardboard-4/inputs/scenes/comparison/source-scene.xml.pp.xml"
config = json.load(open(directory + "/inputs/config.json", "r"))

# Load the dictionary
dfile = directory + "/inputs/" + config["bsdf-preprocess"]["file"]
tabledfile = directory + "/inputs/bsdf-table-dictionary.json"
dictionary = json.load(open(dfile, "r"))

if os.path.exists(tabledfile):
    tabledictionary = json.load(open(tabledfile, "r"))
else:
    tabledictionary = dictionary


#if superiteration == -1:
latestWeights = None
initialWeights = None
for si in range(config["remesher"]["iterations"]):
    sitext = format(si).zfill(2)
    sfile = directory + "/errors/bsdf-errors-" + sitext + ".json"
    if os.path.exists(sfile):
        latestWeights = json.load(open(sfile, "r"))["bvals"][-1]            
    if os.path.exists(sfile) and si == 0:
        initialWeights = json.load(open(sfile, "r"))["bvals"][0]

if os.path.exists(directory + "/bsdfs/tablefit.npy"):
    tableWeights = np.load(directory + "/bsdfs/tablefit.npy")
else:
    tableWeights = None

"""else:
    sfile = directory + "/errors/bsdf-errors-00.json"
    initialWeights = json.load(open(sfile, "r"))["bvals"][0]
    sfile = directory + "/errors/bsdf-errors-" + format(superiteration).zfill(2) + ".json"
    latestWeights = json.load(open(sfile, "r"))["bvals"][iteration]"""

# Find the scene files.
#targetScene = directory + "/inputs/scenes/comparison/target-scene.xml"
#sourceScene = directory + "/inputs/scenes/comparison/source-scene.xml"
targetScene = directory + "/inputs/" + config["bsdf-compare"]["target"]
sourceScene = directory + "/inputs/" + config["bsdf-compare"]["source"]

sourceSceneP = sourceScene + ".pp.xml"
sourceSceneTP = sourceScene + ".tablepp.xml"
paramsout = "/tmp/paramsout.json"
paddingout = "/tmp/paddingout.json"

targetSceneP = targetScene + ".pp.xml"

tparamsout = "/tmp/tparamsout.json"
tpaddingout = "/tmp/tpaddingout.json"

targetEmbeddingEnabled = "target-embed" in config["bsdf-compare"] and config["bsdf-compare"]["target-embed"]

if targetEmbeddingEnabled:
    targetWeights = np.load(directory + "/inputs/" + config["bsdf-compare"]["target-weights"])

embedDictionary(dfile, sourceScene, sourceSceneP, paramsout, paddingout)
if os.path.exists(tabledfile):
    embedDictionary(tabledfile, sourceScene, sourceSceneTP, tparamsout, tpaddingout)
else:
    embedDictionary(dfile, sourceScene, sourceSceneTP, tparamsout, tpaddingout)

if targetEmbeddingEnabled:
    embedDictionary(dfile, targetScene, targetSceneP, paramsout, paddingout)

parameterList = json.load(open(paramsout, "r"))
zeroPadding = json.load(open(paddingout, "r"))[0]

parameterListT = json.load(open(tparamsout, "r"))
zeroPaddingT = json.load(open(tpaddingout, "r"))[0]

paramString = buildParameterString(latestWeights, parameterList, zeroPadding)
initialParamString = buildParameterString(initialWeights, parameterList, zeroPadding)

if targetEmbeddingEnabled:
    targetParamString = buildParameterString(targetWeights, parameterList, zeroPadding)

# Import table parameters from MATLAB data file and export them.
if os.path.exists(directory + "/bsdfs/tablefit.npy") and tableWeights is not None:
    tableParamString = buildParameterString(tableWeights, parameterListT, zeroPaddingT)
else:
    tableParamString = None

sortedweights = {}
isortedweights = {}

print len(dictionary['elements'])
for weight, entry in zip(latestWeights, dictionary['elements']):
    print entry
    if not entry['type'] == 'roughconductor':
        continue

    eta = entry['eta']
    alpha = entry['alpha']
    if eta not in sortedweights:
        sortedweights[eta] = {}
    sortedweights[eta][alpha] = weight

for weight, entry in zip(initialWeights, dictionary['elements']):
    print entry
    if not entry['type'] == 'roughconductor':
        continue

    eta = entry['eta']
    alpha = entry['alpha']
    if eta not in isortedweights:
        isortedweights[eta] = {}
    isortedweights[eta][alpha] = weight


"""for eta in sortedweights:
    subdict = sortedweights[eta]
    subdict = sorted(zip(subdict.keys(), subdict.values()), key=lambda x:x[0])

    isubdict = isortedweights[eta]
    isubdict = sorted(zip(isubdict.keys(), isubdict.values()), key=lambda x:x[0])

    sortedweights[eta] = subdict
    isortedweights[eta] = isubdict
    bar_width = 0.35
    plt.bar(range(len(subdict)), [x[1] for x in subdict], bar_width, tick_label=[x[0] for x in subdict])
    plt.bar(np.array(range(len(isubdict))) + bar_width, [x[1] for x in isubdict], bar_width, tick_label=[x[0] for x in isubdict])
    plt.yscale("log")
    plt.legend([format(eta)])
    plt.show()
    plt.savefig("wtcompare-" + ("%.4f" % eta) + ".png")

"""

#sys.exit(0)

if not os.path.exists(directory + "/bsdfs"):
    os.mkdir(directory + "/bsdfs")

if not os.path.exists(directory + "/bsdfs/exports"):
    os.mkdir(directory + "/bsdfs/exports")

if not targetEmbeddingEnabled:
    exportData(targetScene,  directory + "/bsdfs/exports/bexport-target", paramString, title="Target")
else:
    exportData(targetSceneP,  directory + "/bsdfs/exports/bexport-target", targetParamString, title="Target")

exportData(sourceSceneP, directory + "/bsdfs/exports/bexport-final", paramString, title="Final Fit")
exportData(sourceSceneP, directory + "/bsdfs/exports/bexport-initial", initialParamString, title="Initial Fit")

if not targetEmbeddingEnabled:
    exportNDF(targetScene,  directory + "/bsdfs/exports/bexport-target", paramString, title="Target")
else:
    exportNDF(targetSceneP,  directory + "/bsdfs/exports/bexport-target", targetParamString, title="Target")

exportNDF(sourceSceneP, directory + "/bsdfs/exports/bexport-final", paramString, title="Final Fit")
exportNDF(sourceSceneP, directory + "/bsdfs/exports/bexport-initial", initialParamString, title="Initial Fit")

if tableParamString is not None:
    exportData(sourceSceneTP, directory + "/bsdfs/exports/bexport-table", tableParamString, title="Table Fit")

if tableParamString is not None:
    exportNDF(sourceSceneTP, directory + "/bsdfs/exports/bexport-table", tableParamString, title="Table Fit")

"""
for si in range(config["remesher"]["iterations"]):
    print("Exporting " + format(si).zfill(2))
    sitext = format(si).zfill(2)
    sfile = directory + "/errors/bsdf-errors-" + sitext + ".json"
    if os.path.exists(sfile):
        weights = json.load(open(sfile, "r"))["bvals"][-1]
    else: 
        break

    currentParamString = buildParameterString(weights, parameterList, zeroPadding)
    exportData(sourceSceneP, directory + "/bsdfs/exports/bexport-" + format(si).zfill(2), currentParamString, title="Iteration " + format(si).zfill(2))
    #currentParamString = buildParameterString(initialWeights, parameterList, zeroPadding)
"""

for si in range(config["remesher"]["iterations"]):
    sitext = format(si).zfill(2)
    sfile = directory + "/errors/bsdf-errors-" + sitext + ".json"
    if os.path.exists(sfile):
        weightsList = json.load(open(sfile, "r"))["bvals"]
    else: 
        break

    for i, weights in enumerate(weightsList):
        itext = format(i).zfill(4)
        print("Exporting " + sitext + "." + itext)

        currentParamString = buildParameterString(weights, parameterList, zeroPadding)
        exportData(sourceSceneP, directory + "/bsdfs/exports/bexport-" + sitext + "-" + itext, currentParamString, title="Iteration " + sitext + "." + itext)
        exportNDF(sourceSceneP, directory + "/bsdfs/exports/bexport-" + sitext + "-" + itext, currentParamString, title="Iteration " + sitext + "." + itext)