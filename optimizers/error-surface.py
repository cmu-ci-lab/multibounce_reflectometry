# Plots error surface.

from dictionary_embedded import embedDictionary
from parameters import prepareParameters
import json
import numpy as np
import sys
import os
from shutil import copyfile
import hdsutils
import optparse

import matplotlib.pyplot as plt

def buildPString(hparams, params):
    pstring = ""
    for hparameter in params["hyper-parameter-list"]:
        pstring += " -D" + hparameter + "=" + format(hparams[hparameter])
    return pstring

pltShow = False

def getError(hparams, params, sampleCount=128, directory="."):
    pstring = buildPString(hparams, params)
    currents = []
    for i, l in enumerate(params["lights"]["data"]):
        os.system("mitsuba " + directory + "/" + params["scenes"]["intensity"] + " -o " + "./error-surface/" + format(i).zfill(2) + ".hds" + " -Dirradiance=" + format(params["lights"]["intensity-data"][i]) + " -Dwidth=" + format(params["target"]["width"]) + " -Dheight=" + format(params["target"]["height"]) + " -DlightX=" + format(l[0]) + " -DlightY=" + format(l[1]) + " -DlightZ=" + format(l[2]) + " -Ddepth=" + format(params["estimator"]["depth"]) + " -DsampleCount=" + format(sampleCount) + pstring + " > /dev/null")
        currents.append(hdsutils.loadHDSImage("./error-surface/" + format(i).zfill(2) + ".hds"))

    plt.close()

    nImgs = len(params["lights"]["intensity-data"])
    fig = plt.figure(figsize=(nImgs, 2))

    for i in range(nImgs):
        fig.add_subplot(nImgs, 2, (i*2)+1)
        print(params["target"]["data"].shape)
        plt.imshow(params["target"]["data"][:,:,i])
        fig.add_subplot(nImgs, 2, (i*2)+2)
        plt.imshow(currents[i][:,:,0])

    plt.show(block=False)
    plt.pause(2)

    return np.sum(np.square(params["target"]["data"] - np.stack(currents, axis=0)))

def getParameterMap(params, directory):
    runnerParameterMap = {}

    if type(params["hyper-parameter-list"]) is list:
        runnerParameterList = params["hyper-parameter-list"]
    else:
        runnerParameterList = json.load(open(params["hyper-parameter-list"], "r"))
        params["hyper-parameter-list"] = runnerParameterList

    if "hyper-parameters" in params["original"] and type(params["original"]["hyper-parameters"]) is unicode:
        runnerParameterValues = np.load(directory + "/" + params["original"]["hyper-parameters"])
        params["original"]["hyper-parameters"] = dict(zip(runnerParameterList, runnerParameterValues))

    if "hyper-parameters" in params["estimator"] and type(params["estimator"]["hyper-parameters"]) is unicode:
        runnerParameterValues = np.load(directory + "/" + params["estimator"]["hyper-parameters"])
        params["estimator"]["hyper-parameters"] = dict(zip(runnerParameterList, runnerParameterValues))

    if "hyper-parameters" in params["bsdf-estimator"] and type(params["bsdf-estimator"]["hyper-parameters"]) is unicode:
        runnerParameterValues = np.load(directory + "/" + params["bsdf-estimator"]["hyper-parameters"])
        params["bsdf-estimator"]["hyper-parameters"] = dict(zip(runnerParameterList, runnerParameterValues))

    # If map specified
    runnerParameterMap = None
    if "hyper-parameters" in params["original"] and type(params["original"]["hyper-parameters"]) is dict:
        runnerParameterMap = params["original"]["hyper-parameters"]
    elif "hyper-parameters" in params["estimator"] and type(params["estimator"]["hyper-parameters"]) is dict:
        runnerParameterMap = params["estimator"]["hyper-parameters"]

    # If file specified.
    runnerParameterValues = None
    if "hyper-parameters" in params["original"] and type(params["original"]["hyper-parameters"]) is unicode:
        runnerParameterValues = np.load(directory + "/" + params["original"]["hyper-parameters"])
    elif "hyper-parameters" in params["estimator"] and type(params["estimator"]["hyper-parameters"]) is unicode:
        runnerParameterValues = np.load(directory + "/" + params["estimator"]["hyper-parameters"])

    return runnerParameterMap, runnerParameterList, runnerParameterValues



class ErrorSurfaceGenerator:
    def __init__(self):
        self.data = None

parser = optparse.OptionParser()
parser.add_option("-p", "--params", dest="params", default=None, type="string")
parser.add_option("-l", "--linear", dest="linear", action="store_true", default=False)

(options, args) = parser.parse_args()

directory = args[0]

config = json.load(open(directory + "/config.json", "r"))

if not os.path.exists("error-surface"):
    print("Creating error-surface")
    os.mkdir("error-surface")

# Load lights and source images.
params = prepareParameters(config, directory=directory)

if "bsdf-preprocess" in params and params["bsdf-preprocess"]["enabled"]:
    ixml = params["scenes"]["intensity"]
    ixmlout = ixml + ".pp.xml"

    gxml = params["scenes"]["gradient"]
    gxmlout = gxml + ".pp.xml"

    print(type(params["hyper-parameter-list"]))

    if not type(params["hyper-parameter-list"]) is unicode:
        print("hyper-parameter-list has been specified while bsdf preprocessing is active. Aborting")
        sys.exit(1)

    paramsout = params["hyper-parameter-list"]
    paddingout = params["zero-padding"]
    dfile = directory + "/" + params["bsdf-preprocess"]["file"]
    embedDictionary(dfile, directory + "/" + ixml, directory + "/" + ixmlout, paramsout, paddingout)
    embedDictionary(dfile, directory + "/" + gxml, directory + "/" + gxmlout, paramsout, paddingout)

    # Replace the original values with the modified files.
    params["scenes"]["intensity"] = ixmlout
    params["scenes"]["gradient"] = gxmlout

originalfile = sys.argv[1] + "/" + params["initialization"]["file"]
copyfile(originalfile, "/tmp/mts_mesh_intensity_slot_0.ply")

pstring = ""

resolution = (25, 25)

surface = np.zeros(resolution)

wts1 = np.linspace(0.0, 1.0, resolution[0])
wts2 = np.linspace(0.0, 1.0, resolution[1])

runnerParameterMap, runnerParameterList, runnerParameterValues = getParameterMap(params, sys.argv[1])

if options.params:
    print "Detected non-zero labels ", [label for label in runnerParameterMap if runnerParameterMap[label] != 0.0]

    labels = sys.argv[2].split(";")

    if len(labels) != 3:
        print("Error: Number of labels must be 3")
        sys.exit(1)

    progress = 0
    maxval = runnerParameterMap[labels[0]] + runnerParameterMap[labels[1]] + runnerParameterMap[labels[2]]
elif options.linear:



"""for x, wt1 in enumerate(wts1):
    for y, wt2 in enumerate(wts2):
        runnerParameterMapM = ParamGenerator.generateParamsPoint(runnerParameterMap)

        #if (wt1 + wt2) < 1:
        #    L = getError(runnerParameterMap, params, sampleCount=2, directory=sys.argv[1])
        #    print(x,y,wt1,wt2,L)
        #    surface[x,y] = L

        progress += 1
        print("\r" + ("%.1f" % ((float(progress * 100)/(resolution[0]*resolution[1]))) + "\r"))
"""

data = None
for paramMap in generator.getIterator():
    L = getError(paramMap, params, sampleCount=2, directory=sys.argv[1])

    progress += 1
    print("\r" + ("%.1f" % ((float(progress * 100)/(resolution[0]*resolution[1]))) + "\r"))

np.save("error-surface/surface.npy", surface)