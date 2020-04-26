# Comparison between a coefficient-based BRDF and the original BRDF
import json
import os
import np2exr
import hdsutils
import numpy as np
import sys
import exr2tif
import matplotlib.pyplot as plt
from shutil import copyfile
import merl_io

execfile(os.path.dirname(__file__) + "/../optimizers/dictionary_embedded.py")

def buildParameterString(weights, plist, padding):
    pstring = ""
    for weight, pname in zip(weights, plist):
        pstring += "-D" + pname + "=" + format(weight) + " "

    return pstring

ENVMAPS = ["doge", "field", "uffizi"]

def renderBSDFComparison(directory, superiteration=-1, iteration=-1, outprefix="bsdf-compare-", envmaps=ENVMAPS, skiptarget=False, skipinitial=False):
    config = json.load(open(directory + "/inputs/config.json", "r"))

    tabularEnabled = "tabular-bsdf" in config["targets"]

    # Load the dictionary
    dfile = directory + "/inputs/" + config["bsdf-preprocess"]["file"]
    dictionary = json.load(open(dfile, "r"))

    if superiteration == -1:
        latestWeights = None
        initialWeights = None
        latestTable = None
        initialTable = None
        for si in range(config["remesher"]["iterations"]):
            sitext = format(si).zfill(2)
            sfile = directory + "/errors/bsdf-errors-" + sitext + ".json"
            if os.path.exists(sfile):
                latestWeights = json.load(open(sfile, "r"))["bvals"][-1]
                latestTable = json.load(open(sfile, "r"))["tbvals"][-1]
            if os.path.exists(sfile) and si == 0:
                initialWeights = json.load(open(sfile, "r"))["bvals"][0]
                initialTable = json.load(open(sfile, "r"))["tbvals"][0]
    else:
        sfile = directory + "/errors/bsdf-errors-00.json"
        initialWeights = json.load(open(sfile, "r"))["bvals"][0]
        initialTable = json.load(open(sfile, "r"))["tbvals"][0]
        sfile = directory + "/errors/bsdf-errors-" + format(superiteration).zfill(2) + ".json"
        latestWeights = json.load(open(sfile, "r"))["bvals"][iteration]
        latestTable = json.load(open(sfile, "r"))["tbvals"][iteration]

    # Find the scene files.
    targetScene = directory + "/inputs/" + config["bsdf-compare"]["target"]
    sourceScene = directory + "/inputs/" + config["bsdf-compare"]["source"]

    sourceSceneP = sourceScene + ".pp.xml"
    targetSceneP = targetScene + ".pp.xml"

    paramsout = "/tmp/paramsout.json"
    paddingout = "/tmp/paddingout.json"

    embedDictionary(dfile, sourceScene, sourceSceneP, paramsout, paddingout)
    if "target-embed" in config["bsdf-compare"] and config["bsdf-compare"]["target-embed"]:
        if "target-weights" not in config["bsdf-compare"]:
            print("Target weights not specified when target-embed is enabled")
            sys.exit(1)
        embedDictionary(dfile, sourceScene, targetSceneP, paramsout, paddingout)
    else:
        targetSceneP = targetScene

    if "target-weights" in config["bsdf-compare"]:
        targetWeights = np.load(directory + "/inputs/" + config["bsdf-compare"]["target-weights"])

    parameterList = json.load(open(paramsout, "r"))
    zeroPadding = json.load(open(paddingout, "r"))[0]

    paramString = buildParameterString(latestWeights, parameterList, zeroPadding)
    initialParamString = buildParameterString(initialWeights, parameterList, zeroPadding)

    if "target-weights" in config["bsdf-compare"]:
        targetParamString = buildParameterString(targetWeights, parameterList, zeroPadding)
    
    if tabularEnabled:
        targetTable = np.load(directory + "/inputs/" + config["targets"]["tabular-bsdf"])

    totalSquaredError = []
    for i, envmap in enumerate(envmaps):
        sourceOutfile = "/tmp/bsdf-compare-source-" + envmap + ".hds"
        targetOutfile = "/tmp/bsdf-compare-target-" + envmap + ".hds"
        initialOutfile = "/tmp/bsdf-compare-initial-" + envmap + ".hds"
        localParams = " -Denvmap=" + envmap + ".exr" + " -DmeshSlot=0 -Ddepth=-1 -DsampleCount=256 -p 8 > /dev/null" 

        print("mitsuba \"" + sourceSceneP + "\" -o \"" + sourceOutfile + "\" " + paramString + " " + localParams)
        print("mitsuba \"" + targetSceneP + "\" -o \"" + targetOutfile + "\" " + targetParamString + " " + localParams)
        
        if tabularEnabled:
            merl_io.merl_write("/tmp/tabular-bsdf-0.binary", latestTable)
        os.system("mitsuba \"" + sourceSceneP + "\" -o \"" + sourceOutfile + "\" " + paramString + " " + localParams)
        if not skipinitial:
            if tabularEnabled:
                merl_io.merl_write("/tmp/tabular-bsdf-0.binary", initialTable)
            os.system("mitsuba \"" + sourceSceneP + "\" -o \"" + initialOutfile + "\" " + initialParamString + " " + localParams)
        if not skiptarget:
            if tabularEnabled:
                merl_io.merl_write("/tmp/tabular-bsdf-0.binary", targetTabularBSDF)
            os.system("mitsuba \"" + targetSceneP + "\" -o \"" + targetOutfile + "\" " + targetParamString + " " + localParams)

        sourceImg = hdsutils.loadHDSImage(sourceOutfile)
        targetImg = hdsutils.loadHDSImage(targetOutfile)
        initialImg = hdsutils.loadHDSImage(initialOutfile)

        W,H,_ = sourceImg.shape
        squaredError = np.sum(np.square(sourceImg - targetImg))
        totalSquaredError.append(squaredError)

        differenceImg = (sourceImg - targetImg).reshape((W,H))
        positives = (differenceImg > 0) * differenceImg * 10
        negatives = (differenceImg <= 0) * -differenceImg * 10
        zeros = np.zeros_like(differenceImg)

        # Copy HDS images to folder.
        print "copying: ", sourceOutfile, outprefix + "-" + envmap + "-target.hds"
        copyfile(sourceOutfile, outprefix + "-" + envmap + "-target.hds")
        np.save(outprefix + "-" + envmap + "-diff.npy", differenceImg)

        finalImg = np.squeeze(np.concatenate([sourceImg, targetImg, negatives.reshape((W,H,1)), positives.reshape((W,H,1)), initialImg], axis=1))
        finalImg = finalImg / np.max(finalImg)

        exrname = outprefix + "-" + envmap + ".exr"
        tiffname = outprefix + "-" + envmap + ".tiff"

        np2exr.developSimple(np.squeeze(finalImg), exrname)
        exr2tif.convert(exrname, tiffname, channel='R', multiplier=10.0)

    return np.sum(totalSquaredError)

if __name__ == "__main__":
    directory = sys.argv[1]
    directive = sys.argv[2]

    if directive == "all":
        if not os.path.exists(directory + "/bsdfs"):
            os.mkdir(directory + "/bsdfs")

        serrors = []

        configfile = directory + "/inputs/config.json"
        config = json.load(open(configfile, "r"))
        simax = config["remesher"]["iterations"]
        errorsfile = directory + "/bsdfs/errors.json"

        for si in range(simax):
            if os.path.exists(directory + "/errors/bsdf-errors-" + format(si).zfill(2) + ".json"):
                print("Rendering: ", si)
                tse = renderBSDFComparison(directory, superiteration=si, outprefix=directory + "/bsdfs/bsdfs-" + format(si).zfill(2))
                serrors.append(float(tse))
                json.dump({"sphere-errors": serrors}, open(errorsfile, "w"))
    elif directive == "first":
        if not os.path.exists(directory + "/bsdfs"):
            os.mkdir(directory + "/bsdfs")

        si = 0

        serrors = []

        configfile = directory + "/inputs/config.json"
        config = json.load(open(configfile, "r"))
        imax = config["bsdf-estimator"]["iterations"]
        errorsfile = directory + "/bsdfs/errors-" + format(si).zfill(2) + ".json"

        for i in range(imax):
            if os.path.exists(directory + "/errors/bsdf-errors-" + format(si).zfill(2) + ".json"):
                print("Rendering: ", i)
                if i == 0:
                    tse = renderBSDFComparison(directory, superiteration=si, iteration=i, outprefix=directory + "/bsdfs/bsdfs-" + format(si).zfill(2) + "-" + format(i).zfill(2))
                else:
                    tse = renderBSDFComparison(directory, superiteration=si, iteration=i, outprefix=directory + "/bsdfs/bsdfs-" + format(si).zfill(2) + "-" + format(i).zfill(2), skiptarget=True, skipinitial=True)
                serrors.append(float(tse))
                json.dump({"sphere-errors": serrors}, open(errorsfile, "w"))
    else:
        si = format(directive)
        renderBSDFComparison(directory, superiteration=si, outprefix="bsdfs-" + format(si).zfill(2))