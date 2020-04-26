import json
import os
import hdsutils
import shdsutils
import numpy as np
from dictionary_embedded import embedDictionary
import load_normals
import rawutils
import multiprocessing
import sys

def toMap(lst, vals):
    m = {}
    for l, v in zip(lst, vals):
        m[l] = v
    return m

def mergeMaps(m1, m2):
    m3 = dict(m1)
    for k in m2:
        m3[k] = m2[k]
    return m3

class Parallelizer:

    def __init__(self, k=8):
        self.pool = multiprocessing.Pool(k)

    def parallelRender(self, renderables, kwargss):
        if len(outputs) != testset.numLights():
            print "Error: Expected exactly ", testset.numLights(), " outputs, got ", len(outputs)
            return None

        if len(kwargss) != testset.numLights():
            print "Error: Expected exactly ", testset.numLights(), " kwargss, got ", len(kwargss)
            return None

        renderStrings = [ renderable._makeRenderString(**kwargs)[0] for kwargs, renderable in zip(kwargss, renderables) ]
        outputFiles = [ renderable._makeRenderString(**kwargs)[1] for kwargs, renderable in zip(kwargss, renderables) ]

        data = []

        for outputFile in enumerate(outputFiles):
            if os.path.exists(outputFile):
                os.system("rm " + outputFile)

        self.pool.map(os.system, renderStrings)

        for k, outputFile in enumerate(outputFiles):
            data.append(self.readback(outputFile, self.kwargss[k]))

        return data

    def _render(renderString):
        os.system(renderString)



class MitsubaRenderable:
    def __init__(self, file, outputFile=None):
        self.eParams = {}
        self.params = {}
        self.file = file
        self.outputFile = outputFile

    def setFile(self, file):
        self.file = file
    
    def getFile(self):
        return self.file

    def setParameter(self, key, val):
        self.params[key] = val
    
    def setEmbeddedParameter(self, key, val):
        self.eParams[key] = val
    
    def setOutput(self, outputFile):
        self.outputFile = outputFile

    def _buildEParameterString(self, eParams):
        pstring = ""
        for hparameter in eParams:
            pstring += " -D" + hparameter + "=" + format(eParams[hparameter])
        return pstring

    def _makeRenderString(self, embedded={}, **kwargs):
        params = dict(self.params)
        eParams = dict(self.eParams)
        for e in embedded:
            if embedded[e] is not None:
                eParams[e] = embedded[e]
        
        for p in kwargs:
            if kwargs[p] is not None:
                params[p] = kwargs[p]

        epstring = self._buildEParameterString(eParams)

        outputFile = self.outputFile
        pstring = ""
        if "distribution" in params:
            pstring += " -c " + params["distribution"]

        if "quiet" in params and params["quiet"]:
            pstring += " > /dev/null"
        
        if "localThreads" in params:
            pstring += " -p " + format(params["localThreads"])
        
        if "blockSize" in params:
            pstring += " -b " + format(params["blockSize"])

        if "output" in params:
            outputFile = params["output"]

        command = "mitsuba " + self.file + " -o " + outputFile + " " + epstring + " " + pstring

        return command, outputFile, params, eParams

    def render(self, embedded={}, **kwargs):
        # Make copies
        command, outputFile, params, eParams = self._makeRenderString( embedded, **kwargs)

        if not ("quiet" in params and params["quiet"]):
            print(command)
            sys.stderr.write(command)

        # Remote existing output file before rendering to avoid silent failures.
        if os.path.exists(outputFile):
            os.system("rm " + outputFile)

        os.system(command)

        return outputFile

    def renderReadback(self, readmode="hds", embedded={}, **kwargs):
        outputFile = self.render(embedded, **kwargs)

        return self.readback(outputFile, readmode, embedded, **kwargs)

    def readback(self, readbackPath, readmode="hds", embedded={}, **kwargs):
        outputFile = readbackPath
        params = dict(self.params)
        for p in kwargs:
            if kwargs[p] is not None:
                params[p] = kwargs[p]

        if readmode == "hds":
            return hdsutils.loadHDSImage(outputFile)

        elif readmode == "shds":
            print("numberOfWeights: ", params["numBSDFWeights"])
            return shdsutils.loadSHDS(
                    outputFile,
                    numWeights=params["numBSDFWeights"] if "numBSDFWeights" in params else 0
                )

        elif readmode == "raw":
            if "ignoreIndex" not in self.eParams:
                print("'$ignoreIndex' must be specified as an embedded parameter (not as part of kwargs)")
                return None

            return rawutils.loadRAW(
                outputFile,
                numSlices=self.eParams["ignoreIndex"]
            )

        else:
            print("Invalid readmode ", readmode)
            return None


class Testset:
    def __init__(self, directory):
        self.directory = directory
        self.config = json.load(open(directory + "/config.json", "r"))
        self._setFlags()

        self.runnerParameterList = None
        self.initialBSDFParameters = None
        self.targetBSDFParameters = None

        self.targetWidth = None
        self.targetHeight = None
        self.targetSamples = None
        self.targetDepth = None

        self.renderables = None
        self.gradientRenderables = None

        self._referenceImages = None

        self._load()
        self._buildRenderables()
        self._buildGradientRenderables()

    def _setFlags(self):
        pass

    def _load(self):
        params = self.config
        directory = self.directory

        self.bsdfAdaptiveSampled = "bsdf-adaptive" in params["bsdf-estimator"]["samples"] and params["bsdf-estimator"]["samples"]["bsdf-adaptive"]

        # Embed dictionary if necessary
        if "bsdf-preprocess" in params and params["bsdf-preprocess"]["enabled"]:
            if self.bsdfAdaptiveSampled:
                outSamplesFile = directory + "/" + params["weight-samples-parameter-list"]
            else:
                outSamplesFile = None

            embedDictionary(
                directory + "/" + params["bsdf-preprocess"]["file"],
                directory + "/" + params["scenes"]["intensity"],
                directory + "/" + params["scenes"]["intensity"] + ".pp.xml",
                directory + "/" + params["hyper-parameter-list"],
                directory + "/" + params["zero-padding"],
                self.bsdfAdaptiveSampled,
                outSamplesFile,
                "bsdf-adaptive-mode" in params["bsdf-estimator"]["samples"] and params["bsdf-estimator"]["samples"]["bsdf-adaptive-mode"]
            )

            embedDictionary(
                directory + "/" + params["bsdf-preprocess"]["file"],
                directory + "/" + params["scenes"]["gradient"],
                directory + "/" + params["scenes"]["gradient"] + ".pp.xml",
                directory + "/" + params["hyper-parameter-list"],
                directory + "/" + params["zero-padding"],
                self.bsdfAdaptiveSampled,
                outSamplesFile,
                False
            )

        if type(params["hyper-parameter-list"]) is list:
            runnerParameterList = params["hyper-parameter-list"]
        else:
            runnerParameterList = json.load(open(directory + "/" + params["hyper-parameter-list"], "r"))

        if "hyper-parameters" in params["original"]:
            if type(params["original"]["hyper-parameters"]) is not dict:
                runnerParameterValues = np.load(directory + "/" + params["original"]["hyper-parameters"])
            else:
                runnerParameterValues = params["original"]["hyper-parameters"]
        
        if "hyper-parameters" in params["estimator"]:
            if type(params["estimator"]["hyper-parameters"]) is not dict:
                runnerParameterValues = np.load(directory + "/" + params["estimator"]["hyper-parameters"])
            else:
                runnerParameterValues = params["estimator"]["hyper-parameters"]

        if "hyper-parameters" in params["bsdf-estimator"]:
            if type(params["bsdf-estimator"]["hyper-parameters"]) is not dict:
                runnerParameterValues = np.load(directory + "/" + params["bsdf-estimator"]["hyper-parameters"])
            else:
                runnerParameterValues = params["bsdf-estimator"]["hyper-parameters"]

        self.runnerParameterList = runnerParameterList
        self.initialBSDFParameters = runnerParameterValues

        if "hyper-parameters" in params["target"]:
            if type(params["target"]["hyper-parameters"]) is not dict:
                self.targetBSDFParameters = np.load(directory + "/" + params["target"]["hyper-parameters"])
            else:
                self.targetBSDFParameters = params["target"]["hyper-parameters"]
        
        if "tabular-bsdf" in  params["target"]:
            self.targetTabularBSDFParameters = np.load(directory + "/" + params["target"]["tabular-bsdf"])

        if "tabular-bsdf" in params["bsdf-estimator"]:
            self.initialTabularBSDFParameters  = np.load(directory + "/" + params["bsdf-estimator"]["tabular-bsdf"]["initialization"])

        if ("lights" in params) and ("file" in params["lights"]):
            lightlines = open(directory + "/" + params["lights"]["file"], "r").readlines()
            lights = [np.array([float(l.split(" ")[1]), float(l.split(" ")[2]), float(l.split(" ")[3])]) for l in lightlines]
            self._lightDirections = lights
            #copyfile(directory + "/" + params["lights"]["file"], "lights.lt")

        if ("lights" in params) and ("intensity-file" in params["lights"]):
            intensitylines = open(directory + "/" + params["lights"]["intensity-file"], "r").readlines()
            intensities = [np.array(float(i)) for i in intensitylines]
            params["lights"]["intensity-data"] = intensities
            self._lightIntensities = intensities
            #copyfile(directory + "/" + params["lights"]["file"], "lights.lt")

        if ("lights" in params) and ("intensity-file" not in params["lights"]):
            params["lights"]["intensity-data"] = np.array([10.0] * len(params["lights"]["data"]))
            self._lightIntensities = intensities
        
        if "width" in params["target"]:
            self.targetWidth = params["target"]["width"]
        
        if "height" in params["target"]:
            self.targetHeight = params["target"]["height"]

        if "depth" in params["target"]:
            self.targetDepth = params["target"]["depth"]
        
        if "samples" in params["target"]:
            self.targetSamples = params["target"]["samples"]

        if "mesh" in params["target"]:
            self.targetMeshPath = self.directory + "/" + params["target"]["mesh"]
        
        if "file" in params["initialization"]:
            self._initialMeshPath = self.directory + "/" + params["initialization"]["file"]
        

        # Build BSDF dictionary
        if "bsdf-preprocess" in params:
            self.bsdfDictionary = json.load(open(self.directory + "/" + params["bsdf-preprocess"]["file"], "r"))
            self.bsdfDictionaryElements = self.bsdfDictionary["elements"]
            self._buildBSDFElementStrings()
        else:
            self.bsdfDictionary = None
        
        if params["target"]["type"] == "file":
            # TODO: Load images.
            self._referenceImages = None
        elif params["target"]["type"] == "npy":
            self._referenceImages = np.load(self.directory + "/" + params["target"]["file"])
        
        if "weight-samples-parameter-list" in self.config:
            self.bsdfAdaptiveSamplingParameterList = json.load(open(self.directory + "/" + self.config["weight-samples-parameter-list"], "r"))
        else:
            self.bsdfAdaptiveSamplingParameterList = None

        self._mask = None
        if "mask" in self.config:
            if self.config["mask"]["type"] == "file":
                self._mask = np.load(self.directory + "/" + self.config["mask"]["file"])

        self._targetNormals = None
        if "normals-file" in self.config["target"]:
            self._targetNormals = np.load(self.directory + "/" + self.config["target"]["normals-file"])

    def _buildRenderables(self):
        self.renderables = []

        if self.sceneIntensityPostProcessed() is None:
            return

        for intensity, direction in zip(self.lightIntensities(), self.lightDirections()):
            renderable = MitsubaRenderable(self.sceneIntensityPostProcessed())
            renderable.setEmbeddedParameter("lightX", direction[0])
            renderable.setEmbeddedParameter("lightY", direction[1])
            renderable.setEmbeddedParameter("lightZ", direction[2])
            renderable.setEmbeddedParameter("irradiance", intensity)

            if self.targetMeshPath is not None:
                renderable.setEmbeddedParameter("mesh", self.targetMeshPath)

            if self.targetWidth is not None:
                renderable.setEmbeddedParameter("width", self.targetWidth)
            else:
                renderable.setEmbeddedParameter("width", 256)

            if self.targetHeight is not None:
                renderable.setEmbeddedParameter("height", self.targetHeight)
            else:
                renderable.setEmbeddedParameter("height", 256)

            if self.targetDepth is not None:
                renderable.setEmbeddedParameter("depth", self.targetDepth)
            else:
                renderable.setEmbeddedParameter("depth", -1)
            
            if self.targetSamples is not None:
                renderable.setEmbeddedParameter("sampleCount", self.targetSamples)
            else:
                renderable.setEmbeddedParameter("sampleCount", 64)
            
            # Set target BSDF mapping
            if self.targetBSDFMap() is not None:
                for k, v in self.targetBSDFMap():
                    renderable.setEmbeddedParameter(k, v)
            
            # Set target BSDF sampling
            if self.bsdfAdaptiveSampled:
                for k, v in zip(self.bsdfAdaptiveSamplingParameterList, self.targetBSDFParameters):
                    renderable.setEmbeddedParameter(k, v)

            renderable.setParameter("blockSize", 8)

            self.renderables.append(renderable)
    
    def _buildGradientRenderables(self):
        self.gradientRenderables = []

        if self.sceneGradientPostProcessed() is None:
            return

        for intensity, direction in zip(self.lightIntensities(), self.lightDirections()):
            renderable = MitsubaRenderable(self.sceneGradientPostProcessed())
            renderable.setEmbeddedParameter("lightX", direction[0])
            renderable.setEmbeddedParameter("lightY", direction[1])
            renderable.setEmbeddedParameter("lightZ", direction[2])
            renderable.setEmbeddedParameter("irradiance", intensity)
            renderable.setEmbeddedParameter("meshSlot", 0)

            if self.targetMeshPath is not None:
                renderable.setEmbeddedParameter("mesh", self.targetMeshPath)

            if self.targetWidth is not None:
                renderable.setEmbeddedParameter("width", self.targetWidth)
            else:
                renderable.setEmbeddedParameter("width", 256)

            if self.targetHeight is not None:
                renderable.setEmbeddedParameter("height", self.targetHeight)
            else:
                renderable.setEmbeddedParameter("height", 256)

            if self.targetDepth is not None:
                renderable.setEmbeddedParameter("depth", self.targetDepth)
            else:
                renderable.setEmbeddedParameter("depth", -1)

            if self.targetSamples is not None:
                renderable.setEmbeddedParameter("sampleCount", self.targetSamples)
            else:
                renderable.setEmbeddedParameter("sampleCount", 64)

            renderable.setParameter("blockSize", 8)

            renderable.setParameter(
                    "numBSDFWeights", 
                    len(self.bsdfDictionary["elements"]) + self.zeroPadding()
                )
            self.gradientRenderables.append(renderable)

    def _buildBSDFElementStrings(self):
        self.bsdfDictionaryElementStrings = []
        for element in self.bsdfDictionaryElements:
            if element["type"] == "diffuse":
                self.bsdfDictionaryElementStrings.append("diffuse/r:" + format(element["reflectance"]))
            elif element["type"] == "roughconductor":
                self.bsdfDictionaryElementStrings.append("ggx/a:" + format(element["alpha"]) + "/eta:" + format(element["eta"]))
            else:
                self.bsdfDictionaryElementStrings.append("unknown")

    def renderable(self, i):
        try:
            return self.renderables[i]
        except:
            return None

    def numLights(self):
        return len(self._lightDirections)

    def targetNormals(self):
        return self._targetNormals

    def initialMeshPath(self):
        return self._initialMeshPath

    def mask(self):
        return self._mask

    def lightIntensities(self):
        return self._lightIntensities

    def lightDirections(self):
        return self._lightDirections

    def parameterList(self):
        return self.runnerParameterList

    def numIterations(self):
        return self.config["estimator"]["iterations"]
    
    def numBSDFIterations(self):
        return self.config["bsdf-estimator"]["iterations"]

    def dictionaryPath(self):
        if "bsdf-preprocess" in self.config:
            return self.config["bsdf-preprocess"]["file"]

    def initialBSDFMap(self):
        if self.initialBSDFParameters is not None and self.runnerParameterList is not None:
            return zip(self.runnerParameterList, self.initialBSDFParameters)

    def targetBSDFMap(self):
        if self.targetBSDFParameters is not None and self.runnerParameterList is not None:
            return zip(self.runnerParameterList, self.targetBSDFParameters)

    def initialBSDF(self):
        return self.initialBSDFParameters

    def targetBSDF(self):
        return self.targetBSDFParameters

    def targetTabularBSDF(self):
        return self.targetTabularBSDFParameters

    def initialTabularBSDF(self):
        return self.initialTabularBSDFParameters

    def zeroPadding(self):
        params = self.config
        if "zero-padding" in params and type(params["zero-padding"]) is int:
            zeroPadding = parameters["zero-padding"]
        elif "zero-padding" in params:
            zeroPadding = json.load(open(self.directory + "/" + params["zero-padding"], "r"))[0]
        else:
            zeroPadding = 0

        return zeroPadding

    def isTargetBSDFParametric(self):
        return self.targetBSDFParameters != None

    def isBSDFAdaptiveSampled(self):
        return self.bsdfAdaptiveSampled

    def sceneIntensity(self):
        ifile = self.directory + "/" + self.config["scenes"]["intensity"]
        if os.path.exists(ifile):
            return ifile

    def sceneIntensityPostProcessed(self):
        ifile = self.directory + "/" + self.config["scenes"]["intensity"] + ".pp.xml"
        if os.path.exists(ifile):
            return ifile

    def sceneGradient(self):
        gfile = self.directory + "/" + self.config["scenes"]["gradient"]
        if os.path.exists(gfile):
            return gfile

    def sceneGradientPostProcessed(self):
        ifile = self.directory + "/" + self.config["scenes"]["gradient"] + ".pp.xml"
        if os.path.exists(ifile):
            return ifile
    
    def sceneColors(self):
        cfile = self.directory + "/" + self.config["scenes"]["colors"]
        if os.path.exists(cfile):
            return cfile

    def embedDictionary(self):
        pass

    def referenceImages(self):
        # If there are reference images available return them immediately.
        return self._referenceImages

    def embedOnto(self, filename, outfile=None, adaptive=False, ignore=True):
        if outfile is None:
            outfile = filename + ".pp.xml"
        embedDictionary(
                self.directory + "/" + self.config["bsdf-preprocess"]["file"],
                filename,
                outfile,
                "/tmp/hplist.json",
                "/tmp/zplist.json",
                adaptive,
                "/tmp/samples.json",
                ignore
            )


# Main dataset reader. For rapid development times.
class Dataset:
    def __init__(self, directory):
        self.directory = directory
        self.testset = Testset(directory + "/inputs")

        self.itercounts = None
        self.superitercount = None

        self.latestBSDF = None
        self.BSDFs = None
        self.tabularBSDFs = None

        MAX_ITERS = 20000
        MAX_SUPERITERS = 20000
        self._load()

    def _load(self):
        self.BSDFs = []
        self.tabularBSDFs = []
        for i in range(self.testSet().config["remesher"]["iterations"]):
            befile = self.directory + "/errors/bsdf-errors-" + format(i).zfill(2) + ".json"
            if os.path.exists(befile):
                self.berrordata = json.load(open(befile, "r"))
                self.latestBSDF = self.berrordata["bvals"][-1]
                self.BSDFs.append(self.berrordata["bvals"])
                self.tabularBSDFs.append(self.berrordata["tbvals"])
        
        if self.testSet().bsdfAdaptiveSampled:
            self.bsdfAdaptiveSamplingParameterList = json.load(open(self.directory + "/" + self.testSet().config["weight-samples-parameter-list"], "r"))

    def testSet(self):
        return self.testset

    def itercount(self):
        pass

    def superitercount(self):
        pass

    def lastAvailableBSDF(self):
        return self.latestBSDF[:-self.testSet().zeroPadding()]

    def lastAvailableBSDFList(self):
        pass

    def initialBSDF(self):
        return self.testset.initialBSDF()

    def BSDFAt(self, iteration=0, superiteration=0):
        try:
            return np.array(self.BSDFs[superiteration][iteration])
        except:
            return None

    def _loadBSDFErrorFile(self, iteration=0, superiteration=0):
        try:
            return json.load(
                 open(self.directory + "/errors/bsdf-errors-" + format(superiteration).zfill(2) + "-" + format(iteration).zfill(4) + ".json", "r"))
        except Exception as e:
            print(e)
            return None

    def tabularBSDFAt(self, iteration=0, superiteration=0):
        print(len(self.tabularBSDFs))
        if len(self.tabularBSDFs) == 0:
            try:
                return np.array(self._loadBSDFErrorFile(iteration, superiteration)["tbvals"])
            except Exception as e:
                print(e)
                return None

        try:
            return np.array(self.tabularBSDFs[superiteration][iteration])
        except:
            return None
    

    def meshfileAt(self, iteration, superiteration):
        meshfile = self.directory + "/meshes/normals/" +\
                    format(superiteration).zfill(2) + "/" +\
                    format(iteration).zfill(4) + ".ply"
        if os.path.exists(meshfile):
            return meshfile
        else:
            return None

    def BSDFListAt(self, superiteration=0):
        try:
            return self.BSDFs[superiteration]
        except:
            return None

    def errorAtN(self, lindex, iteration=0, superiteration=0):
        return np.load(self.directory + "/images/difference-errors/npy/" + format(superiteration).zfill(2) + "/" + format(iteration).zfill(4) +
                "-img-" + format(lindex).zfill(2) + ".npy")
