
import optparse
import boto3
import os
import datetime
import termcolor

from dataset_reader import Dataset, Testset, toMap, mergeMaps

import merl_io

from shutil import copyfile
import numpy as np

import matplotlib.pyplot as plt
import hdsutils

parser = optparse.OptionParser()
parser.add_option("-e", "--envmap", dest="envmap", default="uffizi")
parser.add_option("--samples", dest="sampleCount", default=1024, type="int")
parser.add_option("-c", "--distribution", dest="distribution", default=None)
parser.add_option("-o", "--output", dest="output", default=None)
parser.add_option("-t", "--threshold", dest="threshold", default=0.8, type="float")
parser.add_option("-b", "--bsdf-file", dest="bsdfFile", default="/tmp/tabular-bsdf-0.binary")
parser.add_option("-r", "--reductor", dest="reductor", default=None)
parser.add_option("--resolution", dest="resolution", default=64)

(options, args) = parser.parse_args()

directory = args[0]

dataset = Dataset(directory)
testset = dataset.testSet()

# Prep sphere renderables by embedding the BSDF format onto the sphere scene.
sphereXML = os.path.dirname(__file__) + "/data/sphere-gradient-embeddable.xml"
sphereXMLP = os.path.dirname(__file__) + "/data/sphere-gradient-postprocessed.xml"
testset.embedOnto(sphereXML, sphereXMLP)

renderable = testset.gradientRenderables[0]
renderable.setEmbeddedParameter("meshSlot", 0)
renderable.setFile(sphereXMLP)


Es = []

if options.reductor is None:
    print("Specify reductor (error texture) using -r <npy-file-path>")
    sys.exit(1)

reductor = np.load(options.reductor)
#reductor[reductor > options.threshold] = options.threshold
#reductor[reductor < -options.threshold] = -options.threshold
reductor[reductor < options.threshold] = 0
reductor[reductor > -options.threshold] = 0

hdsutils.writeHDSImage("/tmp/reductor-0.hds", reductor.shape[0], reductor.shape[1], reductor.shape[2], reductor)


renderable.setEmbeddedParameter("width", reductor.shape[0])
renderable.setEmbeddedParameter("height", reductor.shape[1])

renderable.setEmbeddedParameter("envmap", options.envmap + ".exr")

#merl_io.merl_write("/tmp/tabular-bsdf-0.binary", dataset.tabularBSDFAt(iteration=options.iteration))
if options.bsdfFile != "/tmp/tabular-bsdf-0.binary":
    copyfile(options.bsdfFile, "/tmp/tabular-bsdf-0.binary")

print "Rendering coverage"
_,_,coverage = renderable.renderReadback(
        readmode="shds",
        distribution=options.distribution,
        output="/tmp/coverage-testing.shds",
        embedded={"sampleCount": options.sampleCount},
        localThreads=2,
        quiet=False)

print("Ours: ", np.sum((sphereImage - sphereImageTarget) ** 2))
print("Initial: ", np.sum((sphereImageInitial - sphereImageTarget) ** 2))
print("Compare: ", np.sum((sphereImageCompare - sphereImageTarget) ** 2))

#np.save(baseDir + "/" + format(i).zfill(4) + "-" + envmap + ".npy", sphereImage)
#np.save(baseDir + "/target-" + envmap + ".npy", sphereImageTarget)
#np.save(baseDir + "/" + format(i).zfill(4) + "-error-" + envmap + ".npy", sphereImageTarget - sphereImage)

np.save(baseDir + "/bsdf-envmap-errors.json", {"errors": Es})