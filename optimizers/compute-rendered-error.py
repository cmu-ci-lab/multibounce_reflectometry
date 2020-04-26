import optparse
import os
import json
import numpy as np
from shutil import copyfile
from dataset_reader import Dataset, Testset
import merl_io

parser = optparse.OptionParser()
parser.add_option("-s", "--samples", dest="sampleCount", default=1024, type="int")
parser.add_option("-e", "--envmaps", dest="envmaps", default="uffizi;doge;field")
parser.add_option("-i", "--iteration", dest="iteration", default=None, type="int")
parser.add_option("-c", "--distribution", dest="distribution", default=None, type="string")
parser.add_option("-k", "--skip", dest="skip", default=1, type="int")

(options, args) = parser.parse_args()

envmaps = [envmap for envmap in options.envmaps.split(";")]

dataset = Dataset(args[0])

directory = args[0]

testset = dataset.testSet()

sphereXML = os.path.dirname(__file__) + "/data/sphere-embeddable.xml"
sphereXMLP = os.path.dirname(__file__) + "/data/sphere-postprocessed.xml"
testset.embedOnto(sphereXML, sphereXMLP)

renderable = testset.renderables[0]
renderable.setEmbeddedParameter("meshSlot", 0)
renderable.setFile(sphereXMLP)

renderable.setEmbeddedParameter("width", 64)
renderable.setEmbeddedParameter("height", 64)

Es = []
baseDir = directory + "/tabular-render-compare/"

if not os.path.exists(baseDir):
    os.mkdir(baseDir)


targets = []

for envmap in envmaps:
    print "Rendering target sphere: ", envmap

    renderable.setEmbeddedParameter("envmap", envmap + ".exr")

    merl_io.merl_write("/tmp/tabular-bsdf-0.binary", testset.targetTabularBSDF())

    sphereImageTarget = renderable.renderReadback(
        readmode="hds",
        distribution=options.distribution,
        output="/tmp/sphere-bsdf-testing.hds",
        embedded={"sampleCount": options.sampleCount},
        localThreads=2,
        quiet=True)

    np.save(baseDir + "/target-" + envmap + ".npy", sphereImageTarget)

    targets.append(sphereImageTarget)

for i in range(-1, testset.numBSDFIterations(), options.skip):
    if options.iteration != None and options.iteration != i:
        continue

    if dataset.tabularBSDFAt(iteration=i) is None and i != -1:
        continue

    if i != -1:
        print("Rendering iteration " + format(i) + "...")
    else:
        print("Rendering initialization")

    error = 0.0
    for eidx, envmap in enumerate(envmaps):
        renderable.setEmbeddedParameter("envmap", envmap + ".exr")

        if i != -1:
            merl_io.merl_write("/tmp/tabular-bsdf-0.binary", dataset.tabularBSDFAt(iteration=i))
        else:
            merl_io.merl_write("/tmp/tabular-bsdf-0.binary", testset.initialTabularBSDF())

        sphereImage = renderable.renderReadback(
            readmode="hds",
            distribution=options.distribution,
            output="/tmp/sphere-bsdf-testing.hds",
            embedded={"sampleCount": options.sampleCount},
            localThreads=2,
            quiet=False)

        if i == -1:
            np.save(baseDir + "/init-" + envmap + ".npy", sphereImage)
            np.save(baseDir + "/init-error-" + envmap + ".npy", sphereImageTarget - sphereImage)
        else:
            np.save(baseDir + "/" + format(i).zfill(4) + "-" + envmap + ".npy", sphereImage)
            np.save(baseDir + "/" + format(i).zfill(4) + "-error-" + envmap + ".npy", sphereImageTarget - sphereImage)

        error += np.sum((sphereImage - targets[eidx]) ** 2)

    Es.append(error)
    print("Error @ " + format(i).zfill(4) + ": " + format(error))

    np.save(baseDir + "/bsdf-envmap-errors.json", {"errors": Es})
