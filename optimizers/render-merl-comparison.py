
import optparse
import boto3
import os
import datetime
import termcolor

from dataset_reader import Dataset, Testset, toMap, mergeMaps

import merl_io

from shutil import copyfile
import numpy as np

parser = optparse.OptionParser()
parser.add_option("-i", "--iteration",
                  dest="iteration", default=0, type="int")
parser.add_option("-s", "--super-iteration",
                  dest="superiteration", default=0, type="int")
parser.add_option("-m", "--compare", 
                  dest="compare", default=None)
parser.add_option("-e", "--envmaps", dest="envmaps", default="uffizi;doge;field")
parser.add_option("-r", "--resolution", dest="resolution", default=64, type="int")
parser.add_option("--samples", dest="sampleCount", default=1024, type="int")
parser.add_option("-c", "--distribution", dest="distribution", default=None)
parser.add_option("-o", "--output", dest="output", default=None)
parser.add_option("-g", "--coverage", dest="coverage", default=None)
parser.add_option("--source-mesh", dest="sourceMesh", action="store_true", default=False)

(options, args) = parser.parse_args()


directory = args[0]

dataset = Dataset(directory)
testset = dataset.testSet()

envmaps = [envmap for envmap in options.envmaps.split(";") if envmap != ""]

if options.compare is None:
    print("Provide a comparison BRDF as -m <merl-binary>")

# Prep sphere renderables by embedding the BSDF format onto the sphere scene.
sphereXML = os.path.dirname(__file__) + "/data/sphere-embeddable.xml"
sphereXMLP = os.path.dirname(__file__) + "/data/sphere-postprocessed.xml"
testset.embedOnto(sphereXML, sphereXMLP)

renderable = testset.renderables[0]
renderable.setEmbeddedParameter("meshSlot", 0)

if not options.sourceMesh:
    renderable.setFile(sphereXMLP)

renderable.setEmbeddedParameter("width", options.resolution)
renderable.setEmbeddedParameter("height", options.resolution)

Es = []
baseDir = directory + "/merl-compare/"

if not os.path.exists(baseDir):
    os.mkdir(baseDir)

if dataset.tabularBSDFAt(iteration=options.iteration) is None:
    print "Couldn't find tabular BSDF at i=", options.iteration
    sys.exit(1)

for envmap in envmaps:
    renderable.setEmbeddedParameter("envmap", envmap + ".exr")

    merl_io.merl_write("/tmp/tabular-bsdf-0.binary", dataset.tabularBSDFAt(iteration=options.iteration))

    print "Rendering ", envmap, " - ", " Optimized"
    sphereImage = renderable.renderReadback(
        readmode="hds",
        distribution=options.distribution,
        output="/tmp/sphere-bsdf-testing.hds",
        embedded={"sampleCount": options.sampleCount},
        localThreads=2,
        quiet=True)

    merl_io.merl_write("/tmp/tabular-bsdf-0.binary", testset.initialTabularBSDF())

    print "Rendering ", envmap, " - ", " Initial"
    sphereImageInitial = renderable.renderReadback(
        readmode="hds",
        distribution=options.distribution,
        output="/tmp/sphere-bsdf-testing.hds",
        embedded={"sampleCount": options.sampleCount},
        localThreads=2,
        quiet=True)

    merl_io.merl_write("/tmp/tabular-bsdf-0.binary", testset.targetTabularBSDF())

    print "Rendering ", envmap, " - ", " Target"
    sphereImageTarget = renderable.renderReadback(
        readmode="hds",
        distribution=options.distribution,
        output="/tmp/sphere-bsdf-testing.hds",
        embedded={"sampleCount": options.sampleCount},
        localThreads=2,
        quiet=True)

    copyfile("/tmp/tabular-bsdf-0.binary", options.compare)

    print "Rendering ", envmap, " - ", " Compare"
    sphereImageCompare = renderable.renderReadback(
        readmode="hds",
        distribution=options.distribution,
        output="/tmp/sphere-bsdf-testing.hds",
        embedded={"sampleCount": options.sampleCount},
        localThreads=2,
        quiet=True)

    print("Ours: ", np.sum((sphereImage - sphereImageTarget) ** 2))
    print("Initial: ", np.sum((sphereImageInitial - sphereImageTarget) ** 2))
    print("Compare: ", np.sum((sphereImageCompare - sphereImageTarget) ** 2))

    dOurs = np.concatenate([
        sphereImage,
        sphereImageTarget,
        np.abs(sphereImage - sphereImageTarget) * 10
    ], axis=0)

    dInitial = np.concatenate([
        sphereImageInitial,
        sphereImageTarget,
        np.abs(sphereImageInitial - sphereImageTarget) * 10
    ], axis=0)

    dCompare = np.concatenate([
        sphereImageCompare,
        sphereImageTarget,
        np.abs(sphereImageCompare - sphereImageTarget) * 10
    ], axis=0)

    """np.save(envmap + "-" + options.output, np.concatenate([
            sphereImage,
            sphereImageTarget,
            sphereImageInitial,
            sphereImageCompare
        ], axis=1))
    """

    np.save(baseDir + envmap + "-compare.npy", np.concatenate([
            dOurs,
            dInitial,
            dCompare
        ], axis=1))

    np.save(baseDir + envmap + "-ours-error.npy", np.abs(sphereImage - sphereImageTarget) * 10)
    np.save(baseDir + envmap + "-initial-error.npy", np.abs(sphereImageInitial - sphereImageTarget) * 10)
    np.save(baseDir + envmap + "-compare-error.npy", np.abs(sphereImageCompare - sphereImageTarget) * 10)
    
    #np.save(baseDir + "/" + format(i).zfill(4) + "-" + envmap + ".npy", sphereImage)
    #np.save(baseDir + "/target-" + envmap + ".npy", sphereImageTarget)
    #np.save(baseDir + "/" + format(i).zfill(4) + "-error-" + envmap + ".npy", sphereImageTarget - sphereImage)



np.save(baseDir + "/bsdf-envmap-errors.json", {"errors": Es})