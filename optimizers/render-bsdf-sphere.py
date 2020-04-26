# Renders BSDF onto a sphere.
import optparse
import os
from dataset_reader import Dataset, Testset, mergeMaps, toMap
import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt

parser = optparse.OptionParser()
parser.add_option("-I", "--super-iteration", dest="superIndex", default=-1)
parser.add_option("-i", "--iteration", dest="index", default=-1)
parser.add_option("-s", "--sample-count", dest="sampleCount", default=256)
parser.add_option("-c", "--distribution", dest="distribution", default=None, type="str")
parser.add_option("-o", "--output", dest="output", type="str")
parser.add_option("-d", "--dry-run", action="store_true", dest="dryRun")

(options, args) = parser.parse_args()

directory = args[0]

dataset = Dataset(directory)
if options.superIndex == -1 and options.index == -1:
    targetBSDF = dataset.lastAvailableBSDF()
else:
    targetBSDF = dataset.BSDFAt(iteration=options.index, superiteration=options.superIndex)
print targetBSDF

if options.dryRun:
    sys.exit(0)

testset = dataset.testSet()

sphereXML = os.path.dirname(__file__) + "/data/sphere-embeddable.xml"
sphereXMLP = os.path.dirname(__file__) + "/data/sphere-postprocessed.xml"
testset.embedOnto(sphereXML, sphereXMLP)

renderable = testset.renderables[0]
renderable.setFile(sphereXMLP)
renderable.setEmbeddedParameter("envmap", "doge.exr")

if options.distribution is not None:
    localThreads = 2
else:
    localThreads = 8

sphereImage = renderable.renderReadback(
        readmode="hds",
        distribution=options.distribution,
        output="/tmp/sphere-bsdf-testing.hds",
        embedded=mergeMaps(
                {"sampleCount": options.sampleCount},
                toMap(testset.parameterList(), targetBSDF)
        ),
        localThreads=localThreads,
        quiet=False)

print(testset.targetBSDF())
sphereImageTarget = renderable.renderReadback(
        readmode="hds",
        distribution=options.distribution,
        output="/tmp/sphere-bsdf-testing.hds",
        embedded=mergeMaps(
                {"sampleCount": options.sampleCount},
                toMap(testset.parameterList(), testset.targetBSDF())
        ),
        localThreads=localThreads,
        quiet=False)


fullImage = np.concatenate([sphereImage, sphereImageTarget], axis=1)
print(fullImage.shape)
plt.imshow(fullImage[:,:,0])
plt.show()

cv2.imwrite(options.output, (fullImage * 255).astype(np.uint8))