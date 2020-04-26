# Convert local testset into Diligent.
import numpy as np
import scipy.io.matlab as matlab
import cv2
import sys
import os
import optparse

# Load the testset tools.
from dataset_reader import Testset
#execfile(os.path.dirname(__file__) + "/../optimizers/dataset_reader.py")

parser = optparse.OptionParser()
parser.add_option("-c", "--distribution", type="str", default=None, dest="distribution")
parser.add_option("-e", "--exposure", type="float", default=1.0, dest="exposure")
parser.add_option("-s", "--ref-samples", type="int", default=128, dest="refSamples")
(options, args) = parser.parse_args()

directory = args[0]
target = args[1]

if not os.path.exists(target):
    os.mkdir(target)

if not len(os.listdir(target)) == 0:
    print("Target directory should be empty")
    sys.exit(1)

testset = Testset(directory)

images = None
if testset.referenceImages() is None:
    # No reference images available. Render them
    references = []
    for k in range(testset.numLights()):
        testset.renderables[k].setEmbeddedParameter("sampleCount", options.refSamples)
        testset.renderables[k].setEmbeddedParameter("blockSize", 8)

        reference = testset.renderables[k].renderReadback(
            readmode="hds",
            distribution=options.distribution,
            output="/tmp/testset2diligent-temp.hds",
            localThreads=2,
            quiet=False
        )
        references.append(reference)
    images = np.stack(references, axis=2)
else:
    images = testset.referenceImages()

# Step 1: Write image files
print("Writing image files...")
filenames = []
for i in range(testset.numLights()):
    filename = format(i + 1).zfill(3) + ".png"
    cv2.imwrite(target + "/" + filename, np.clip(images[:,:,i] * options.exposure * 65536.0, 0.0, 65536.0).astype(np.uint16))
    filenames.append(filename)

# Step 2: Write filename list
print("Writing filename list...")
open(target + "/filenames.txt", "w").write("".join([filename + "\n" for filename in filenames])[:-1])

# Step 3: Write directions
print("Writing directions...")
directions = [ format(-d[0]) + " " + format(-d[1]) + " " + format(-d[2]) for d in testset.lightDirections() ]
open(target + "/light_directions.txt", "w").write("".join([ d + "\n" for d in directions]))

# Step 4: Write intensities
print("Writing intensities...")
open(target + "/light_intensities.txt", "w").write("".join([ format(i * options.exposure) + "\n" for i in testset.lightIntensities() ]))

# Step 5: Write mask
print("Writing mask...")
cv2.imwrite(target + "/mask.png", testset.mask() * 65536.0)

# Step 6: Write the ground truth normals.
print("Writing ground truth normals...")
matlab.savemat(target + "/Normal_gt.mat", {"Normal_gt": -testset.targetNormals()})
normals = testset.targetNormals()
nfiletext = ""
for x in normals.reshape((normals.shape[0] * normals.shape[1] * normals.shape[2],)):
    nfiletext += (format(x) + "\n")
open(target + "/normal.txt", "w").write(nfiletext[:-1])

# --
print("Done.")