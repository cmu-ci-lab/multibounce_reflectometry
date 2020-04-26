from photometric import photometric
import optparse
import cv2
import scipy.io.matlab as matlab
import numpy as np
import matplotlib.pyplot as plt
import os

parser = optparse.OptionParser()

parser.add_option("-m", "--mask", dest="mask", default=None)
parser.add_option("-i", "--images", dest="images", default=None)
parser.add_option("-d", "--directions",  dest="directions", default=None)
parser.add_option("-k", "--intensities",  dest="intensities", default=None)

(options, args) = parser.parse_args()

mask = cv2.imread(options.mask)
mask = mask > 0.1

directions = matlab.loadmat(options.directions)["directions"]
intensities = matlab.loadmat(options.intensities)["intensities"]

directions = directions * intensities.T

img1 = cv2.imread(options.images + "/00.tiff", -1)
W,H,_ = img1.shape

images = []
for i in range(len(os.listdir(options.images))):
    images.append(
        cv2.resize(
            cv2.cvtColor(
                    cv2.imread(options.images + "/" + format(i).zfill(2) + ".tiff", -1),
                            cv2.COLOR_RGB2GRAY
                ),
            (H//4, W//4)
            )
        )

images = np.stack(images, axis=2)

n, intensities = photometric(images.transpose([2,0,1]), directions.T)
plt.imshow(n*0.5 + 0.5)
plt.show()