import sys
import os
import OpenEXR
import numpy as np
import Imath
import array
from PIL import Image

from libtiff import TIFF
from shutil import copyfile
#if len(sys.argv) != 3:
#    print "usage: exrnormalize.py exr-input-file exr-output-file"
#    sys.exit(1)

def convert(fname, ofname, multiplier=1.0, channel='Y'):
    # Open the input file
    f = OpenEXR.InputFile(fname)

    # Compute the size
    dw = f.header()['dataWindow']
    sz = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

    # Read the three color channels as 32-bit floats
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)

    #print(fname)
    (Y) = [array.array('f', f.channel(Chan, FLOAT)).tolist() for Chan in (channel) ]
    Y = np.array(Y).reshape(sz) * multiplier

    tiff = TIFF.open(ofname, mode='w')

    tiff.write_image(np.clip(Y * 64, 0, 255).astype(np.uint8))
    tiff.close()

if __name__ == "__main__":
    directory = sys.argv[1]
    multiplier = 1.0
    if len(sys.argv) > 2:
        multiplier = float(sys.argv[2])

    if not os.path.exists(directory + "/tiff"):
        os.mkdir(directory + "/tiff")

    for d in os.listdir(directory):
        fname = directory + "/" + d
        ofname = directory + "/tiff/" + d
        if d.endswith(".exr"):
            convert(fname, ofname.replace(".exr", ".tif"), multiplier)

    copyfile(directory + "/sources.mat", directory + "/tiff/sources.mat")