import sys
import os
import OpenEXR
import numpy as np
import Imath
import array
from PIL import Image

from libtiff import TIFF
from shutil import copyfile

directory = sys.argv[1]

def developSimpleFromFiles(fname, ofname):
    developSimple(np.load(fname), ofname)

def developSimple(vals, ofname):
    W,H = vals.shape[:2]

    if len(vals.shape) == 3 and vals.shape[2] == 3:
        R = vals[:,:,0]
        G = vals[:,:,1]
        B = vals[:,:,2]

        (Rs, Gs, Bs) = [ array.array('f', chan.reshape((W*H)).tolist()).tostring() for chan in (R, G, B)]

        # Write the three color channels to the output file
        out = OpenEXR.OutputFile(ofname, OpenEXR.Header(W, H))
        out.writePixels({'R' : Rs, 'G' : Gs, 'B' : Bs})

    elif len(vals.shape) == 2:
        Y = vals

        (Ys,) = [array.array('f', chan.reshape((W*H)).tolist()).tostring() for chan in (Y,)]

        # Write the three color channels to the output file
        out = OpenEXR.OutputFile(ofname, OpenEXR.Header(W, H))
        out.writePixels({'R' : Ys, 'G' : Ys, 'B' : Ys})
        #out.writePixels({'Y' : Ys})
    else:
        print("Only 1 and 3 channel images supported")
        return


def developCompositeFromFiles(negfname, posfname, ofname, mode="XYZ"):
    # Open the polarized input files
    nvals = np.load(negfname)
    pvals = np.load(posfname)
    developComposite(nvals, pvals, ofname, mode)

def developCompositeFromFile(fname, ofname, mode="XYZ"):
    vals = np.load(fname)
    nvals = (vals < 0) * vals
    pvals = (vals > 0) * vals
    developComposite(nvals, pvals, ofname, mode)

def developComposite(nvals, pvals, ofnames, mode="XYZ"):
    W,H = nvals.shape[:2]
    pX = pvals[:,:,0]
    pY = pvals[:,:,1]
    pZ = pvals[:,:,2]
    nX = nvals[:,:,0]
    nY = nvals[:,:,1]
    nZ = nvals[:,:,2]

    (pXs, pYs, pZs, nXs, nYs, nZs) = [ array.array('f', chan.reshape((W*H)).tolist()).tostring() for chan in (pX, pY, pZ, nX, nY, nZ)]

    if mode == "PN":
        # Write the three color channels to the output file
        out = OpenEXR.OutputFile(ofnames[0], OpenEXR.Header(W, H))
        #out.writePixels({'pX' : pXs, 'pY' : pYs, 'pZ' : pZs, 'nX' : nXs, 'nY' : nYs, 'nZ' : nZs})
        out.writePixels({'R' : pXs, 'G' : pYs, 'B' : pZs})
        # Write the three color channels to the output file
        out = OpenEXR.OutputFile(ofnames[1], OpenEXR.Header(W, H))
        #out.writePixels({'pX' : pXs, 'pY' : pYs, 'pZ' : pZs, 'nX' : nXs, 'nY' : nYs, 'nZ' : nZs})
        out.writePixels({'R' : nXs, 'G' : nYs, 'B' : nZs})
    elif mode == "XYZ":
        # Write the three color channels to the output file
        out = OpenEXR.OutputFile(ofnames[0], OpenEXR.Header(W, H))
        #out.writePixels({'pX' : pXs, 'pY' : pYs, 'pZ' : pZs, 'nX' : nXs, 'nY' : nYs, 'nZ' : nZs})
        out.writePixels({'R' : nXs, 'B' : pXs})
        # Write the three color channels to the output file
        out = OpenEXR.OutputFile(ofnames[1], OpenEXR.Header(W, H))
        #out.writePixels({'pX' : pXs, 'pY' : pYs, 'pZ' : pZs, 'nX' : nXs, 'nY' : nYs, 'nZ' : nZs})
        out.writePixels({'R' : nYs, 'B' : pYs})
        # Write the three color channels to the output file
        out = OpenEXR.OutputFile(ofnames[2], OpenEXR.Header(W, H))
        #out.writePixels({'pX' : pXs, 'pY' : pYs, 'pZ' : pZs, 'nX' : nXs, 'nY' : nYs, 'nZ' : nZs})
        out.writePixels({'R' : nZs, 'B' : pZs})