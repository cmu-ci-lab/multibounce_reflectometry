
import np2exr
import sys
import os
import numpy as np

def exrify(d):
    for f in os.listdir(d):
        if f.endswith(".npy"):
            f = d + "/" + f
            print(f)
            negfile = f
            xexrfile = f.replace(".npy", ".x.exr")
            yexrfile = f.replace(".npy", ".y.exr")
            zexrfile = f.replace(".npy", ".z.exr")
            exrfile = f.replace(".npy", ".exr")
            
            s = np.load(f).shape
            if len(s) == 3:
                np2exr.developCompositeFromFile(f, [xexrfile, yexrfile, zexrfile], mode="XYZ")
            elif len(s) == 2:
                np2exr.developSimpleFromFiles(f, exrfile)

        elif os.path.isdir(d + "/" + f):
            exrify(d + "/" + f)

if __name__ == '__main__':
    exrify(sys.argv[1])