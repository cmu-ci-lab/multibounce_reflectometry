
import np2exr
import sys
import os

def exrify(d):
    for f in os.listdir(d):
        if f.endswith(".n.npy"):
            f = d + "/" + f
            print(f)
            negfile = f
            posfile = f.replace(".n.npy", ".p.npy")
            if not os.path.exists(posfile):
                continue
            xexrfile = f.replace(".n.npy", ".x.exr")
            yexrfile = f.replace(".n.npy", ".y.exr")
            zexrfile = f.replace(".n.npy", ".z.exr")
            np2exr.developCompositeFromFiles(negfile, posfile, [xexrfile, yexrfile, zexrfile], mode="XYZ")
        elif os.path.isdir(d + "/" + f):
            exrify(d + "/" + f)

if __name__ == '__main__':
    exrify(sys.argv[1])