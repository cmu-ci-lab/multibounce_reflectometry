# Obtain normals from lighting

import numpy as np
import math

def load_from_csv(fprefix, W, H):
    f1 = open(fprefix+"_1.csv", "r").readlines();
    f2 = open(fprefix+"_2.csv", "r").readlines();
    f3 = open(fprefix+"_3.csv", "r").readlines();
    normals = np.zeros((W,H,3));
    x = 0
    for l1, l2, l3 in zip(f1, f2, f3):
        y = 0
        for val1, val2, val3 in zip(l1.split(","), l2.split(","), l3.split(",")):
            normals[x, y, 0] = float(val1);
            normals[x, y, 1] = float(val2);
            normals[x, y, 2] = float(val3);
            if math.isnan(normals[x, y, 0]):
                normals[x, y, 0] = 0
                normals[x, y, 1] = 0
                normals[x, y, 2] = 1
            y+=1;
        x+=1;

    assert(x == W);
    assert(y == H);
    print normals[-20:,-20:,:]

    return normals;

def get_normals(images, lights):
    pass
