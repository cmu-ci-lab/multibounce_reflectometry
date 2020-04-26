import os
import struct
import sys
import numpy as np
import Image
import math
f = open(sys.argv[1], "rb");
ks = struct.unpack("iii", f.read(12));
print(ks);
#f.close();


ifs = [];


imgs = [];
for i in range(8):
    imgs.append(np.zeros((ks[1],ks[2],3)));


#print(ifs);
for k in range(ks[0]):
    upc = struct.unpack("iiifff", f.read(24));
    #if np.abs(upc[5]) > 100 or np.abs(upc[4]) > 100 or np.abs(upc[3]) > 100:
    #    print(upc)
    
    if np.isinf(upc[5]) > 100 or np.isinf(upc[4]) > 100 or np.isinf(upc[3]) > 100:
        print(upc)
    
    imgs[upc[2]][upc[1], upc[0], 0] = upc[3];
    imgs[upc[2]][upc[1], upc[0], 1] = upc[4];
    imgs[upc[2]][upc[1], upc[0], 2] = upc[5];

    if(upc[1] == 114):
       print(upc);
    
    ifs.append(upc);

for i in range(8):
    Image.fromarray((np.power(np.abs(imgs[i]), 0.6) * 0.04 * 255.0).astype(np.uint8)).save('gradient-n' + format(i) + '-d-1.png');


#print(ifs);
