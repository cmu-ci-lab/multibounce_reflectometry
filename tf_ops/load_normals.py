import numpy as np

def load_normals(filename):
    plyfile = open(filename, "r");
    plines = plyfile.readlines();
    print("SIZE: ",len(plines));
    
    vertexCount = 0
    start = False;
    normals = [];
    curr = 0
    leftover = 0
    for line in plines:
        
        if line.startswith("element vertex"):
            vertexCount = int(line.split(" ")[2]);
            normals = np.zeros((vertexCount, 3));
            leftover = vertexCount;

        if start:
            normals[curr, 0] = float(line.split(" ")[3]);
            normals[curr, 1] = float(line.split(" ")[4]);
            normals[curr, 2] = float(line.split(" ")[5]);
            curr = curr + 1
            leftover = leftover - 1
            if leftover == 0:
                break;
        
        if line == "end_header\n":
            print("Found end header")
            start = True;


    return normals;

#print(load_normals("/tmp/mts_srcmesh.ply"))
