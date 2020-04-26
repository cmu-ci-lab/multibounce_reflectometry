import numpy as np

def load_normals(filename):
    plyfile = open(filename, "r");
    plines = plyfile.readlines();
    print("[LoadNormals] SIZE: ",len(plines));
    
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

def emplace_normals_as_colors(filename, outfile, values):
    plyfile = open(filename, "r");
    outplyfile = open(outfile, "w");

    plines = plyfile.readlines();
    print("[LoadNormals] SIZE: ",len(plines));
    
    vertexCount = 0
    start = False;
    curr = 0
    leftover = 0
    idx = 0
    for line in plines:
        if " nx " in line:
            outplyfile.write("property float red")
        elif " nx " in line:
            outplyfile.write("property float green")
        elif " nx " in line:
            outplyfile.write("property float blue")
        elif start:
            print line
            vx = line.split(" ")[0]
            vy = line.split(" ")[1]
            vz = line.split(" ")[2]
            cx = line.split(" ")[3]
            cy = line.split(" ")[4]
            cz = line.split(" ")[5]

            newline = vx + " " + " " + vy + " " + vz + " " + format(values[idx, 0]) + " " + format(values[idx, 1]) + " " + format(values[idx, 2])
            outplyfile.write(newline + "\n")
            idx += 1

            curr = curr + 1
            leftover = leftover - 1
            if leftover == 0:
                break

        elif line == "end_header\n":
            start = True
            outplyfile.write("end_header\n")

        else:
            outplyfile.write(line + "\n")