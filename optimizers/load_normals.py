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

    return normals

def load_vertices(filename):
    plyfile = open(filename, "r");
    plines = plyfile.readlines();
    print("SIZE: ",len(plines));

    vertexCount = 0
    start = False;
    vertices = [];
    curr = 0
    leftover = 0
    for line in plines:
        
        if line.startswith("element vertex"):
            vertexCount = int(line.split(" ")[2]);
            vertices = np.zeros((vertexCount, 3));
            leftover = vertexCount;

        if start:
            vertices[curr, 0] = float(line.split(" ")[0]);
            vertices[curr, 1] = float(line.split(" ")[1]);
            vertices[curr, 2] = float(line.split(" ")[2]);
            curr = curr + 1
            leftover = leftover - 1
            if leftover == 0:
                break;
        
        if line == "end_header\n":
            print("Found end header")
            start = True;

    return vertices


def emplace_normals_as_colors(filename, outfile, values, asfloat=False, asnormals=False):
    plyfile = open(filename, "r")
    outplyfile = open(outfile, "w")

    plines = plyfile.readlines()

    vertexCount = 0
    start = False
    curr = 0
    leftover = 0
    idx = 0
    for line in plines:
        if " nx\n" in line:
            if not asnormals:
                if not asfloat:
                    outplyfile.write("property uchar red\n")
                else:
                    outplyfile.write("property float red\n")
            else:
                outplyfile.write("property float nx\n")
        elif " ny\n" in line:
            if not asnormals:
                if not asfloat:
                    outplyfile.write("property uchar green\n")
                else:
                    outplyfile.write("property float green\n")
            else:
                outplyfile.write("property float ny\n")
        elif " nz\n" in line:
            if not asnormals:
                if not asfloat:
                    outplyfile.write("property uchar blue\n")
                else:
                    outplyfile.write("property float blue\n")
            else:
                outplyfile.write("property float nz\n")

        elif line.startswith("element vertex"):
            vertexCount = int(line.split(" ")[2])
            vertices = np.zeros((vertexCount, 3))
            leftover = vertexCount
            outplyfile.write(line)
        elif start and leftover > 0:
            vx = line.split(" ")[0]
            vy = line.split(" ")[1]
            vz = line.split(" ")[2]
            cx = line.split(" ")[3]
            cy = line.split(" ")[4]
            cz = line.split(" ")[5]

            if not asfloat:
                newline = vx + " " + vy + " " + vz + " " + format(min(int(values[idx, 0] * 255), 255)) + " " + format(min(int(values[idx, 1] * 255), 255)) + " " + format(min(int(values[idx, 2] * 255), 255))
                outplyfile.write(newline + "\n")
            else:
                newline = vx + " " + vy + " " + vz + " " + format(values[idx, 0]) + " " + format(values[idx, 1]) + " " + format(values[idx, 2])
                #print("Writing: ", newline)
                outplyfile.write(newline + "\n")

            idx += 1

            curr = curr + 1
            leftover = leftover - 1
            #if leftover == 0:
            #    break

        elif line == "end_header\n":
            start = True
            outplyfile.write("end_header\n")

        else:
            outplyfile.write(line)

#print(load_normals("/tmp/mts_srcmesh.ply"))
