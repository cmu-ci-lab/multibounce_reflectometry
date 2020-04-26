import sys
import os

directory = sys.argv[1]
superindices = os.listdir(directory + "/meshes/gradients/")

def splitPolarity(plyfile, negplyfile, posplyfile):

    print("PROCESSING: " + plyfile)
    f = open(plyfile, "r")

    nof = open(negplyfile, "w")
    pof = open(posplyfile, "w")
    for line in f.readlines():
        if " nx\n" in line:
            nof.write("property " + dataType + " red\n")
            pof.write("property " + dataType + " red\n")
        elif " ny\n" in line:
            nof.write("property " + dataType + " green\n")
            pof.write("property " + dataType + " green\n")
        elif " nz\n" in line:
            nof.write("property " + dataType + " blue\n")
            pof.write("property " + dataType + " blue\n")
        elif len(line.split(" ")) == 6:
            vx = line.split(" ")[0]
            vy = line.split(" ")[1]
            vz = line.split(" ")[2]

            gx = 0
            gy = 0
            gz = 0
            if not useFloat:
                gx = min(max(int(float(line.split(" ")[3]) * 0.5 * 256), -255), 255)
                gy = min(max(int(float(line.split(" ")[4]) * 0.5 * 256), -255), 255)
                gz = min(max(int(float(line.split(" ")[5]) * 0.5 * 256), -255), 255)
            else:
                gx = float(line.split(" ")[3])
                gy = float(line.split(" ")[4])
                gz = float(line.split(" ")[5])

            nof.write(vx + " " + vy + " " + vz + " ")
            pof.write(vx + " " + vy + " " + vz + " ")

            if gx > 0:
                pof.write(format(gx) + " ")
                nof.write("0 ")
            else:
                nof.write(format(-gx) + " ")
                pof.write("0 ")

            if gy > 0:
                pof.write(format(gy) + " ")
                nof.write("0 ")
            else:
                nof.write(format(-gy) + " ")
                pof.write("0 ")

            if gz > 0:
                pof.write(format(gz) + "\n")
                nof.write("0\n")
            else:
                nof.write(format(-gz) + "\n")
                pof.write("0\n")
        else:
            nof.write(line)
            pof.write(line)

def makePlyNames(plyfile):
    negplyfile = plyfile.replace(".ply",".nply")
    posplyfile = plyfile.replace(".ply",".pply")
    return (negplyfile, posplyfile)

useFloat = True
#if "MTSTF_DEBUG_USE_FLOAT" in os.environ:
#    print("Using 96-bit FP color")
#    useFloat = True

dataType = "uchar"
if useFloat:
    dataType = "float"

for superindex in superindices:
    meshes = directory + '/meshes/gradients/' + superindex
    for mesh in os.listdir(meshes):
        plyfile = directory + '/meshes/gradients/' + superindex + '/' + mesh
        if plyfile.endswith(".ply.ply"):
            continue
        if not plyfile.endswith(".ply"):
            continue

        negplyfile = directory + '/meshes/gradients/' + superindex + '/' + mesh.replace(".ply",".nply")
        posplyfile = directory + '/meshes/gradients/' + superindex + '/' + mesh.replace(".ply",".pply")
        splitPolarity(plyfile, negplyfile, posplyfile)

    meshes = directory + '/meshes/normaldeltas/' + superindex
    for mesh in os.listdir(meshes):
        plyfile = directory + '/meshes/normaldeltas/' + superindex + '/' + mesh
        if plyfile.endswith(".ply.ply"):
            continue
        if not plyfile.endswith(".ply"):
            continue

        negplyfile = directory + '/meshes/normaldeltas/' + superindex + '/' + mesh.replace(".ply",".nply")
        posplyfile = directory + '/meshes/normaldeltas/' + superindex + '/' + mesh.replace(".ply",".pply")
        splitPolarity(plyfile, negplyfile, posplyfile)