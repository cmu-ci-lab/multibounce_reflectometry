import sys
import os

from shutil import copyfile
from hdsutils import loadHDSImage

import numpy as np

def create_dir(dirname):
    if not os.path.exists(dirname):
        os.mkdir(dirname)

def makeRenderNames(meshfile):
    return (
        meshfile.replace("ply", "hds").replace(".nhds", ".n.hds").replace(".phds", ".p.hds"),
        meshfile.replace("ply", "npy").replace(".nnpy", ".n.npy").replace(".pnpy", ".p.npy"))

def renderMesh(meshname, targethdsfile, targetnpyfile, scenename, lazy=True, W=256, H=256):
    #print("Target file: " + targethdsfile)
    if (not os.path.exists(targethdsfile)) or (not lazy):
        copyfile(meshname, "/tmp/mts_mesh_copy_4.ply")
        os.system("mitsuba " + scenename + " -o \"" + targethdsfile + "\" -Dwidth=" + format(W) + " -Dheight=" + format(H))

    if (not os.path.exists(targetnpyfile) and os.path.exists(targethdsfile)) or (not lazy):
        narr = loadHDSImage(targethdsfile)
        np.save(targetnpyfile, narr)

if __name__ == '__main__':
    directory = sys.argv[1]
    #scene_name = sys.argv[2]
    normals_scene_name = directory + "/inputs/scenes/normals-scene.xml"
    colors_scene_name = directory + "/inputs/scenes/colors-scene.xml"

    superindices = os.listdir(directory + "/meshes/normals")
    create_dir(directory + "/renders")
    create_dir(directory + "/renders/normals")
    create_dir(directory + "/renders/gradients")
    create_dir(directory + "/renders/normaldeltas")

    categories = [("normals", normals_scene_name), ("gradients", colors_scene_name), ("normaldeltas", colors_scene_name)]
    for meshtype, scene_name in categories:
        for superindex in superindices:
            create_dir(directory + "/renders/" + meshtype + "/" + superindex)
            files = os.listdir(directory + "/meshes/" + meshtype + "/" + superindex)
            for fname in files:
                print(directory + "/renders/" + meshtype + "/" + superindex + "/" + fname)
                if fname.endswith("ply") and not (".ply.ply" in fname):
                    fullmeshname = directory + "/meshes/" + meshtype + "/" + superindex + "/" + fname
                    targethdsfile = directory + "/renders/" + meshtype + "/" + superindex + "/" + fname.replace("ply", "hds").replace(".nhds", ".n.hds").replace(".phds", ".p.hds")
                    targetnpyfile = directory + "/renders/" + meshtype + "/" + superindex + "/" + fname.replace("ply", "npy").replace(".nnpy", ".n.npy").replace(".pnpy", ".p.npy")
                    if not os.path.exists(targethdsfile):
                        copyfile(fullmeshname, "/tmp/mts_mesh_copy_3.ply")
                        os.system("mitsuba " + scene_name + " -o \"" + targethdsfile + "\"")

                    if not os.path.exists(targetnpyfile) and os.path.exists(targethdsfile):
                        narr = loadHDSImage(targethdsfile)
                        np.save(targetnpyfile, narr)

