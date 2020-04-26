# Restarting code.
from shutil import copytree

import os

def copyall(srcfolder, destfolder, spath):
    if os.path.exists(srcfolder + "/" + spath):
        if not os.path.exists(destfolder + "/" + spath):
            os.makedirs(os.path.dirname(destfolder + "/" + spath))

        copytree(srcfolder + "/" + spath, destfolder + "/" + spath)

# Copies a source directory to the target starting from a specific source iteration.
def midcopy(srcfolder, destfolder, srciter):

    copyall(srcfolder, destfolder, "/targets")
    copyall(srcfolder, destfolder, "/inputs")
    copyall(srcfolder, destfolder, "/logs")

    for idx, itext in enumerate(srciter):
        sitext = format(idx).zfill(2)

        copyall(srcfolder, destfolder, "/meshes/totalgradients/" + sitext)
        copyall(srcfolder, destfolder, "/meshes/gradients/" + sitext)
        copyall(srcfolder, destfolder, "/meshes/normaldeltas/" + sitext)
        copyall(srcfolder, destfolder, "/meshes/normals/" + sitext)
        copyall(srcfolder, destfolder, "/meshes/remeshed/" + sitext + ".ply")

        copyall(srcfolder, destfolder, "/renders/gradients/" + sitext)
        copyall(srcfolder, destfolder, "/renders/normals/" + sitext)
        copyall(srcfolder, destfolder, "/renders/totalgradients/" + sitext)
        copyall(srcfolder, destfolder, "/renders/normaldeltas/" + sitext)

        copyall(srcfolder, destfolder, "/images/current/npy/" + sitext)
        copyall(srcfolder, destfolder, "/images/difference-errors/npy/" + sitext)
        copyall(srcfolder, destfolder, "/images/normalized-absolute-errors/png/" + sitext)
        copyall(srcfolder, destfolder, "/images/normalized-absolute-errors/npy/" + sitext)
        copyall(srcfolder, destfolder, "/images/normalized-difference-errors/npy/" + sitext)

        copyall(srcfolder, destfolder, "/images/errors/errors-" + sitext + ".json")
        copyall(srcfolder, destfolder, "/images/errors/bsdf-errors-" + sitext + ".json")

    pass

def allcopy(srcfolder, destfolder):
    copyall(srcfolder, destfolder, "/")