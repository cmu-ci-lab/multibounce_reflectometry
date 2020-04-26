
import tensorflow as tf
import numpy as np
import sys
import json
import load_normals
import datalib as dataio
import json

from PIL import Image
import os
from shutil import copyfile
import hdsutils
import merl_io

# Load operators
dirname = os.path.dirname(os.path.realpath(__file__))

def renderImage(normals, lights, bsdf, depth, samples):
    mitsuba = tf.load_op_library(dirname + '/../qdispatch/mitsuba_v2_vtx_op.so')
    with tf.Session() as sess:
        ref = tf.concat([ mitsuba.mitsuba(parameter_map=tf.constant(normals, dtype=tf.float32), params=p, bsdf=tf.constant(bsdf, dtype=tf.float32), depth=tf.constant(float(depth), dtype=tf.float32), samples=tf.constant(float(samples), dtype=tf.float32),
                              instances=tf.constant(1.0), unitindex=idx) for idx, p in enumerate(lights) ], axis=2)
        refimg = ref.eval()
        return refimg

def loadParameters(filename):
    params = json.load(open(filename, "r"))
    directory = os.path.dirname(filename)

    if "normals-config" in params:
        nparams = json.load(open(directory + "/" + params["normals-config"], "r"))
        for item in nparams:
            if item in params:
                print("WARNING: '" + item + "' ignored in normals config. Already exists in parent")
            else:
                params[item] = nparams[item]

    return params, directory

def prepareParameters(params, directory = "."):
    # Load lights.
    if ("lights" in params) and ("file" in params["lights"]):
        lightlines = open(directory + "/" + params["lights"]["file"], "r").readlines()
        lights = [np.array([float(l.split(" ")[1]), float(l.split(" ")[2]), float(l.split(" ")[3])]) for l in lightlines]
        params["lights"]["data"] = lights
        #copyfile(directory + "/" + params["lights"]["file"], "lights.lt")

    if ("lights" in params) and ("intensity-file" in params["lights"]):
        intensitylines = open(directory + "/" + params["lights"]["intensity-file"], "r").readlines()
        intensities = [np.array(float(i)) for i in intensitylines]
        params["lights"]["intensity-data"] = intensities
        #copyfile(directory + "/" + params["lights"]["file"], "lights.lt")

    if ("lights" in params) and ("intensity-file" not in params["lights"]):
        params["lights"]["intensity-data"] = np.array([10.0] * len(params["lights"]["data"]))
        #copyfile(directory + "/" + params["lights"]["file"], "lights.lt")

    # Load normals
    if ("initialization" in params) and ("file" in params["initialization"]):
        print("Initialization: Loading initialization from mesh file.")
        # If 'file' exists, then load from PLY file.
        params["initialization"]["data"] = load_normals.load_normals(directory + "/" + params["initialization"]["file"])
        # Make a copy.
        copyfile(directory + "/" + params["initialization"]["file"], "initialization.ply")

    elif ("initialization" in params) and ("derived" in params["initialization"]) and (params["target"]["type"] == "render"):
        # 'derived' = the initialization is prepared by manipulating the original normals.
        print("Initialization: Deriving initialization by perturbing target")
        originalfile = directory + "/" + params["target"]["mesh"]
        normals = load_normals.load_normals(originalfile)
        params["initialization"]["data"] = TRANSFORMS[params["initialization"]["derived"]](normals)

        copyfile(originalfile, "initialization.ply")

    if "tabular-bsdf" in params["bsdf-estimator"] and "subspace-regularization" in params["bsdf-estimator"]["tabular-bsdf"]:
        params["bsdf-estimator"]["tabular-bsdf"]["subspace-regularization"]["subspace"] = np.load(directory + "/" + params["bsdf-estimator"]["tabular-bsdf"]["subspace-regularization"]["subspace"])


    # Render image if necessary.
    if ("target" in params) and (params["target"]["type"] == "render") and ("mesh" in params["target"]):
        # Render
        print("Target: Rendering target from mesh")
        originalfile = directory + "/" + params["target"]["mesh"]
        copyfile(originalfile, "/tmp/mts_srcmesh.ply")

        normals = load_normals.load_normals(originalfile)
        params["target"]["normals"] = normals

        if "zero-padding" in params and type(params["zero-padding"]) is int:
            zeroPadding = parameters["zero-padding"]
        elif "zero-padding" in params and type(params["zero-padding"]) is unicode:
            zeroPadding = json.load(open(params["zero-padding"], "r"))[0]
        else:
            zeroPadding = 0

        if type(params['original']['hyper-parameters']) == dict:
            hyperparams = np.array([ params["original"]["hyper-parameters"][k] for k in params["hyper-parameter-list"] ] + [0] * zeroPadding)
        else:
            hyperparams = np.concatenate([np.load(directory + "/" + params["original"]["hyper-parameters"]), np.zeros((zeroPadding,))], axis=0)

        params["target"]["data"] = renderImage(
                        normals,
                        [ l for l in params["lights"]["data"] ],
                        hyperparams,
                        params["target"]["depth"],
                        params["target"]["samples"])

        if not os.path.exists("targets"):
            os.mkdir("targets")
        if not os.path.exists("targets/png"):
            os.mkdir("targets/png")
        if not os.path.exists("targets/npy"):
            os.mkdir("targets/npy")

        for k in range(len(params["lights"]["data"])):
            Image.fromarray(((params["target"]["data"][:,:,k] / 10.0) * 255.0).astype(np.uint8)).save('targets/png/target-image-' + format(k).zfill(2) + '.png')
            dataio.writeNumpyData(params["target"]["data"][:,:,k], 'targets/npy/target-image-' + format(k).zfill(2))

    elif ("target" in params) and (params["target"]["type"] == "file") and ("files" in params["target"]):
        print("Target: Loading target images from files.")
        alldata = []
        for f in params["target"]["files"]:
            fname = directory + "/" + f
            data = hdsutils.loadHDSImage(fname)
            alldata.append(data.tolist())

        params["target"]["data"] = (np.array(alldata)[:,:,:,0]).transpose([1,2,0])

        print(params["target"]["data"].shape)

        originalfile = directory + "/" + params["target"]["refmesh"]
        normals = load_normals.load_normals(originalfile)
        params["target"]["normals"] = normals

        if not os.path.exists("targets"):
            os.mkdir("targets")
        if not os.path.exists("targets/png"):
            os.mkdir("targets/png")
        if not os.path.exists("targets/npy"):
            os.mkdir("targets/npy")

        for k in range(len(params["lights"]["data"])):
            Image.fromarray(((params["target"]["data"][:,:,k] / 10.0) * 255.0).astype(np.uint8)).save('targets/png/target-image-' + format(k).zfill(2) + '.png')
            dataio.writeNumpyData(params["target"]["data"][:,:,k], 'targets/npy/target-image-' + format(k).zfill(2))
    elif ("target" in params) and (params["target"]["type"] == "ext-render") and ("mesh" in params["target"]) and ("scene" in params["target"]):
        # Render
        print("Target: Rendering target from mesh")
        originalfile = directory + "/" + params["target"]["mesh"]
        copyfile(originalfile, "/tmp/mts_mesh_intensity_slot_0.ply")

        #normals = load_normals.load_normals(originalfile)
        #params["target"]["normals"] = normals

        #hyperparams = np.array([ params["original"]["hyper-parameters"][k] for k in params["hyper-parameter-list"] ])

        #params["target"]["data"] = renderImage(
        #                normals,
        #                [ l for l in params["lights"]["data"] ],
        #                hyperparams,
        #                params["target"]["depth"],
        #                params["target"]["samples"])

        hstring = ""

        if "tabular-bsdf" in params["target"]:
            targetTablarBSDF = np.load(directory + "/" + params["target"]["tabular-bsdf"])
            merl_io.merl_write("/tmp/tabular-bsdf-0.binary", targetTablarBSDF)

        if "hyper-parameters" in params["original"]:
            for hparam in params["hyper-parameter-list"]:
                hstring += " -D" + hparam + "=" + format(params["original"]["hyper-parameters"][hparam])
        elif "hyper-parameters" in params["target"]:
            hvalues = np.load(directory + "/" + params["target"]["hyper-parameters"])
            for hparam, hval in zip(params["hyper-parameter-list"], hvalues):
                hstring += " -D" + hparam + "=" + format(hval)

        targets = []
        if not os.path.exists("targets"):
            os.mkdir("targets")
        
        width = 256
        height = 256
        if params["target"]["width"]:
            width = params["target"]["width"]
        if params["target"]["height"]:
            height = params["target"]["height"]

        # If spatial adaptive sampling is enabled, output a sample texture.
        if "spatial-adaptive" in params["bsdf-estimator"]["samples"] and params["bsdf-estimator"]["samples"]["spatial-adaptive"]:
            data = np.ones((width, height, 1))
            hdsutils.writeHDSImage("/tmp/sampler-0.hds", width, height, 1, data)

        for i, l in enumerate(params["lights"]["data"]):
            command = "mitsuba " + directory + "/" + params["target"]["scene"] + " -o " + directory + "/targets/" + format(i).zfill(2) + ".hds -DlightX=" + format(l[0]) + " -DlightY=" + format(l[1]) + " -DlightZ=" + format(l[2]) + " -Ddepth=" + format(params["target"]["depth"]) + " -DsampleCount=" + format(params["target"]["samples"]) + " -Dwidth=" + format(width) + " -Dheight=" + format(height) + " -Dirradiance=" + format(params["lights"]["intensity-data"][i]) + " -DmeshSlot=0 " + hstring
            print(command)
            os.system(command)
            targets.append(hdsutils.loadHDSImage(directory + "/" + "targets/" + format(i).zfill(2) + ".hds"))

        print("Targets shape: ", np.array(targets).shape)

        params["target"]["data"] = np.array(targets)[:,:,:,0].transpose([1,2,0])

        originalfile = directory + "/" + params["target"]["mesh"]
        normals = load_normals.load_normals(originalfile)
        params["target"]["normals"] = normals

        if not os.path.exists("targets/png"):
            os.mkdir("targets/png")
        if not os.path.exists("targets/npy"):
            os.mkdir("targets/npy")

        for k in range(len(params["lights"]["data"])):
            Image.fromarray(((params["target"]["data"][:,:,k] / 10.0) * 255.0).astype(np.uint8)).save('targets/png/target-image-' + format(k).zfill(2) + '.png')
            dataio.writeNumpyData(params["target"]["data"][:,:,k], 'targets/npy/target-image-' + format(k).zfill(2))

    elif ("target" in params) and (params["target"]["type"] == "npy") and ("file" in params["target"]):
        # Load
        params["target"]["data"] = np.load(directory + "/" + params["target"]["file"])
        params["target"]["normals"] = np.load(directory + "/" + params["target"]["normals-file"])

    elif "target" in params:
        print("Unrecognized target mode ", params["target"])

    copyfile("initialization.ply", "/tmp/mts_srcmesh.ply")
    return params
