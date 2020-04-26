# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for custom user ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path

import tensorflow as tf
#import matplotlib.pyplot as plt
import numpy as np
import mitsuba_vtx_grad
import sys
import json
import load_normals
import datalib as dataio

from PIL import Image

from shutil import copyfile

# Load operators
dirname = os.path.dirname(os.path.realpath(__file__))
mitsuba = tf.load_op_library(dirname + '/../mitsuba_v2_vtx_op.so')

def target_transform(arr):
    return arr + np.array([0.2, 0.4, 0])

def random_transform(arr):
    tarr = arr + (np.random.random(arr.shape) * 0.3)
    tarr = tarr / np.linalg.norm(arr, axis=1).reshape((arr.shape[0], 1))
    return tarr

TRANSFORMS = {
    "uniform" : target_transform,
    "random" : random_transform
}

def variable_sample_count(n, error, params):
    for stop, samples in zip(params["stops"], params["samples"]):
        if n < stop:
            return samples

SAMPLERS = {
    "variable" : variable_sample_count
}

def renderImage(normals, lights, depth, samples):
    with tf.Session() as sess:
        ref = tf.concat([ mitsuba.mitsuba(parameter_map=tf.constant(normals, dtype=tf.float32), params=p, depth=tf.constant(float(depth), dtype=tf.float32), samples=tf.constant(float(samples), dtype=tf.float32),
                              instances=tf.constant(1.0), unitindex=idx) for idx, p in enumerate(lights) ], axis=2)
        refimg = ref.eval()
        return refimg

class MitsubaEstimator:

    def estimate(self, parameters):

        depthact = tf.constant(parameters["estimator"]["depth"], dtype=tf.float32)
        samplecount = parameters["estimator"]["samples"]
        _normals = parameters["initialization"]["data"]
        _targetnormals = parameters["target"]["normals"]
        refimg = parameters["target"]["data"]
        lights = parameters["lights"]["data"]
        optimizerParams = parameters["estimator"]["optimizer"]
        maxIterations = parameters["estimator"]["iterations"]

        sample_func = SAMPLERS[parameters["estimator"]["samples"]["type"]]
        sample_func_parameters = parameters["estimator"]["samples"]

        # Transfer active mesh to temporary mesh.
        #copyfile(srcmesh, "/tmp/mts_srcmesh.ply")

        #lightvals = open(lights).readlines()
        # Also create a file describing the exact lights used
        #lightsfile = open("light.lt", "w")
        #lightsfile.writelines(lights)

        print("Number of lights: ", len(lights))

        params = [ tf.constant(l, dtype=tf.float32) for l in lights ]

        targetspecs = tf.constant([0.1, 0.1, 0.1])
    
        targetnormals = tf.constant(_targetnormals, dtype=tf.float32)
        normals = tf.Variable(_normals, dtype=tf.float32)

        depthinfty = tf.constant(-1.0)

        scountEst = tf.Variable(sample_func(0, 0, sample_func_parameters), dtype=tf.float32)
        oneInstance = tf.constant(1.0)

        print("Building graph...")
        I = tf.concat([ mitsuba.mitsuba(parameter_map=normals, params=p, depth=depthact, samples=scountEst,
                        instances=oneInstance, unitindex=idx) for idx, p in enumerate(params) ], axis=2)

        for k in range(len(lights)):
            Image.fromarray(((refimg[:,:,k] / 10.0) * 255.0).astype(np.uint8)).save('reference-image-' + format(k).zfill(2) + '.png')
            dataio.writeNumpyData(refimg[:,:,k], 'reference-image-' + format(k).zfill(2))

        L = tf.reduce_sum(tf.pow(I - refimg,2))

        optimizer = None
        if optimizerParams["type"] == "sgd":
            optimizer = tf.train.GradientDescentOptimizer(float(optimizerParams["params"]))
        elif optimizerParams["type"] == "adam":
            optimizer = tf.train.AdamOptimizer(float(optimizerParams["params"]))
        elif optimizerParams["type"] == "sgd-momentum":
            optimizer = tf.train.MomentumOptimizer(float(optimizerParams["lr"]), float(optimizerParams["momentum"]), use_nesterov=False)
        elif optimizerParams["type"] == "sgd-nesterov":
            optimizer = tf.train.MomentumOptimizer(float(optimizerParams["lr"]), float(optimizerParams["momentum"]), use_nesterov=True)
        else:
            raise ValueError("Optimizer type '" + optimizerParams["type"] + "' not recognized")

        train = optimizer.minimize(L, var_list=[normals])

        normalized = normals / tf.sqrt(tf.reduce_sum(tf.square(normals), 1, keep_dims=True))

        projectN = normals.assign(normalized)

        gerror = tf.reduce_sum(tf.pow(normals - targetnormals, 2))

        normalized_error = tf.abs(I - refimg) / refimg

        nerror = tf.abs(normals - targetnormals)

        As = []
        Ws = []
        Es = []
        print("####$$####")
        with tf.Session() as sess:
            init = tf.initialize_all_variables()
            sess.run(init)
            for i in range(0, maxIterations):

                sess.run(scountEst.assign(sample_func(i, Es[:-1], sample_func_parameters)))
                print("Raw prior normals ")
                print(sess.run(normals)[12359])
                (_, Lval, ne) = sess.run((train, L, normalized_error))
                copyfile("/tmp/mts_mesh_copy_1.ply", "mesh-modded-normal-" + format(i).zfill(4) + ".ply")

                # Copy over gradient meshes created by the tensorflow gradient ops
                for k in range(len(lights)):
                    copyfile(
                            "/tmp/mts_mesh_gradients-" + format(k) + ".ply",
                            "mesh-gradients-" + format(i).zfill(4) + "-img" + format(k).zfill(2) + ".ply")

                print("Raw updated normals ")
                print(sess.run(normals)[12359])

                # Run normal reprojection
                sess.run(projectN)

                print("Raw projected normals")
                print(sess.run(normals)[12359])

                for k in range(ne.shape[2]):
                    if i > 9999 or k > 99:
                        print("[WARNING] cannot output error output for greater than 10000 iterations or 100 lights")
                    else:
                        Image.fromarray(((ne[:,:,k] / 10.0) * 255.0).astype(np.uint8)).save('normalized-error-' + format(i).zfill(4) + '-img-' + format(k).zfill(2) + '.png')
                        dataio.writeNumpyData(ne[:,:,k], 'normalized-error-' + format(i).zfill(4) + '-img-' + format(k).zfill(2))

                # Compute normals error.
                Nval = sess.run(gerror)
                absNormals = sess.run(nerror)

                print("NERROR: ", Nval, file=sys.stderr)
                print("IERROR: ", Lval, file=sys.stderr)

                As.append(Nval.tolist())
                Es.append(Lval.tolist())

                # Write values to an errors file.
                efile = open("errors.json", "w")
                json.dump({"nerrors":As, "ierrors":Es}, efile)
                efile.close()

        print("####$$####")

def loadHDSImage(filename):
    data = open(filename, "r").read()
    (width, height, channels) = struct.unpack('iii', data[0:12])

    data = data[12:]

    npdata = np.zeros((width,height,channels), dtype=tf.float32)
    for x in range(width):
        for y in range(height):
            for z in range(channels):
                npdata[x,y,z] = struct.unpack('f', data[x*y*z*4 : x*y*z*4 + 4])
    return npdata

def prepareParameters(params, directory = "."):
    # Load lights.
    if ("lights" in params) and ("file" in params["lights"]):
        lightlines = open(directory + "/" + params["lights"]["file"], "r").readlines()
        lights = [np.array([float(l.split(" ")[1]), float(l.split(" ")[2]), float(l.split(" ")[3])]) for l in lightlines]
        params["lights"]["data"] = lights
        copyfile(directory + "/" + params["lights"]["file"], "lights.lt")

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
        #copyfile(originalfile, "/src/mts_srcmesh.ply")
        copyfile(originalfile, "initialization.ply")

    # Render image if necessary.
    if ("target" in params) and (params["target"]["type"] == "render") and ("mesh" in params["target"]):
        # Render
        print("Target: Rendering target from mesh")
        originalfile = directory + "/" + params["target"]["mesh"]
        copyfile(originalfile, "/tmp/mts_srcmesh.ply")

        normals = load_normals.load_normals(originalfile)
        params["target"]["normals"] = normals
        
        params["target"]["data"] = renderImage(
                        normals,
                        params["lights"]["data"],
                        params["target"]["depth"],
                        params["target"]["samples"])

    elif ("target" in params) and (params["target"]["type"] == "file") and ("files" in params["target"]):
        print("Target: Loading target images from files.")
        alldata = []
        for f in params["target"]["files"]:
            fname = directory + "/" + f
            data = loadHDSImage(fname)
            alldata.append(data.tolist())
        params["target"]["data"] = np.array(alldata)

    # Set the initialization back for the main rendering.
    copyfile("initialization.ply", "/tmp/mts_srcmesh.ply")
    
    return params

if __name__ == '__main__':
    testparams = json.load(open(sys.argv[1], "r"))
    # Copy initial parameters file
    copyfile(sys.argv[1], "parameters.json")

    directory = os.path.dirname(sys.argv[1])
    parameters = prepareParameters(testparams, directory)
    # Run a test
    MitsubaEstimator().estimate(parameters)