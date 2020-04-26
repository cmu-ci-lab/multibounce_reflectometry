import tensorflow as tf
import numpy as np
import mitsuba_vtx_grad
import sys
import json
import load_normals
import datalib as dataio
import os
import splitpolarity
import rendernormals
import scipy.ndimage
import hdsutils
import cv2
import bivariate_proj

# A special optimizer that weighs vectors uniformly across the 3 dimensions.
# This is necessary to avoid changes to the vector directions.
from grouped_adam import GroupedAdamOptimizer, SetAdamOptimizer

# Another special optimizer for sigma-1 non-zero weights training.
from exponentiated_adam import ExpGradSetAdamOptimizer

# A powerful MCMC optimizer
from mala_optimizer import MALANormalOptimizer

from PIL import Image

from shutil import copyfile

from remesher.algorithms.poisson.integrator import integrate, normalsFromField

from adjacency import MeshAdjacencyBuilder

import matplotlib.pyplot as plt

# Load operators
dirname = os.path.dirname(os.path.realpath(__file__))
mitsuba = tf.load_op_library(dirname + '/../qdispatch/mitsuba_v2_vtx_stacked_op.so')

def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

def target_transform(arr):
    return arr + np.array([0.2, 0.4, 0])

def random_transform(arr):
    tarr = arr + (np.random.random(arr.shape) * 0.3)
    tarr = tarr / np.linalg.norm(arr, axis=1).reshape((arr.shape[0], 1))
    return tarr

def getDecay(optimizerParams):
    decay = 1.0
    if "decay" in optimizerParams:
        decay = optimizerParams["decay"]
    return decay

def getBetas(optimizerParams):
    beta1 = 0.9
    beta2 = 0.999
    if "beta1" in optimizerParams:
        beta1 = optimizerParams["beta1"]
    if "beta2" in optimizerParams:
        beta2 = optimizerParams["beta2"]

    return beta1, beta2

def processOptimizerParameters(optimizerParams, superiteration=0):
    #okeys = optimizerParams.keys()
    for field, tfield in [("params-generator", "params")]:
        if type(optimizerParams[field]) is dict:
            if optimizerParams[field]["type"] == "static":
                optimizerParams[field] = optimizerParams[field]["value"]
            elif optimizerParams[field]["type"] == "list":
                for idx, stop in enumerate(optimizerParams[field]["stops"]):
                    if superiteration < stop:
                        optimizerParams[field] = optimizerParams[field]["values"][idx]
                        break
            elif optimizerParams[field]["type"] == "factor":
                factor = optimizerParams[field]["initial-value"] * (optimizerParams[field]["factor"] ** (superiteration + 1))
                print("FACTOR: ", factor, " ", optimizerParams[field]["factor"], " field: ", field)
                optimizerParams[tfield] = factor
            else:
                print("optimizer param ", field, " invalid type ", optimizerParams[field]["type"])
                sys.exit(1)

    print("Optimizer params at ", superiteration, " ", optimizerParams)
    return optimizerParams

def getOptimizerFromParameters(optimizerParams, mainLength=-1, reprojectionLength=-1, paddedLength=-1, disableSets=False):
    optimizer = None
    if optimizerParams["type"] == "sgd":
        optimizer = tf.train.GradientDescentOptimizer(float(optimizerParams["params"]))
    elif optimizerParams["type"] == "adam":
        beta1, beta2 = getBetas(optimizerParams)
        optimizer = tf.train.AdamOptimizer(float(optimizerParams["params"]), beta1=beta1, beta2=beta2)
    elif optimizerParams["type"] == "grouped-adam":
        beta1, beta2 = getBetas(optimizerParams)
        decay = getDecay(optimizerParams)
        optimizer = GroupedAdamOptimizer(float(optimizerParams["params"]), beta1=beta1, beta2=beta2, decay=decay)
    elif optimizerParams["type"] == "sgd-momentum":
        optimizer = tf.train.MomentumOptimizer(float(optimizerParams["lr"]), float(optimizerParams["momentum"]), use_nesterov=False)
    elif optimizerParams["type"] == "sgd-nesterov":
        optimizer = tf.train.MomentumOptimizer(float(optimizerParams["lr"]), float(optimizerParams["momentum"]), use_nesterov=True)
    elif optimizerParams["type"] == "set-adam":
        beta1, beta2 = getBetas(optimizerParams)
        if "sets" in optimizerParams:
            tvarsets = [ tf.constant(varset, dtype=tf.float32) for varset in optimizerParams["sets"] ]
        else:
            tvarsets = None
        optimizer = SetAdamOptimizer(float(optimizerParams["params"]), beta1=beta1, beta2=beta2, sets=tvarsets)
    elif optimizerParams["type"] == "eg-adam":
        beta1, beta2 = getBetas(optimizerParams)
        if "sets" in optimizerParams and type(optimizerParams["sets"]) is list:
            tvarsets = [ tf.constant(varset, dtype=tf.float32) for varset in optimizerParams["sets"] ]
        elif "sets" in optimizerParams and optimizerParams["sets"] == "main-only":
            tvarsets = [ tf.constant([1] * mainLength + [0] * (paddedLength - mainLength), dtype=tf.float32) ]
            exclusionSet = tf.constant([0] * reprojectionLength + [1] * (mainLength - reprojectionLength) + [0] * (paddedLength - mainLength))
        else:
            tvarsets = None

        if disableSets:
            tvarsets = []

        optimizer = ExpGradSetAdamOptimizer(float(optimizerParams["params"]), beta1=beta1, beta2=beta2, sets=tvarsets, excl=exclusionSet)
    elif optimizerParams["type"] == "mala":
        beta1, beta2 = getBetas(optimizerParams)
        optimizer = MALANormalOptimizer(tau=float(optimizerParams["params"]), beta2=beta2)
    else:
        raise ValueError("Optimizer type '" + optimizerParams["type"] + "' not recognized")

    return optimizer

def computeBSDFSampleWeights(bc_v, bc_m, exps, snr, vals, dictionary=None, mode="SNR"):
    """
        Computes BSDF sampling weights. To be used for BSDF-Adaptive sampling facilitated through the
        "MixtureSampledBSDF" plugin.
    """
    if mode == "SNR":
        epsilon = 1e-3
        # Work only on dictionary elements. Beyond this point, there may be padding elements.
        return snr[:len(dictionary["elements"])] + epsilon
    elif mode == "invAlpha":
        epsilon = 1e-3
        wts = []
        for element in dictionary["elements"]:
            if element["type"] == "roughconductor":
                wts.append(1/(element["alpha"] + epsilon))
            elif element["type"] == "diffuse":
                wts.append(1.0)
            else:
                wts.append(1.0)
        return np.array(wts)
    elif mode == "bsdfWeight":
        return vals[:len(dictionary["elements"])]
    else:
        return None

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

def directSpatialTexture(iter, de, scount, dim):
    # Take absolute value and blur the textures a little.
    ade = np.abs(de)
    bade = np.zeros_like(ade)
    for i in range(ade.shape[2]):
        bade[:,:,i] = scipy.ndimage.filters.gaussian_filter(ade[:,:,i], 4.0)

    # 'normalize' the sampling texture with the L1 metric. (Set mean to 1.)
    nbade = bade / np.mean(bade)

    # Add a minimum value of 4 samples to every pixel.
    snbade = nbade + 4.0/scount

    return snbade, scount

def uniformSpatialTexture(iter, de, scount, dim):
    return np.ones(dim), scount

SPATIAL_SAMPLING_FNS = {
    "direct": directSpatialTexture,
    "uniform": uniformSpatialTexture,
    "inverse": None
}
   
def computeSpatialTexture(iter, de, scount, dim, mode):
    if mode not in SPATIAL_SAMPLING_FNS:
        print "Spatial Texture Mode '", mode, "' not available."
    if iter != 0:
        return SPATIAL_SAMPLING_FNS[mode](iter, de, scount, dim)
    else:
        return SPATIAL_SAMPLING_FNS["uniform"](iter, de, scount, dim)

def enforceIntegrability(vertices, normals, W=256, H=256):
    normalMap = MeshAdjacencyBuilder.buildNormalMap(vertices, normals, width=W, height=H)
    zfield = integrate(normalMap, mean=0.0, zflipped=False)
    #plt.imshow(zfield)
    #plt.show()
    normals = normalsFromField(zfield)
    #plt.imshow(normals)
    #plt.show()
    # Convert back to a list of normals.
    vmap = MeshAdjacencyBuilder.buildVertexMap(vertices, normals, width=W, height=H)
    normals = normals[vmap[:,0], vmap[:,1], :]
    return normals

dirpath = os.path.dirname(__file__)
def optimizeNormals(parameters, superindex=-1, bsdfoverride=None, W=256, H=256, iports=[7554], gports=[7555], indexMap=None, indexValidity=None, bsdfProps={}):
    depthact = tf.constant(parameters["estimator"]["depth"], dtype=tf.float32)
    samplecount = parameters["estimator"]["samples"]
    _normals = parameters["initialization"]["data"]
    _vertices = parameters["initialization"]["vertex-data"]
    _targetnormals = parameters["target"]["normals"]
    refimg = parameters["target"]["data"]
    lights = parameters["lights"]["data"]
    intensities = parameters["lights"]["intensity-data"]
    optimizerParams = parameters["estimator"]["optimizer"]
    maxIterations = parameters["estimator"]["iterations"]
    bsdfIterations = parameters["bsdf-estimator"]["iterations"]
    albedoEnabled = parameters["estimator"]["albedo"]

    errorWeights = tf.constant(parameters["target"]["weights"], dtype=tf.float32)
    bsdfErrorWeights = tf.constant(parameters["target"]["bsdf-weights"], dtype=tf.float32)

    # Zero padding.
    if "zero-padding" in parameters and type(parameters["zero-padding"]) is int:
        try:
            zeroPadding, subDiffs = parameters["zero-padding"]
        except ValueError:
            zeroPadding = parameters["zero-padding"]
            subDiffs = 0

    elif "zero-padding" in parameters and type(parameters["zero-padding"]) is unicode:
        _params = json.load(open(parameters["zero-padding"], "r"))
        try:
            zeroPadding, subDiffs = _params
        except ValueError:
            zeroPadding = _params[0]
            subDiffs = 0
    else:
        zeroPadding = 0
        subDiffs = 0

    if type(parameters["hyper-parameter-list"]) is list:
        hyperList = parameters["hyper-parameter-list"] + ["0"] * zeroPadding
    elif type(parameters["hyper-parameter-list"]) is unicode:
        hyperList = json.load(open(parameters["hyper-parameter-list"], "r")) + ["0"] * zeroPadding
    else:
        hyperList = None
        print("No hyper-parameter-list found in config file. Aborting..")
        sys.exit(1)

    mainLength = len(hyperList) - zeroPadding
    reprojectionLength = mainLength - subDiffs
    paddedLength = len(hyperList)

    bsdfParameters = parameters["bsdf-estimator"]
    tabularBSDFParameters = parameters["bsdf-estimator"]["tabular-bsdf"]
    bsdfAdaptiveSampling = "bsdf-adaptive" in bsdfParameters["samples"] and bsdfParameters["samples"]["bsdf-adaptive"]
    bsdfSpatialSampling = "spatial-adaptive" in bsdfParameters["samples"] and bsdfParameters["samples"]["spatial-adaptive"]
    bsdfOptimizerParams = parameters["bsdf-estimator"]["optimizer"]
    tabularBsdfOptimizerParams = parameters["bsdf-estimator"]["tabular-bsdf"]["optimizer"]
    bsdfHyperParams = parameters["bsdf-estimator"]["hyper-parameters"]

    if "bsdf-preprocess" in parameters and parameters["bsdf-preprocess"]["enabled"]:
        bsdfDictionary = json.load(open("inputs/" + parameters["bsdf-preprocess"]["file"], "r"))
    elif not bsdfAdaptiveSampling:
        bsdfDictionary = None
    else:
        print("Support for adaptive sampling without BSDF preprocessing not implemented")
        sys.exit(1)

    if "weight-reprojection" in parameters["bsdf-estimator"] and type(parameters["bsdf-estimator"]["weight-reprojection"]) is list:
        bsdfReprojectionParams = parameters["bsdf-estimator"]["weight-reprojection"]
        wtlength = len(bsdfReprojectionParams)
    elif "weight-reprojection" in parameters["bsdf-estimator"] and parameters["bsdf-estimator"]["weight-reprojection"] == True:
        wtlength = reprojectionLength
        bsdfReprojectionParams = hyperList[:reprojectionLength]
    else:
        wtlength = 0
        bsdfReprojectionParams = []

    bsdfReprojectionMask = [0] * len(hyperList)

    for pname in bsdfReprojectionParams:
        idx = hyperList.index(pname)
        bsdfReprojectionMask[idx] = 1

    if type(parameters["bsdf-estimator"]["update-list"]) is list:
        bsdfUpdateParams = parameters["bsdf-estimator"]["update-list"]
        bsdfUpdateMask = [0] * len(hyperList)

    elif parameters["bsdf-estimator"]["update-list"] == True:
        bsdfUpdateParams = hyperList[:mainLength]
        bsdfUpdateMask = [1] * mainLength + [0] * zeroPadding

    for pname in bsdfUpdateParams:
        idx = hyperList.index(pname)
        bsdfUpdateMask[idx] = 1

    print("bsdfReprojectionMask ", bsdfReprojectionMask)
    print("bsdfUpdateMask ", bsdfUpdateMask)
    print("mainLength", mainLength)
    print("paddedLength", paddedLength)

    bsdfReprojectionMask = tf.constant(bsdfReprojectionMask, dtype=tf.float32)
    bsdfUpdateMask = tf.constant(bsdfUpdateMask, dtype=tf.float32)

    sample_func = SAMPLERS[parameters["estimator"]["samples"]["type"]]
    sample_func_parameters = parameters["estimator"]["samples"]
    bsdf_sample_func = SAMPLERS[parameters["bsdf-estimator"]["samples"]["type"]]
    bsdf_sample_func_parameters = parameters["bsdf-estimator"]["samples"]

    # Create directories
    superindexSuffix = format(superindex).zfill(2)
    mkdir("meshes/gradients/" + superindexSuffix)
    mkdir("meshes/totalgradients/" + superindexSuffix)
    mkdir("meshes/normals/" + superindexSuffix)
    mkdir("meshes/normaldeltas/" + superindexSuffix)

    mkdir("renders/gradients/" + superindexSuffix)
    mkdir("renders/totalgradients/" + superindexSuffix)
    mkdir("renders/normals/" + superindexSuffix)
    mkdir("renders/normaldeltas/" + superindexSuffix)

    mkdir("images/normalized-absolute-errors/png/" + superindexSuffix)
    mkdir("images/normalized-absolute-errors/npy/" + superindexSuffix)
    mkdir("images/normalized-difference-errors/npy/" + superindexSuffix)
    mkdir("images/difference-errors/npy/" + superindexSuffix)
    mkdir("images/unweighted-difference-errors/npy/" + superindexSuffix)
    mkdir("images/current/npy/" + superindexSuffix)
    mkdir("images/samplers/npy/" + superindexSuffix)

    hyperparams = np.array([])

    if "hyper-parameters" in parameters["estimator"]:
        hyperparams = np.array(parameters["estimator"]["hyper-parameters"].values())
    elif "hyper-parameters" in parameters["original"]:
        hyperparams = np.array(parameters["original"]["hyper-parameters"].values())

    # Build parameters and hyper-parameters for the rendering.

    params = [ tf.constant(np.concatenate([np.array([W,H]), l, np.array([i])], axis=0), dtype=tf.float32, name="l-and-d-vector") for l,i in zip(lights, intensities) ]

    hparams = bsdfParameters["hyper-parameters"]
    hparams['0'] = 0 # Padding default value
    bsdf = tf.Variable([hparams[k] for k in hyperList])
    tabularBSDF = tf.Variable(tabularBSDFParameters["initialization"], dtype=tf.float32)

    # TODO: FINISH
    if bsdfAdaptiveSampling:
        #print()
        #samplingWeightList = json.load(open(parameters["weight-samples-parameter-list"], "r"))
        samplewts = tf.Variable([1.0 for k in parameters["weight-samples-parameter-list"]])
    else:
        samplewts = tf.Variable([])
    # -----------

    if bsdfoverride is not None:
        bsdf = tf.Variable(bsdfoverride, dtype=tf.float32)

    targetspecs = tf.constant([0.1, 0.1, 0.1])

    targetnormals = tf.constant(_targetnormals, dtype=tf.float32)
    normals = tf.Variable(_normals, dtype=tf.float32)

    scountEst = tf.Variable(sample_func(0, 0, sample_func_parameters), dtype=tf.float32)
    oneInstance = tf.constant(1.0)

    print("Building graph...")
    #I = tf.concat([ mitsuba.mitsuba(parameter_map=normals, params=p, bsdf=bsdf, depth=depthact, samples=scountEst,
    #            instances=tf.constant(1.0), unitindex=idx) for idx, p in enumerate(params) ], axis=2)

    niports = len(iports)
    ngports = len(gports)

    #I = tf.stack([ mitsuba.mitsuba(parameter_map=normals, params=p, bsdf=bsdf, depth=depthact, samples=scountEst,
    #            serverindex=tf.constant([iports[idx % niports], gports[idx % ngports]], name="server-ports"), unitindex=idx) for idx, p in enumerate(params) ], axis=2)

    portstack = tf.stack([ tf.constant([iports[idx % niports], gports[idx % ngports]], name="server-ports") for idx, p in enumerate(params) ], axis=1)
    paramstack = tf.stack(params, axis=1)
    #print(portstack.shape)
    #print(paramstack.shape)

    I = mitsuba.mitsuba(parameter_map=normals, params=paramstack, bsdf=bsdf, tabular_bsdf=tabularBSDF, samplewts=samplewts, depth=depthact, samples=scountEst,
                serverindex=portstack, unitindex=0)

    I = tf.squeeze(I)

    refimgtensor = tf.constant(refimg, dtype=tf.float32)
    print("Reference size: ", parameters["target"]["data"])

    regularization = None
    if "regularization" in parameters["estimator"] and parameters["estimator"]["regularization"]["enabled"]:
        if parameters["estimator"]["regularization"]["type"] == "smoothness":
            l = parameters["estimator"]["regularization"]["lambda"]
            imapX = np.pad(indexMap, ((0,0),(1,0)), mode='edge')[:,:-1]
            imapY = np.pad(indexMap, ((1,0),(0,0)), mode='edge')[:-1,:]
            imapX2 = np.pad(indexMap, ((0,0),(0,1)), mode='edge')[:,1:]
            imapY2 = np.pad(indexMap, ((0,1),(0,0)), mode='edge')[1:,:]

            regX = (tf.gather(normals, tf.constant(indexMap), axis=0) - tf.gather(normals, tf.constant(imapX), axis=0))
            regY = (tf.gather(normals, tf.constant(indexMap), axis=0) - tf.gather(normals, tf.constant(imapY), axis=0))
            regX2 = (tf.gather(normals, tf.constant(indexMap), axis=0) - tf.gather(normals, tf.constant(imapX2), axis=0))
            regY2 = (tf.gather(normals, tf.constant(indexMap), axis=0) - tf.gather(normals, tf.constant(imapY2), axis=0))
            #regY = (normals[tf.constant(indexMap), :] - normals[tf.constant(imapY), :])
            regularization = l * (tf.reduce_sum(regX ** 2) + tf.reduce_sum(regY ** 2) + tf.reduce_sum(regX2 ** 2) + tf.reduce_sum(regY2 ** 2))
    else:
        regularization = tf.constant(0, dtype=tf.float32)

    L = tf.reduce_sum(tf.pow((I - refimg) * errorWeights,2)) + regularization
    vertexMap = MeshAdjacencyBuilder.buildVertexMap(_vertices, _normals, width=W, height=H)
    Lu = tf.gather_nd(tf.reduce_sum(tf.pow((I - refimg) * errorWeights,2), axis=2), tf.constant(vertexMap))
    print L
    print normals

    if "subspace-regularization" in tabularBSDFParameters:
        M = tabularBSDFParameters["subspace-regularization"]["subspace"]
        M = tf.cast(M , tf.float32)
        Xvec = tf.reshape(tf.transpose(tabularBSDF), [8100,1])
        Lreg = tf.matmul( tf.matmul( tf.transpose(Xvec) , M) , Xvec)
        Lreg *= tabularBSDFParameters["subspace-regularization"]["lambda"]
        print("Subspace Regularization enabled with lambda " + format(tabularBSDFParameters["subspace-regularization"]["lambda"]))
    else:
        Lreg = tf.constant(0.0)

    bsdfL = tf.reduce_sum(tf.pow(((I - refimg)) * bsdfErrorWeights,2)) + Lreg
    #L = tf.reduce_sum(tf.losses.huber_loss(refimg * errorWeights, I * errorWeights))
    #logL = tf.reduce_sum(tf.log(tf.pow((I - refimg) * bsdfErrorWeights,2) + 1))
    #logL = tf.reduce_sum(tf.losses.huber_loss(refimg * bsdfErrorWeights, I * bsdfErrorWeights))

    lfactors = tf.reduce_sum(I, axis=[0,1]) / tf.reduce_sum(refimgtensor, axis=[0,1])

    optimizerParams = processOptimizerParameters(optimizerParams, superiteration=superindex)
    bsdfOptimizerParams = processOptimizerParameters(bsdfOptimizerParams, superiteration=superindex)
    tabularBsdfOptimizerParams = processOptimizerParameters(tabularBsdfOptimizerParams, superiteration=superindex)

    optimizer = getOptimizerFromParameters(optimizerParams, mainLength=mainLength, paddedLength=paddedLength)
    if optimizerParams["type"] == "mala":
        optimizer.set_unraveled_loss(Lu)

    bsdfOptimizer = getOptimizerFromParameters(bsdfOptimizerParams, mainLength=mainLength, reprojectionLength=reprojectionLength, paddedLength=paddedLength)
    tabularBSDFOptimizer = getOptimizerFromParameters(tabularBsdfOptimizerParams, mainLength=mainLength, reprojectionLength=reprojectionLength, paddedLength=paddedLength, disableSets=True)

    if ("normalize-gradients" in optimizerParams) and optimizerParams["normalize-gradients"]:
        xGsandVs = optimizer.compute_gradients(L, var_list=[normals])
        newXGsandVs = []
        for xGrad, xVar in xGsandVs:
            newXGrad = (xGrad / tf.norm(xGrad, axis=1, keepdims=True))
            #newXGrad = xGrad
            newXGsandVs.append((newXGrad, xVar))

        #yGsandVs = [(tf.clip_by_value(xGrad, -1500.0, 1500.0), xVar) for xGrad, xVar in xGsandVs]
        #vAndGs = zip(xGradients, xVariables)
        train = optimizer.apply_gradients(newXGsandVs)
    else:
        #newXGrad = tf.zeros((1,1))
        #train = optimizer.minimize(L, var_list=[normals])
        xGsandVs = optimizer.compute_gradients(L, var_list=[normals])
        print xGsandVs
        newXGsandVs = []
        for xGrad, xVar in xGsandVs:
            newXGrad = xGrad
            #newXGrad = xGrad
            newXGsandVs.append((newXGrad, xVar))
        
        if optimizerParams["type"] == "mala":
            precomp = optimizer.pre_compute(newXGsandVs)
        else:
            precomp = None

        train = optimizer.apply_gradients(newXGsandVs)

    allBSDFXGsandVs = bsdfOptimizer.compute_gradients(bsdfL, var_list=[bsdf, tabularBSDF])
    newBSDFXGsandVs = []
    bsdfXGrad, bsdfXVar = allBSDFXGsandVs[0]
    tabularBSDFXGrad, tabularBSDFXVar = allBSDFXGsandVs[1]

    # TODO: TEMPORARY: WARN: 
    # Project Ws onto a sphere to speed up convergence.
    if (wtlength != 0) and (("pre-project" in optimizerParams)) and (optimizerParams["pre-project"]):
        bsdfDeltaWs = bsdfXGrad[:wtlength]
        bsdfWs = bsdfXVar[:wtlength]
        bsdfWs = bsdfWs / tf.norm(bsdfWs)

        newBsdfDeltaWs = bsdfDeltaWs - tf.tensordot(bsdfWs, bsdfDeltaWs, 1) * (bsdfWs)

        newBsdfXGrad = tf.concat([newBsdfDeltaWs, bsdfXGrad[wtlength:]], axis=0) * bsdfUpdateMask
    else:
        newBsdfXGrad = bsdfXGrad * bsdfUpdateMask

    newBSDFXGsandVs.append((newBsdfXGrad, bsdfXVar))
    newBSDFXGsandVs.append((tabularBSDFXGrad, tabularBSDFXVar))

    #bsdfXGrad, bsdfXVar = bsdfXGsandVs[0]
    bsdfTrain = tf.group(bsdfOptimizer.apply_gradients(newBSDFXGsandVs[:1]), tabularBSDFOptimizer.apply_gradients(newBSDFXGsandVs[1:]))

    _debug_tabular_m = tabularBSDFOptimizer.get_slot(tabularBSDFXVar, 'm')
    _debug_tabular_v = tabularBSDFOptimizer.get_slot(tabularBSDFXVar, 'v')

    #bsdfTrain = bsdfOptimizer.minimize(L, var_list=[bsdf])
    cnormals = tf.stack( [ normals[:,0], normals[:,1], tf.clip_by_value(normals[:,2], 0, np.infty) ], 1 )
    normalized = cnormals / tf.sqrt(tf.reduce_sum(tf.square(cnormals), axis=1, keep_dims=True))
    projectN = normals.assign(normalized)

    # Project BSDF weights.
    wtsum = tf.reduce_sum(bsdf * bsdfReprojectionMask)
    projectW = bsdf.assign(tf.clip_by_value(bsdfReprojectionMask * (bsdf / wtsum) + (1 - bsdfReprojectionMask) * bsdf, 0.0, 0.99))

    #gerror = tf.reduce_sum(tf.pow(normals - targetnormals, 2))
    #nerror = normals - targetnormals

    difference_error = (I - refimg) * errorWeights
    unweighted_difference_error = (I - refimg)
    normalized_absolute_error = tf.abs(difference_error) / refimg
    normalized_difference_error = difference_error / refimg

    superprefix = ""
    if superindex != -1:
        superprefix = "si-" + format(superindex).zfill(2) + "-"

    # Store strings for scene files.
    colorsSceneFile = "inputs/scenes/colors-scene.xml"
    normalsSceneFile = "inputs/scenes/normals-scene.xml"

    with tf.Session() as sess:
        writer = tf.summary.FileWriter("output", sess.graph)

        init = tf.initialize_all_variables()
        sess.run(init)

        Bval = sess.run(bsdf)

        if precomp is not None:
            print("Running pre-compute")
            (_, d_fs_go, d_fs_vo, d_fs_et, d_fs_vta, d_fs_t) = sess.run([
                    precomp,
                    optimizer._debug_fs_grad_old_assign,
                    optimizer._debug_fs_var_old_assign,
                    optimizer._debug_fs_effective_tau,
                    optimizer._debug_fs_v_t_assign,
                    optimizer._debug_fs_tau
                ])
            print("pre-compute complete")

        Bs = []
        TBs = []
        TBUs = []
        Ls = []
        Ms = []
        Vs = []
        tgMs = []
        tgVs = []
        Es = []
        Ss = []
        Rs = []
        SNRs = []
        Bgrads = []
        TBgrads = []
        TBUgrads = []
        Bmodgrads = []

        if superindex == 0 and "extended-phase" in parameters["bsdf-estimator"]:
            _bsdfIterations = bsdfIterations + parameters["bsdf-estimator"]["extended-phase"]
        else:
            _bsdfIterations = bsdfIterations

        lastDifference = None
        for i in range(0, _bsdfIterations):
            print("Superiteration " +\
                format(superindex) +\
                ", bsdf iteration " +\
                format(i) + "/" +
                format(_bsdfIterations))
            # Write BSDF spatial sample texture.
            if bsdfSpatialSampling:
                scount = bsdf_sample_func(i, 0, bsdf_sample_func_parameters)
                spatialTextures, sampleMultiplier = computeSpatialTexture(
                    i, lastDifference, scount, 
                    (W,H,portstack.shape[1]), 
                    mode=bsdfParameters["samples"]["spatial-adaptive-mode"])
                for k in range(spatialTextures.shape[2]):
                    hdsutils.writeHDSImage("/tmp/sampler-" + format(k) + ".hds",
                            spatialTextures.shape[0],
                            spatialTextures.shape[1],
                            1, spatialTextures[:,:,k][:,:,np.newaxis])
                    dataio.writeNumpyData(
                        spatialTextures[:,:,k],
                        'images/samplers/npy/' +\
                            superindexSuffix + '/' +\
                            format(i + maxIterations).zfill(4) +
                            '-img-' + format(k).zfill(2)
                        )
                    cv2.imwrite('images/samplers/png/' +\
                            superindexSuffix + '/' +\
                            format(i + maxIterations).zfill(4) +
                            '-img-' + format(k).zfill(2) + ".png",
                            ((spatialTextures[:,:,k]) * (255/4.0)).astype(np.uint8)
                        )

            if not bsdfSpatialSampling:
                sess.run(scountEst.assign(bsdf_sample_func(i, 0, bsdf_sample_func_parameters)))
            else:
                sess.run(scountEst.assign(sampleMultiplier))

            (_, Lval, nae, nde, de, ude, iimg, grads, oldgrads, tabulargrads, regval, _tg_m, _tg_v, _bc_v, _bc_m, _exps, _steps, _snr, lfacs) = sess.run((bsdfTrain, L, normalized_absolute_error, normalized_difference_error, difference_error, unweighted_difference_error, I, newBsdfXGrad, bsdfXGrad, tabularBSDFXGrad, Lreg, _debug_tabular_m, _debug_tabular_v, bsdfOptimizer.bc_v, bsdfOptimizer.bc_m, bsdfOptimizer.exps, bsdfOptimizer.step, bsdfOptimizer.snr, lfactors))
            obtainedLFactors = lfacs
            # Compute sum of normals error.
            Bval = sess.run(bsdf)

            # Project Tabular BSDF
            tbsdf = sess.run(tabularBSDF)
            TBUs.append(tbsdf.tolist())
            tbu = tbsdf.tolist()
            ptbsdf = bivariate_proj.bivariate_proj(tbsdf.transpose((1,0,2))).transpose((1,0,2))
            sess.run(tabularBSDF.assign(ptbsdf))
            TBval = ptbsdf

            lastDifference = de

            # Update the BSDF sample weights.
            if bsdfAdaptiveSampling:
                samplewts.assign(computeBSDFSampleWeights(_bc_v, _bc_m, _exps, _snr, bsdf, dictionary=bsdfDictionary, mode=bsdfParameters["samples"]["bsdf-adaptive-mode"]))

            """ Run normal reprojection, if we don't have to optimize for albedo
            if not albedoEnabled:
                sess.run(projectW)"""

            for k in range(iimg.shape[2]):
                if i > 9999 or k > 99:
                    print("[WARNING] cannot output error output for greater than 10000 iterations or 100 lights")
                else:
                    dataio.writeNumpyData(iimg[:,:,k], 'images/current/npy/' + superindexSuffix + '/' + format(i + maxIterations).zfill(4) + '-img-' + format(k).zfill(2))
                    dataio.writeNumpyData(de[:,:,k], 'images/difference-errors/npy/' + superindexSuffix + '/' + format(i + maxIterations).zfill(4) + '-img-' + format(k).zfill(2))
                    dataio.writeNumpyData(nae[:,:,k], 'images/normalized-absolute-errors/npy/' + superindexSuffix + '/' + format(i + maxIterations).zfill(4) + '-img-' + format(k).zfill(2))
                    dataio.writeNumpyData(nde[:,:,k], 'images/normalized-difference-errors/npy/' + superindexSuffix + '/' + format(i + maxIterations).zfill(4) + '-img-' + format(k).zfill(2))
                    dataio.writeNumpyData(ude[:,:,k], 'images/unweighted-difference-errors/npy/' + superindexSuffix + '/' + format(i + maxIterations).zfill(4) + '-img-' + format(k).zfill(2))

            print("B values: ", Bval)
            print("L values: ", Lval)
            print("B grad:   ", grads)
            print("B old grad:   ", oldgrads)
            print("bc_m ", _bc_m)
            print("bc_v ", _bc_v)
            print("snr ", _snr)

            Bs.append(Bval.tolist())
            TBs.append(TBval.tolist())
            Ls.append(Lval.tolist())
            TBgrads.append(tabulargrads.tolist())
            Bgrads.append(oldgrads.tolist())
            Bmodgrads.append(grads.tolist())
            Ms.append(_bc_m.tolist())
            Vs.append(_bc_v.tolist())
            tgMs.append(_tg_m.tolist())
            tgVs.append(_tg_v.tolist())
            Es.append(_exps.tolist())
            Ss.append(_steps.tolist())
            Rs.append(regval.tolist())
            SNRs.append(_snr.tolist())

            #As.append(Nval.tolist())
            #Es.append(Lval.tolist())

            # Write values to an errors file.
            #efile = open("errors/errors-" + superindexSuffix + ".json", "w")
            #json.dump({"nerrors":As, "ierrors":Es}, efile)
            #efile.close()

            outputs = {
                "bvals": Bval.tolist(),
                "tbvals": TBval.tolist(),
                "tbgrads": tabulargrads.tolist(),
                "bgrads": oldgrads.tolist(),
                "tbuvals": tbu,
                "bmodgrads": grads.tolist(),
                "ierrors": Lval.tolist(),
                "bc_m": _bc_m.tolist(),
                "bc_v": _bc_v.tolist(),
                "tg_m": _tg_m.tolist(),
                "tg_v": _tg_v.tolist(),
                "exps": _exps.tolist(),
                "steps": _steps.tolist(),
                "lregs": regval.tolist(),
                "snrs": _snr.tolist()
            
            }
            

            """efile = open("errors/bsdf-errors-" + superindexSuffix + "-lregs.json", "w")
            json.dump({
                "lregs":Rs, 
                "bvals":Bs, 
                "tbvals": TBs, 
                "tbuvals": TBUs,
                "tbgrads": TBgrads, 
                "ierrors":Ls, 
                "bmodgrads":Bmodgrads, 
                "bgrads":Bgrads, 
                "bc_m": Ms, 
                "bc_v": Vs, 
                "tg_m": tgMs, 
                "tg_v": tgVs,
                "exps": Es, 
                "steps": Ss, 
                "snrs": SNRs
            }, efile)"""
            efile = open("errors/bsdf-errors-" + superindexSuffix + "-" + format(i).zfill(4) + ".json", "w")
            json.dump(outputs, efile)
            efile.close()



            """efile = open("errors/bsdf-errors-" + superindexSuffix + "-lregs.json", "w")
            json.dump({"lregs":Rs}, efile)
            efile.close()

            efile = open("errors/bsdf-errors-" + superindexSuffix + "-bvals.json", "w")
            json.dump({"bvals":Bs}, efile)
            efile.close()

            efile = open("errors/bsdf-errors-" + superindexSuffix + "-tbvals.json", "w")
            json.dump({"tbvals": TBs}, efile)
            efile.close()

            efile = open("errors/bsdf-errors-" + superindexSuffix + "-tbgrads.json", "w")
            json.dump({"tbgrads": TBgrads}, efile)
            efile.close()

            efile = open("errors/bsdf-errors-" + superindexSuffix + "-tbgrads.json", "w")
            json.dump({"tbuvals": TBUs}, efile)
            efile.close()

            efile = open("errors/bsdf-errors-" + superindexSuffix + "-tbgrads.json", "w")
            json.dump({"tbgrads": TBgrads}, efile)
            efile.close()
            
            efile = open("errors/bsdf-errors-" + superindexSuffix + "-ierrors.json", "w")
            json.dump({"ierrors":Ls}, efile)
            efile.close()
            
            efile = open("errors/bsdf-errors-" + superindexSuffix + "-bmodgrads.json", "w")
            json.dump({"bmodgrads":Bmodgrads}, efile)
            efile.close()"""

        As = []
        Ws = []
        Es = []
        Rs = []
        for i in range(0, maxIterations):
            print("Superiteration " + format(superindex) + ", normal iteration " + format(i))

            sess.run(scountEst.assign(sample_func(i, Es[:-1], sample_func_parameters)))

            malaDebugVars = []
            if optimizerParams["type"] == "mala":
                # Add debugging variables.
                malaDebugVars = (
                    optimizer._debug_mean_new,
                    optimizer._debug_theta_old,
                    optimizer._debug_q_old,
                    optimizer._debug_var_new,
                    optimizer._debug_theta_new_sample,
                    optimizer._debug_phi_new_sample,
                    optimizer._debug_theta_sample,
                    optimizer._debug_q_new,
                    optimizer._debug_effective_tau,
                    optimizer._debug_new_effective_tau,
                    optimizer._debug_var_old,
                    optimizer._debug_grad_old,
                    optimizer._debug_var_current,
                    optimizer._debug_v_t,
                    optimizer._debug_tau)
            else:
                malaDebugVars = (
                    tf.constant(0),
                    tf.constant(0),
                    tf.constant(0),
                    tf.constant(0),
                    tf.constant(0),
                    tf.constant(0),
                    tf.constant(0),
                    tf.constant(0),
                    tf.constant(0),
                    tf.constant(0),
                    tf.constant(0),
                    tf.constant(0),
                    tf.constant(0),
                    tf.constant(0),
                    tf.constant(0)
                )


            (_, Lval, reg, nae, nde, 
                de, ude, iimg, grads, lfacs, _lr,
                d_mn, d_to, d_qo, d_vn, d_tns, d_pns,
                d_ts, d_qn, d_et, d_net, d_vo, d_go, d_vc, d_vt, d_t) = sess.run(
                                                    (
                                                            train, 
                                                            L, 
                                                            regularization, 
                                                            normalized_absolute_error, 
                                                            normalized_difference_error, 
                                                            difference_error, 
                                                            unweighted_difference_error, 
                                                            I, 
                                                            newXGrad, 
                                                            lfactors,
                                                            optimizer.lr
                                                    ) + malaDebugVars)


            obtainedLFactors = lfacs
            copyfile("/tmp/mts_mesh_intensity_slot_0.ply", "meshes/normals/" + superindexSuffix + "/" + format(i).zfill(4) + ".ply")

            normalfilename = "renders/normals/" + superindexSuffix + "/" + format(i).zfill(4) + ".png"
            os.system("mitsuba " + dirpath + "/../tools/monitor/xml/normals.xml -o \"" + normalfilename + "\" -Dmesh=\"" + "meshes/normals/" + superindexSuffix + "/" + format(i).zfill(4) + ".ply" + "\" -Dwidth=256 -Dheight=256 -DsampleCount=4 > /dev/null")

            # Compute individual normals error.
            ndplyfile = "normaldeltas/" + superindexSuffix + "/" + format(i).zfill(4) + ".ply"
            #Nvals = sess.run(nerror)
            #load_normals.emplace_normals_as_colors(
            #        "/tmp/mts_mesh_intensity_slot_0.ply",
            #        "meshes/" + ndplyfile,
            #        Nvals,
            #        asfloat=True)
            ndnegplyfile, ndposplyfile = splitpolarity.makePlyNames(ndplyfile)
            splitpolarity.splitPolarity("meshes/" + ndplyfile, "meshes/" + ndnegplyfile, "meshes/" + ndposplyfile)
            ndneghdsfile, ndnegnpyfile = rendernormals.makeRenderNames(ndnegplyfile)
            ndposhdsfile, ndposnpyfile = rendernormals.makeRenderNames(ndposplyfile)
            rendernormals.renderMesh("meshes/" + ndnegplyfile, colorsSceneFile, "renders/" + ndneghdsfile, "renders/" + ndnegnpyfile, W=W, H=H)
            rendernormals.renderMesh("meshes/" + ndposplyfile, colorsSceneFile, "renders/" + ndposhdsfile, "renders/" + ndposnpyfile, W=W, H=H)

            # Store final gradients.
            #if ("normalize-gradients" in optimizerParams) and optimizerParams["normalize-gradients"]:
            if True:
                tgplyfile = "totalgradients/" + superindexSuffix + "/" + format(i).zfill(4) + ".ply"
                #if isinstance(grads, IndexedSlicesValue):
                    # Un-sparsify
                _grads = np.zeros(grads.dense_shape)
                _grads[grads.indices, :] = grads.values
                grads = _grads
                print grads.shape

                load_normals.emplace_normals_as_colors(
                        "/tmp/mts_mesh_intensity_slot_0.ply",
                        "meshes/" + tgplyfile,
                        grads,
                        asfloat=True)
                tgnegplyfile, tgposplyfile = splitpolarity.makePlyNames(tgplyfile)
                splitpolarity.splitPolarity("meshes/" + tgplyfile, "meshes/" + tgnegplyfile, "meshes/" + tgposplyfile)
                tgneghdsfile, tgnegnpyfile = rendernormals.makeRenderNames(tgnegplyfile)
                tgposhdsfile, tgposnpyfile = rendernormals.makeRenderNames(tgposplyfile)
                rendernormals.renderMesh("meshes/" + tgnegplyfile, colorsSceneFile, "renders/" + tgneghdsfile, "renders/" + tgnegnpyfile, W=W, H=H)
                rendernormals.renderMesh("meshes/" + tgposplyfile, colorsSceneFile, "renders/" + tgposhdsfile, "renders/" + tgposnpyfile, W=W, H=H)

            # Copy over gradient meshes created by the tensorflow gradient ops
            for k in range(len(lights)):
                gplyfile = "gradients/" + superindexSuffix + "/" + format(i).zfill(4) + "-img" + format(k).zfill(2) + ".ply"
                copyfile(
                        "/tmp/mts_mesh_gradients-" + format(k) + ".ply",
                        "meshes/" + gplyfile)
                gnegplyfile, gposplyfile = splitpolarity.makePlyNames(gplyfile)
                splitpolarity.splitPolarity("meshes/" + gplyfile, "meshes/" + gnegplyfile, "meshes/" + gposplyfile)
                gneghdsfile, gnegnpyfile = rendernormals.makeRenderNames(gnegplyfile)
                gposhdsfile, gposnpyfile = rendernormals.makeRenderNames(gposplyfile)
                rendernormals.renderMesh("meshes/" + gnegplyfile, colorsSceneFile, "renders/" + gneghdsfile, "renders/" + gnegnpyfile, W=W, H=H)
                rendernormals.renderMesh("meshes/" + gposplyfile, colorsSceneFile, "renders/" + gposhdsfile, "renders/" + gposnpyfile, W=W, H=H)

            # Run normal reprojection, if we don't have to optimize for albedo
            if not albedoEnabled:
                print("Projecting normals")
                sess.run(projectN)

            # Optionally reproject onto an integrable surface.
            if "enforce-integrability" in parameters["estimator"] and parameters["estimator"]["enforce-integrability"]:
                print("Enforcing Integrability...")
                normalMap = MeshAdjacencyBuilder.buildNormalMap(_vertices, sess.run(normals), radius=1.0, width=W, height=H)
                #plt.imshow(normalMap)
                #plt.show()
                iprNormals = enforceIntegrability(_vertices, sess.run(normals), W=W, H=H)
                sess.run(normals.assign(iprNormals))

            for k in range(nae.shape[2]):
                if i > 9999 or k > 99:
                    print("[WARNING] cannot output error output for greater than 10000 iterations or 100 lights")
                else:
                    Image.fromarray(((nae[:,:,k] / 10.0) * 255.0).astype(np.uint8)).save('images/normalized-absolute-errors/png/' + superindexSuffix + "/" + format(i).zfill(4) + '-img-' + format(k).zfill(2) + '.png')
                    dataio.writeNumpyData(nae[:,:,k], 'images/normalized-absolute-errors/npy/' + superindexSuffix + '/' + format(i).zfill(4) + '-img-' + format(k).zfill(2))
                    dataio.writeNumpyData(nde[:,:,k], 'images/normalized-difference-errors/npy/' + superindexSuffix + '/' + format(i).zfill(4) + '-img-' + format(k).zfill(2))
                    dataio.writeNumpyData(de[:,:,k], 'images/difference-errors/npy/' + superindexSuffix + '/' + format(i).zfill(4) + '-img-' + format(k).zfill(2))
                    dataio.writeNumpyData(iimg[:,:,k], 'images/current/npy/' + superindexSuffix + '/' + format(i).zfill(4) + '-img-' + format(k).zfill(2))
                    dataio.writeNumpyData(ude[:,:,k], 'images/unweighted-difference-errors/npy/' + superindexSuffix + '/' + format(i).zfill(4) + '-img-' + format(k).zfill(2))


            # Compute sum of normals error.
            #Nval = sess.run(gerror)
            Nval = 0.0

            print("NERROR: ", Nval)
            print("IERROR: ", Lval)
            print("Regularization Error: ", reg)
            print("LR: ", _lr)

            if optimizerParams["type"] == "mala":
                print("MALA Optimizer debug values:")
                debug_pix = 15000
                print("MEAN-NEW: ", d_mn[15000,:])
                print("VAR-CURRENT: ", d_vc[15000,:])
                print("THETA-OLD: ", d_to[15000])
                print("Q-OLD: ", d_qo[15000])
                print("VAR-NEW", d_vn[15000,:])
                print("THETA-NEW-SAMPLE", d_tns[15000])
                print("PHI-NEW-SAMPLE", d_pns[15000])
                print("THETA-SAMPLE", d_ts[15000])
                print("Q-NEW: ", d_qn[15000])
                print("EFFECTIVE-TAU: ", d_et[15000])
                print("NEW-EFFECTIVE-TAU: ", d_net[15000])
                print("VAR-OLD: ", d_vo[15000, :])
                print("GRAD-OLD: ", d_go[15000, :])
                print("FS-GRAD-OLD: ", d_fs_go[15000, :])
                print("FS-VAR-OLD: ", d_fs_vo[15000, :])
                print("FS-EFFECTIVE_TAU: ", d_fs_et[15000])
                print("FS-VT: ", d_fs_vta[15000, :])
                print("VT: ", d_vt[15000,:])
                print("TAU: ", d_t)
                print("FS-TAU: ", d_fs_t)

            As.append(Nval.tolist())
            Es.append(Lval.tolist())
            Rs.append(reg.tolist())

            # Write values to an errors file.
            efile = open("errors/errors-" + superindexSuffix + ".json", "w")
            json.dump({"nerrors":As, "ierrors":Es, "rerrors": Rs}, efile)
            efile.close()

        if ("recalibrate" in parameters["lights"]) and parameters["lights"]["recalibrate"]:
            # Recompute intensity factors.
            factors = obtainedLFactors
        else:
            factors = None

        writer.close()

        return Bval, factors