import numpy as np
import tensorflow as tf
import os
import hdsutils

def loadLinearParameters(f):
    return np.load(f)

def renderDictionary(renderParams, dictionary, directory=".", cache=False):
    data = []
    if not os.path.exists(directory + "/linear-dictionary"):
        os.mkdir(directory + "/linear-dictionary")

    for k, params in enumerate(dictionary):
        targets = []
        for i, l in enumerate(renderParams["lights"]):
            outfile = directory + "/linear-dictionary/entry-" + format(k).zfill(6) + "-img-" + format(i).zfill(2) + ".hds"
            print "Rendering: ", params
            if (not cache) or (not os.path.exists(outfile)):
                os.system(("mitsuba " +
                        directory + "/" + renderParams["scene"]
                        + " -o " + outfile
                        + " -DlightX=" + format(l[0])
                        + " -DlightY=" + format(l[1])
                        + " -DlightZ=" + format(l[2])
                        + " -Ddepth=" + format(renderParams["depth"])
                        + " -DsampleCount=" + format(renderParams["samples"])
                        + " -Dmesh=" + directory + "/" + format(renderParams["mesh"])
                        + " -Dweight1=" + format(params[0])
                        + " -Dweight2=" + format(params[1])
                        + " -Dalpha=" + format(params[2])
                        + " -Deta=" + format(params[3])
                        ))
            targets.append(hdsutils.loadHDSImage(outfile))

        data.append(np.stack(targets, axis=0))

    return np.stack(data, axis=0)

def estimateWeights(dictionary, sourceimg, type="lstsq"):
    # Convert from NxTxWxH to a NxX matrix.
    sh = dictionary.shape
    dictionary = dictionary.reshape([sh[0], sh[1] * sh[2] * sh[3]])

    # XxN
    dictionary = dictionary.transpose([1,0])

    # Xx1
    sourceimg = sourceimg.reshape([sourceimg.shape[0] * sourceimg.shape[1] * sourceimg.shape[2],1])

    if type == "lstsq":
        # Nx1
        W = np.linalg.lstsq(dictionary, sourceimg)
    elif type == "lasso":
        model = linear_model.Lasso(alpha=0.1)
        model.fit(dictionary, sourceimg)
        W = model.coef_
    elif type == "sgd":
        W = solveSGD(dictionary, sourceimg, max_iters=600)

    return W.flatten()

def solveSGD(dictionary, sourceimg, max_iters=600):

    X = dictionary.shape[0]
    N = dictionary.shape[1]
    assert(X == sourceimg.shape[0])

    # Solve Ax=b
    w_ = np.zeros((N,1))
    w = tf.Variable(w_,dtype=tf.float32)

    L = tf.reduce_sum(tf.square(tf.matmul(tf.constant(dictionary, dtype=tf.float32), w) - tf.constant(sourceimg, dtype=tf.float32)))

    optimizer = tf.train.GradientDescentOptimizer(0.01)

    train = optimizer.minimize(L, var_list=[w])

    projectW = w.assign(tf.clip_by_value(w,0,10.0))

    print(dictionary.shape)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(max_iters):
            sess.run(train)
            sess.run(projectW)
            #print sess.run(w), sess.run(L)
    Wval = sess.run(w)

    return Wval

