#!/usr/bin/env python3
"""
Gradients for inner product.
"""

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import sparse_ops

import os

dirname = os.path.dirname(os.path.realpath(__file__))

mitsuba_grad_module = tf.load_op_library(dirname + '/../qdispatch/mitsuba_v2_reductor_stacked_grad.so')

@ops.RegisterGradient("Mitsuba")
def _inner_product_grad_cc(op, grad):
    """
        The gradient for `inner_product` using the operation implemented in C++.

        :param op: `inner_product` `Operation` that we are differentiating, which we can use
        to find the inputs and outputs of the original op.
        :param grad: gradient with respect to the output of the `inner_product` op.
        :return: gradients with respect to the input of `inner_product`.
        """

    return mitsuba_grad_module.mitsuba_grad(grad, op.inputs[0], op.inputs[1], op.inputs[2], op.inputs[3], op.inputs[4], op.inputs[5], op.inputs[6], op.inputs[7], op.inputs[8])
    #return dir(mitsuba_grad_module)#.mitsuba_grad(grad, op.inputs[0], op.inputs[1])