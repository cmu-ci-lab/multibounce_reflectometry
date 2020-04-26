import tensorflow as tf
import numpy as np

class ExpGradSetAdamOptimizer:
    """
        Implements a hybrid optimizer that applies the adaptive gradients provided by Adam 
        with the exponentated update rule that allows the optimization of non-negative
        parameters that are constrained to sum to 1.
    """

    def __init__(
        self,
        lr=0.001,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-08,
        sets=None,
        excl=None):

        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2

        self.epsilon = epsilon

        self.sets = sets
        self.excl = excl

    def prepare(self):
        # Initialize variables.
        pass

    def minimize(
        self,
        L,
        var_list=[]):

        return self.apply_gradients(self.compute_gradients(L, var_list))

    def compute_gradients(self, L, var_list):
        #return [(tf.gradients([L], var_list)[0], var_list[0])]
        return zip(tf.gradients([L], var_list), var_list)

    def apply_gradients(self, gvs):
        if not len(gvs) == 1:
            print("Exponentiated Adam supports exactly 1 variable. Specify additional gradients separately.")
            return None

        var = gvs[0][1]
        grads = gvs[0][0]

        m_t = tf.Variable(tf.zeros_like(var))
        v_t = tf.Variable(tf.zeros_like(var))
        beta1 = tf.Variable(1.0)
        beta2 = tf.Variable(1.0)

        m_t_assign = tf.assign(m_t, (1 - self.beta1) * grads + self.beta1 * m_t)
        v_t_assign = tf.assign(v_t, (1 - self.beta2) * (grads * grads) + self.beta2 * v_t)

        beta1_assign = tf.assign(beta1, tf.scalar_mul(self.beta1, beta1))
        beta2_assign = tf.assign(beta2, tf.scalar_mul(self.beta2, beta2))

        bc_m = m_t_assign / (1 - beta1_assign)
        bc_v = v_t_assign / (1 - beta2_assign)

        self.bc_m = bc_m
        self.bc_v = bc_v

        bc_va = bc_v
        for varset in self.sets:
            print varset
            bc_va = (tf.scalar_mul(tf.reduce_mean(bc_va * varset), varset)) + (1 - varset) * bc_va

        delta = -tf.scalar_mul(self.lr, bc_m/tf.add(tf.sqrt(bc_va), self.epsilon))
        exps = tf.exp(delta)
        self.exps = exps
        self.step = 1.0/( tf.scalar_mul((1 - beta1_assign), tf.add(tf.sqrt(bc_v), self.epsilon)))
        self.snr = tf.abs(self.bc_m / tf.sqrt(self.bc_v))

        expUpdated = var * exps # Exp gradient.
        expUpdated = expUpdated / tf.reduce_sum(expUpdated)

        var_update = tf.assign(var, expUpdated)
        return var_update