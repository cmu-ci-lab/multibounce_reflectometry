
import tensorflow as tf
import numpy as np

class GroupedAdamOptimizer:

    def __init__(
        self,
        lr=0.001,
        beta1=0.9,
        beta2=0.999,
        decay=0.92,
        epsilon=1e-08):

        self.lr_base = tf.constant(lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.decay = decay

        self.epsilon = epsilon

    def prepare(self):
        # Initialize variables.
        pass

    def compute_gradients(self, L, var_list):
        return [(tf.gradients([L], var_list)[0], var_list[0])]

    def minimize(
        self,
        L,
        var_list=[]):

        return self.apply_gradients(self.compute_gradients(L, var_list))

    def apply_gradients(self, gvs):
        if not len(gvs) == 1:
            print("GroupedAdam supports exactly 1 variable. Specify additional gradients separately.")
            return None

        var = gvs[0][1]
        grads = gvs[0][0]

        # Initialize.
        m_t = tf.Variable(tf.zeros_like(var))
        v_t = tf.Variable(tf.zeros_like(var))
        beta1 = tf.Variable(1.0)
        beta2 = tf.Variable(1.0)
        self.lr = tf.Variable(self.lr_base)
        iterations = tf.Variable(0.0)

        m_t_assign = tf.assign(m_t, (1 - tf.constant(self.beta1)) * grads + tf.constant(self.beta1) * m_t)
        v_t_assign = tf.assign(v_t, (1 - tf.constant(self.beta2)) * (tf.square(grads)) + tf.constant(self.beta2) * v_t)

        beta1_assign = tf.assign(beta1, tf.scalar_mul(self.beta1, beta1))
        beta2_assign = tf.assign(beta2, tf.scalar_mul(self.beta2, beta2))
        iterations_assign = tf.assign(iterations, iterations+1)
        lr_assign = tf.assign(self.lr, self.lr_base * ((self.decay) / (iterations_assign + self.decay)))

        bc_m = m_t_assign / (1 - beta1_assign)
        bc_v = v_t_assign / (1 - beta2_assign)

        # This is the only change from standard Adam.
        bc_va = tf.reduce_mean(bc_v, axis=1, keep_dims=True)

        var_update = tf.assign(var, var - (tf.scalar_mul(self.lr, bc_m/tf.add(tf.sqrt(bc_va), self.epsilon))))
        return tf.group(var_update, lr_assign)

# A special case of the grouped adam optimizer that applies
# grouping to a sets of indices
class SetAdamOptimizer:

    def __init__(
        self,
        lr=0.001,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-08,
        sets=None):

        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2

        self.epsilon = epsilon

        self.sets = sets

    def prepare(self):
        # Initialize variables.
        pass

    def minimize(
        self,
        L,
        var_list=[]):

        return self.apply_gradients(self.compute_gradients(L, var_list))

    def compute_gradients(self, L, var_list):
        return [(tf.gradients([L], var_list)[0], var_list[0])]

    def apply_gradients(self, gvs):
        if not len(gvs) == 1:
            print("GroupedAdam supports exactly 1 variable. Specify additional gradients separately.")
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
        #bc_va = tf.reduce_mean(bc_v, axis=1, keep_dims=True)
        bc_va = bc_v
        for varset in self.sets:
            print varset
            bc_va = (tf.scalar_mul(tf.reduce_mean(bc_va * varset), varset)) + (1 - varset) * bc_va

        var_update = tf.assign(var, var - (tf.scalar_mul(self.lr, bc_m/tf.add(tf.sqrt(bc_va), self.epsilon))))
        return var_update