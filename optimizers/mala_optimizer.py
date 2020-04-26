
import tensorflow as tf
import numpy as np


class MALANormalOptimizer:
    """
        Implements Metropolis adjusted Langevin Algorithm, which is a MCMC algorithm
        that uses Langevin dynamics to propose updates and the Metropolis-Hastings method to
        help with fast convergence.
        In addition, this specific implementation adds support for optimizing vectors constrained to
        the unit hemisphere in the half-space z>0

        Note that this optimizer requires an unraveled loss function (per-normal loss) instead of a 
        aggregated loss.
    """

    def __init__(
        self,
        tau=0.1,
        beta2=0.2,
        gamma=2.0):

        self.tau = tf.Variable(tau)
        self.gamma = tf.constant(gamma)
        self.beta2 = tf.constant(beta2)

    def prepare(self):
        # Initialize variables.
        pass

    def compute_gradients(self, L, var_list):
        return [(tf.gradients([L], var_list)[0], var_list[0])]
    
    @classmethod
    def vector_from_tp(cls, theta_sample, phi_sample):
        return tf.stack(
            [
                tf.cos(theta_sample), 
                tf.sin(theta_sample) * tf.cos(phi_sample),
                tf.sin(theta_sample) * tf.sin(phi_sample)
            ], 1)
    
    def tp_to_vector(theta_sample, phi_sample):
        pass

    @classmethod
    def axis_forward_transform(cls, vector, hinge, axis):
        epsilon = tf.constant(1e-6)

        basisZ = axis

        basisY = tf.cross(basisZ, hinge)
        basisY = basisY / (tf.norm(basisY, axis=1, keep_dims=True) + epsilon)

        basisX = tf.cross(basisY, basisZ)
        basisX = basisX / (tf.norm(basisX, axis=1, keep_dims=True) + epsilon)

        tvector = tf.stack([tf.reduce_sum(vector * basisX, axis=1), tf.reduce_sum(vector * basisY, axis=1), tf.reduce_sum(vector * basisZ, axis=1)], 1)
        tvector = tvector / tf.norm(tvector, axis=1, keep_dims=True)
        return tvector

    def axis_reverse_transform(avector, hinge, axis):
        pass

    def minimize(
        self,
        L,
        var_list=[]):
        self.L = L
        return self.apply_gradients(self.compute_gradients(L, var_list))
        #self.L = L

        #var = var_list[0]
        # ------- Declare variables -----------
        #self.grad_old = tf.Variable(tf.zeros_like(var))
        #self.var_old = tf.Variable(tf.zeros_like(var))
        #self.itercount = tf.Variable(0)

    #def set_loss(self, loss):
    #    self.L = loss
    
    def set_unraveled_loss(self, uloss):
        self.Lu = uloss

    def pre_compute(self, gvs):
        # First step computation. Need not compute acceptance probability.
        grad, var = gvs[0]

        self.var_old = tf.Variable(tf.zeros_like(var))
        print(var.shape)
        self.grad_old = tf.Variable(tf.zeros_like(var))
        print(self.Lu)
        self.Lu_old = tf.Variable(tf.zeros((var.shape[0],)))
        self.best_value = tf.Variable(0.0)
        self.best_var = tf.Variable(tf.zeros_like(var))
        self.mean_var = tf.Variable(tf.zeros_like(var))
        self.total_var = tf.Variable(tf.zeros_like(var))
        self.total_weight = tf.Variable(0.0)
        self.v_t = tf.Variable(tf.zeros_like(var))
        self.itercount = tf.Variable(0.0)

        # ------- Build first step graph -------
        # Compute initial gradient as grad_old and updated value as theta
        fs_var_old_assign = tf.assign(self.var_old, var)
        fs_grad_old_assign = tf.assign(self.grad_old, grad)

        v_t_assign = tf.assign(self.v_t, (1 - self.beta2) * (tf.square(grad)) + self.beta2 * self.v_t)
        effective_tau = self.tau / tf.norm(v_t_assign, axis=1)

        # Debugging hooks.
        self._debug_fs_effective_tau = effective_tau
        self._debug_fs_grad_old_assign = fs_grad_old_assign
        self._debug_fs_var_old_assign = fs_var_old_assign
        self._debug_fs_v_t_assign = v_t_assign
        self._debug_fs_tau = self.tau * tf.constant(1.0)

        fs_mean = fs_var_old_assign - effective_tau[:,None] * fs_grad_old_assign

        fs_projected_mean = fs_mean / tf.norm(fs_mean, axis=1, keep_dims=True)

        fs_Lu_old_assign = tf.assign(self.Lu_old, self.Lu)
        fs_var_update = tf.assign(var, fs_projected_mean)

        # 'Executable' in order to initalize the algorithm.
        self.fs_compute = tf.group(fs_var_update, fs_Lu_old_assign)

        return self.fs_compute

    def apply_gradients(self, gvs):

        grad = gvs[0][0]
        var = gvs[0][1]


        # Compute effective tau.
        effective_tau = self.tau / tf.norm(self.v_t, axis=1)

        # ------- Build main MALA graph --------

        # Compute relative distributions (independent of var)
        theta_distribution = tf.distributions.Normal(
                            loc=tf.zeros((var.shape[0],)),
                            scale=tf.sqrt(2.0 * effective_tau))
        phi_distribution = tf.distributions.Uniform(
                            low=tf.zeros((var.shape[0],)),
                            high=tf.ones((var.shape[0],)) * 2 * tf.constant(np.pi),
                        )
        zaxis = tf.stack([
                        tf.zeros((tf.shape(var)[0],)),
                        tf.zeros((tf.shape(var)[0],)),
                        tf.ones((tf.shape(var)[0],))
                    ], 1)

        Lu = self.Lu

        mean_old = self.var_old - effective_tau[:,None] * self.grad_old
        grad_old = self.grad_old * tf.constant(1.0)
        var_old = self.var_old * tf.constant(1.0)
        var_current = var * tf.constant(1.0)

        print(mean_old.shape)
        print(var.shape)
        theta_sample = tf.acos(tf.reduce_sum(mean_old * var, axis=1) / tf.norm(mean_old, axis=1))
        print(theta_sample.shape)
        print(theta_distribution.sample().shape)
        print(phi_distribution.sample().shape)
        # Compute proposal probability density q(var_new|var_old)
        q_new = theta_distribution.prob(theta_sample) * phi_distribution.prob(tf.zeros_like(theta_sample))

        # Compute inverted proposal probability density q(var_old|var_new)
        # This is dependent only on the theta, since the phi is technically uniform
        mean_new = var - effective_tau[:,None] * grad
        theta_old = tf.acos(tf.reduce_sum(mean_new * self.var_old, axis=1) / tf.norm(mean_new, axis=1))
        q_old = theta_distribution.prob(theta_old) * phi_distribution.prob(tf.zeros_like(theta_old))

        # Compute acceptance probability
        acc_probs = tf.minimum(tf.constant(1.0), (q_new * tf.exp(-Lu)) / (q_old * tf.exp(-self.Lu_old)))
        acc_distribution = tf.distributions.Bernoulli(acc_probs)

        # Sample from the acceptance distribution. (Bernouilli)
        accepted = acc_distribution.sample()
        accepted_logical = tf.greater(accepted, tf.constant(0))

        # Modify the old values depending on whether the new value has been accepted
        Lu_old_updated = tf.assign(self.Lu_old, tf.where(accepted_logical, Lu, self.Lu_old))
        grad_old_updated = tf.assign(self.grad_old, tf.where(accepted_logical, grad, self.grad_old))
        var_old_updated = tf.assign(self.var_old, tf.where(accepted_logical, var, self.var_old))
        itercount_updated = tf.assign(self.itercount, self.itercount  + 1)

        # Update the running means and maxes.
        self.var_weight = tf.Variable(0.0)
        self.total_var = tf.Variable(tf.zeros_like(var))
        self.mean_var = tf.Variable(tf.zeros_like(var))
        self.best_value = tf.Variable(tf.zeros_like(self.Lu_old))
        self.best_var = tf.Variable(tf.zeros_like(var))

        total_weight_updated = tf.assign(self.var_weight, self.var_weight + self.tau)
        total_var_updated = tf.assign(self.total_var, self.total_var + self.tau * var_old_updated)
        mean_var_updated = tf.assign(self.mean_var, total_var_updated / total_weight_updated)
        best_value_updated = tf.assign(self.best_value, tf.maximum(self.best_value, Lu_old_updated))
        best_var_updated = tf.assign(self.best_var, tf.where(tf.greater(Lu_old_updated, self.best_value), var_old_updated, self.best_var))

        # Update the Tau number for actual convergence.
        tau_updated = tf.assign(self.tau, self.tau * ( (self.gamma + itercount_updated) / (1.0 + self.gamma + itercount_updated)))

        # Compute adaptive gradient.
        v_t_assign = tf.assign(self.v_t, (1 - self.beta2) * (tf.square(grad)) + self.beta2 * self.v_t)

        # Recompute effective tau.
        new_effective_tau = tau_updated / tf.norm(v_t_assign, axis=1)

        # Recreate the distributions under the new tau
        theta_new_distribution = tf.distributions.Normal(
                            loc=tf.zeros((tf.shape(var)[0],)),
                            scale=tf.sqrt(2.0 * new_effective_tau))
        phi_new_distribution = tf.distributions.Uniform(
                            low=tf.zeros((var.shape[0],)),
                            high=tf.ones((var.shape[0],)) * 2 * tf.constant(np.pi))

        # Sample next proposal using Langevin dynamics.
        mean_old = var_old_updated - new_effective_tau[:,None] * grad_old_updated
        projected_mean_old = mean_old / tf.norm(mean_old, axis=1, keep_dims=True)
        theta_new_sample = theta_new_distribution.sample()
        phi_new_sample = phi_new_distribution.sample()
        relative_vector_new = MALANormalOptimizer.vector_from_tp(theta_new_sample, phi_new_sample)
        var_new = MALANormalOptimizer.axis_forward_transform(relative_vector_new, projected_mean_old, zaxis)

        # Set the newly sampled var as the next value.
        var_updated = tf.assign(var, var_new)

        # Debugging hooks.
        self._debug_mean_new = mean_new
        self._debug_theta_old = theta_old
        self._debug_q_old = q_old
        self._debug_var_new = var_new
        self._debug_v_t = self.v_t * tf.constant(1.0)
        self._debug_theta_new_sample = theta_new_sample
        self._debug_phi_new_sample = phi_new_sample
        self._debug_theta_sample = theta_sample
        self._debug_q_new = q_new
        self._debug_effective_tau = effective_tau
        self._debug_new_effective_tau = new_effective_tau
        self._debug_var_old = var_old
        self._debug_grad_old = grad_old
        self._debug_var_current = var_current
        self._debug_tau = self.tau * tf.constant(1.0)

        return tf.group(var_updated, Lu_old_updated)

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