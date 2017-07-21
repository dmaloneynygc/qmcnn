"""Sampler module."""
from helpers import scope_op, pad, unpad
import tensorflow as tf
import numpy as np


class Sampler(object):
    """Sampler class."""

    MAX_NUM_SAMPLERS = 1000
    SWEEPFACTOR = 10
    THERMFACTOR = 4

    def __init__(self, model, system_shape, r, num_samples, num_flips):
        """Initialise."""
        self.model = model
        self.system_shape = system_shape
        self.r = r
        self.num_samples = num_samples
        self.num_flips = num_flips

        self.n_dims = len(system_shape)
        self.num_spins = np.prod(system_shape)
        self.full_window_shape = (r*2-1,)*self.n_dims
        self.full_window_size = np.prod(self.full_window_shape)
        self.half_window_shape = (r,)*self.n_dims
        self.half_window_size = np.prod(self.half_window_shape)

        self.num_samplers = min(num_samples, self.MAX_NUM_SAMPLERS)
        self.its_per_sample = self.num_spins * self.SWEEPFACTOR
        self.samples_per_sampler = num_samples // self.num_samplers
        self.therm_its = self.samples_per_sampler * self.its_per_sample * \
            self.THERMFACTOR
        self.sample_its = (self.therm_its + (self.samples_per_sampler-1)
                           * self.its_per_sample + 1)

        self.padded_shape = tuple(s+r-1 for s in system_shape)
        self.padded_size = np.prod(self.padded_shape)

        self.new_samples = tf.Variable(True)

        name = "sampler_%08d" % np.random.randint(10**8)
        with tf.device('/cpu:0'), tf.variable_scope(name):
            self.current_samples_var = tf.get_variable(
                "current_samples",
                shape=[self.num_samplers, self.padded_size],
                initializer=tf.constant_initializer(0),
                dtype=tf.int32, trainable=False)
            self.current_factors_var = tf.get_variable(
                "current_factors",
                shape=[self.num_samplers, self.num_spins],
                initializer=tf.constant_initializer(0),
                dtype=tf.complex64, trainable=False)
            self.samples_var = tf.get_variable(
                "samples",
                shape=[self.samples_per_sampler, self.num_samplers,
                       self.num_spins],
                initializer=tf.constant_initializer(0),
                dtype=tf.int32, trainable=False)
            self.flip_positions_var = tf.get_variable(
                "flip_positions",
                shape=[self.sample_its, self.num_samplers, self.num_flips],
                initializer=tf.constant_initializer(0),
                dtype=tf.int32, trainable=False)
            self.accept_sample_var = tf.get_variable(
                "accept_sample",
                shape=[self.sample_its, self.num_samplers],
                initializer=tf.constant_initializer(0.),
                dtype=tf.float32, trainable=False)

    @scope_op()
    def mcmc_reset(self):
        """Reset MCMC variables."""
        uniform_states = tf.random_uniform(
            (self.num_samplers,)+self.system_shape, 0, 2, dtype=tf.int32)*2-1
        uniform_states_padded = pad(uniform_states, self.system_shape,
                                    [(self.r-1)//2]*self.n_dims)
        uniform_states_flattened = tf.reshape(uniform_states_padded,
                                              (self.num_samplers, -1))

        states = tf.cond(self.new_samples,
                         lambda: uniform_states_flattened,
                         lambda: self.current_samples_var)

        factors = tf.reshape(
            self.model.factors(
                tf.reshape(states, (self.num_samplers,)+self.padded_shape)
            ), (self.num_samplers, -1))

        return tf.group(
            tf.assign(self.current_samples_var, states),
            tf.assign(self.current_factors_var, factors),
            tf.assign(self.samples_var,
                      tf.zeros_like(self.samples_var, dtype=tf.int32)),
            tf.assign(self.flip_positions_var, tf.random_uniform(
                [self.sample_its, self.num_samplers, self.num_flips],
                0, self.num_spins, dtype=tf.int32)),
            tf.assign(self.accept_sample_var, tf.random_uniform(
                [self.sample_its, self.num_samplers], 0., 1.,
                dtype=tf.float32)),
        )

    @scope_op()
    def mcmc_step(self, i):
        """Do MCMC Step."""
        centers = self.flip_positions_var[i]

        flipper = np.ones((self.num_spins, self.num_spins), dtype=np.int32)
        flipper[np.arange(self.num_spins), np.arange(self.num_spins)] = -1
        flipper_shaped = flipper.reshape((self.num_spins,) + self.system_shape)
        pad = [[0, 0]] + [[(self.r-1)//2, (self.r-1)//2]]*self.n_dims
        flipper_padded = np.pad(flipper_shaped, pad, 'wrap') \
            .reshape((self.num_spins, self.padded_size))
        flipper_gathered = tf.gather(flipper_padded, centers)
        combined_flipper = tf.reduce_prod(flipper_gathered, 1)

        flipped_spins = self.current_samples_var * combined_flipper
        flipped_factors = tf.reshape(
            self.model.factors(
                tf.reshape(flipped_spins,
                           (self.num_samplers,)+self.padded_shape)),
            (self.num_samplers, self.num_spins))
        accept_prob = tf.pow(tf.abs(tf.exp(
            tf.reduce_sum(flipped_factors - self.current_factors_var, 1))), 2)
        mask = accept_prob > self.accept_sample_var[i]
        # mask = tf.Print(mask, [tf.reduce_mean(tf.cast(mask, tf.float32))])
        mask_idxs = tf.boolean_mask(np.arange(self.num_samplers), mask)
        current_samples_op = tf.scatter_update(
            self.current_samples_var, mask_idxs,
            tf.boolean_mask(flipped_spins, mask))
        current_factors_op = tf.scatter_update(
            self.current_factors_var, mask_idxs,
            tf.boolean_mask(flipped_factors, mask))

        def write_samples():
            """Write current_samples to samples."""
            j = (i - self.therm_its)//self.its_per_sample
            samples = tf.reshape(
                unpad(
                    tf.reshape(self.current_samples_var,
                               (self.num_samplers,)+self.padded_shape),
                    ((self.r-1)//2,)*self.n_dims),
                (self.num_samplers, self.num_spins))
            return tf.scatter_update(
                self.samples_var, j, samples)

        with tf.control_dependencies([current_samples_op, current_factors_op]):
            write_op = tf.cond(
                tf.logical_and(
                    tf.greater_equal(i, self.therm_its),
                    tf.equal((i - self.therm_its) % self.its_per_sample, 0)),
                write_samples, lambda: self.samples_var)

        with tf.control_dependencies([write_op, mask]):
            return i+1

    @scope_op()
    def mcmc_op(self):
        """
        Compute MCMC Samples.

        Returns
        -------
        states : tensor of shape (N, system_size)
        """
        with tf.device('/cpu:0'):
            with tf.control_dependencies([self.mcmc_reset()]):
                loop = tf.while_loop(
                    lambda i: i < self.sample_its,
                    self.mcmc_step,
                    [tf.constant(0)],
                    parallel_iterations=1,
                    back_prop=False
                )
            with tf.control_dependencies([loop]):
                return tf.reshape(self.samples_var,
                                  [self.num_samples, self.num_spins])
