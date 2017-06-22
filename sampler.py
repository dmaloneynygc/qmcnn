"""Sampler module."""
from helpers import scope_op, gather_windows, update_windows, pad
import tensorflow as tf
import numpy as np
from functools import partial


class Sampler(object):
    """Sampler class."""

    MAX_NUM_SAMPLERS = 1000
    SWEEPFACTOR = 10

    def __init__(self, system_shape, k, num_samples):
        """Initialise."""
        self.system_shape = system_shape
        self.k = k
        self.num_samples = num_samples

        self.n_dims = len(system_shape)
        self.num_spins = np.prod(system_shape)
        self.full_window_shape = (k*2-1,)*self.n_dims
        self.full_window_size = np.prod(self.full_window_shape)
        self.half_window_shape = (k,)*self.n_dims
        self.half_window_size = np.prod(self.half_window_shape)

        self.num_samplers = min(num_samples, self.MAX_NUM_SAMPLERS)
        self.its_per_sample = self.num_spins * self.SWEEPFACTOR
        self.samples_per_sampler = num_samples // self.num_samplers
        self.therm_its = self.samples_per_sampler * self.its_per_sample
        self.sample_its = (self.therm_its + (self.samples_per_sampler-1)
                           * self.its_per_sample + 1)

        with tf.variable_scope("sampler_%03d" % np.random.randint(1000)):
            self.current_samples_var = tf.get_variable(
                "current_samples",
                shape=[self.num_samplers, self.num_spins],
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
                shape=[self.sample_its, self.num_samplers],
                initializer=tf.constant_initializer(0),
                dtype=tf.int32, trainable=False)
            self.accept_sample_var = tf.get_variable(
                "accept_sample",
                shape=[self.sample_its, self.num_samplers],
                initializer=tf.constant_initializer(0.),
                dtype=tf.float32, trainable=False)

    @scope_op()
    def mcmc_reset(self, model):
        """Reset MCMC variables."""
        states = tf.random_uniform(
            [self.num_samplers, self.num_spins], 0, 2, dtype=tf.int32)*2-1
        states = tf.cast(states, tf.int32)
        states_shaped = tf.reshape(states,
                                   (self.num_samplers,)+self.system_shape)
        states_padded = pad(states_shaped, self.system_shape,
                            [(self.k-1)//2]*self.n_dims)
        factors = tf.reshape(model.factors(states_padded),
                             (self.num_samplers, -1))

        return tf.group(
            tf.assign(self.current_samples_var, states),
            tf.assign(self.current_factors_var, factors),
            tf.assign(self.samples_var,
                      tf.zeros_like(self.samples_var, dtype=tf.int32)),
            tf.assign(self.flip_positions_var, tf.random_uniform(
                [self.sample_its, self.num_samplers], 0, self.num_spins,
                dtype=tf.int32)),
            tf.assign(self.accept_sample_var, tf.random_uniform(
                [self.sample_its, self.num_samplers], 0., 1.,
                dtype=tf.float32)),
        )

    @scope_op()
    def mcmc_step(self, model, i):
        """Do MCMC Step."""
        centers = self.flip_positions_var[i]

        spin_windows = gather_windows(
            self.current_samples_var, centers,
            self.system_shape, self.full_window_shape)
        factor_windows = gather_windows(
            self.current_factors_var, centers,
            self.system_shape, self.half_window_shape)

        flipper = np.ones(self.full_window_size, dtype=np.int32)
        flipper[(self.full_window_size-1)//2] = -1
        flipped_spins = spin_windows * flipper
        flipped_factors = tf.reshape(
            model.factors(
                tf.reshape(flipped_spins,
                           (self.num_samplers,)+self.full_window_shape)),
            (self.num_samplers, -1))
        accept_prob = tf.pow(tf.abs(tf.exp(
            tf.reduce_sum(flipped_factors - factor_windows, 1))), 2)
        mask = accept_prob > self.accept_sample_var[i]
        current_samples_op = update_windows(
            self.current_samples_var, centers, flipped_spins, mask,
            self.system_shape, self.full_window_shape)
        current_factors_op = update_windows(
            self.current_factors_var, centers, flipped_factors, mask,
            self.system_shape, self.half_window_shape)

        def write_samples():
            """Write current_samples to samples."""
            j = (i - self.therm_its)//self.its_per_sample
            return tf.scatter_update(
                self.samples_var, j, self.current_samples_var)

        with tf.control_dependencies([current_samples_op, current_factors_op]):
            write_op = tf.cond(
                tf.logical_and(
                    tf.greater_equal(i, self.therm_its),
                    tf.equal((i - self.therm_its) % self.its_per_sample, 0)),
                write_samples, lambda: self.samples_var)

        with tf.control_dependencies([write_op, mask]):
            return i+1

    @scope_op()
    def mcmc_op(self, model):
        """
        Compute MCMC Samples.

        Returns
        -------
        states : tensor of shape (N, system_size)
        """
        with tf.control_dependencies([self.mcmc_reset(model)]):
            loop = tf.while_loop(
                lambda i: i < self.sample_its,
                partial(self.mcmc_step, model),
                [tf.constant(0)],
                parallel_iterations=1,
                back_prop=False
            )
        with tf.control_dependencies([loop]):
            return tf.reshape(self.samples_var,
                              [self.num_samples, self.num_spins])
