# This Python file uses the following encoding: utf-8
"""Test module."""
from __future__ import division
import tensorflow as tf
import numpy as np
from time import time
from helpers import scope_op, alignedness, gather_windows, update_windows, \
    all_windows, pad
from models import CRBM, DCRBM
from functools import partial


LEARNING_RATE = 3E-3
K = 5
ALPHA = 4
SCALE = 1E-2
SYSTEM_SHAPE = (10, 10)
N_DIMS = len(SYSTEM_SHAPE)
NUM_SPINS = np.prod(SYSTEM_SHAPE)
FULL_WINDOW_SHAPE = (K*2-1,)*N_DIMS
FULL_WINDOW_SIZE = np.prod(FULL_WINDOW_SHAPE)
HALF_WINDOW_SHAPE = (K,)*N_DIMS
HALF_WINDOW_SIZE = np.prod(HALF_WINDOW_SHAPE)
H = 1.0

NUM_SAMPLES = 10
NUM_SAMPLERS = 10
ITS_PER_SAMPLE = NUM_SPINS * 10
SAMPLES_PER_SAMPLER = NUM_SAMPLES // NUM_SAMPLERS
THERM_ITS = SAMPLES_PER_SAMPLER * ITS_PER_SAMPLE
SAMPLE_ITS = THERM_ITS + (SAMPLES_PER_SAMPLER-1) * ITS_PER_SAMPLE + 1
OPTIMIZATION_ITS = 10000
ENERGY_BATCH_SIZE = 10


@scope_op()
def create_vars(model):
    """Add vars to graph."""
    model.get_variables()

    with tf.variable_scope("sampler"):
        tf.get_variable("current_samples",
                        shape=[NUM_SAMPLERS, NUM_SPINS],
                        initializer=tf.constant_initializer(0),
                        dtype=tf.int32, trainable=False)
        tf.get_variable("current_factors",
                        shape=[NUM_SAMPLERS, NUM_SPINS],
                        initializer=tf.constant_initializer(0),
                        dtype=tf.complex64, trainable=False)
        tf.get_variable("samples",
                        shape=[SAMPLES_PER_SAMPLER, NUM_SAMPLERS, NUM_SPINS],
                        initializer=tf.constant_initializer(0),
                        dtype=tf.int32, trainable=False)
        tf.get_variable("flip_positions", shape=[SAMPLE_ITS, NUM_SAMPLERS],
                        initializer=tf.constant_initializer(0),
                        dtype=tf.int32, trainable=False)
        tf.get_variable("accept_sample", shape=[SAMPLE_ITS, NUM_SAMPLERS],
                        initializer=tf.constant_initializer(0.),
                        dtype=tf.float32, trainable=False)


@scope_op()
def loss_op(factors, energies):
    """
    Compute loss.

    Parameters
    ----------
    factors : Tensor of shape (N, system_size)
    energies : Tensor of shape (N,)

    Returns
    -------
    loss : tensor of shape (N,)
    """
    n = tf.cast(tf.shape(factors)[0], tf.complex64)
    energies = tf.cast(energies, tf.complex64)
    log_psi = tf.reduce_sum(factors, range(1, N_DIMS+1))
    log_psi_conj = tf.conj(log_psi)
    energy_avg = tf.reduce_sum(energies)/n
    loss = tf.reduce_sum(energies*log_psi_conj, 0)/n - \
        energy_avg * tf.reduce_sum(log_psi_conj, 0)/n
    return tf.real(loss)


@scope_op()
def energy_op(model, states):
    """
    Compute local energies of Ising model.

    Parameters
    ----------
    states : Tensor of shape (N, system_size)

    Returns
    -------
    energies : tensor of shape (N,)
    """
    batch_size = tf.shape(states)[0]
    states_shaped = tf.reshape(states, (batch_size,)+SYSTEM_SHAPE)
    states_padded = pad(states_shaped, SYSTEM_SHAPE, [(K-1)//2]*N_DIMS)
    factors = tf.reshape(model.factors(states_padded), (batch_size, -1))
    factor_windows = all_windows(factors, SYSTEM_SHAPE, HALF_WINDOW_SHAPE)
    spin_windows = all_windows(states, SYSTEM_SHAPE, FULL_WINDOW_SHAPE)
    flipper = np.ones(FULL_WINDOW_SIZE, dtype=np.int32)
    flipper[(FULL_WINDOW_SIZE-1)//2] = -1
    spins_flipped = spin_windows * flipper
    factors_flipped = tf.reshape(
        model.factors(tf.reshape(
            spins_flipped, (batch_size*NUM_SPINS,)+FULL_WINDOW_SHAPE)),
        (batch_size, NUM_SPINS, HALF_WINDOW_SIZE))

    log_pop = tf.reduce_sum(factors_flipped - factor_windows, 2)
    energy = -H * tf.reduce_sum(tf.exp(log_pop), 1) - \
        tf.cast(alignedness(states, SYSTEM_SHAPE), tf.complex64)
    return energy / NUM_SPINS


@scope_op()
def mcmc_reset(model):
    """Reset MCMC variables."""
    with tf.variable_scope('sampler', reuse=True):
        current_samples = tf.get_variable('current_samples', dtype=tf.int32)
        current_factors = tf.get_variable('current_factors',
                                          dtype=tf.complex64)
        samples = tf.get_variable('samples', dtype=tf.int32)
        flip_positions = tf.get_variable('flip_positions', dtype=tf.int32)
        accept_sample = tf.get_variable('accept_sample', dtype=tf.float32)

    states = tf.random_uniform(
        [NUM_SAMPLERS, NUM_SPINS], 0, 2, dtype=tf.int32)*2-1
    states = tf.cast(states, tf.int32)
    states_shaped = tf.reshape(states, (NUM_SAMPLERS,)+SYSTEM_SHAPE)
    states_padded = pad(states_shaped, SYSTEM_SHAPE, [(K-1)//2]*N_DIMS)
    factors = tf.reshape(model.factors(states_padded), (NUM_SAMPLERS, -1))

    return tf.group(
        tf.assign(current_samples, states),
        tf.assign(current_factors, factors),
        tf.assign(samples, tf.zeros_like(samples, dtype=samples.dtype)),
        tf.assign(flip_positions, tf.random_uniform(
            [SAMPLE_ITS, NUM_SAMPLERS], 0, NUM_SPINS, dtype=tf.int32)),
        tf.assign(accept_sample, tf.random_uniform(
            [SAMPLE_ITS, NUM_SAMPLERS], 0., 1., dtype=tf.float32)),
    )


@scope_op()
def mcmc_step(model, i):
    """Do MCMC Step."""
    with tf.variable_scope('sampler', reuse=True):
        current_samples = tf.get_variable('current_samples', dtype=tf.int32)
        current_factors = tf.get_variable(
            'current_factors', dtype=tf.complex64)
        samples = tf.get_variable('samples', dtype=tf.int32)
        flip_positions = tf.get_variable('flip_positions', dtype=tf.int32)
        accept_sample = tf.get_variable('accept_sample', dtype=tf.float32)

    centers = flip_positions[i]

    spin_windows = gather_windows(
        current_samples, centers, SYSTEM_SHAPE, FULL_WINDOW_SHAPE)
    factor_windows = gather_windows(
        current_factors, centers, SYSTEM_SHAPE, HALF_WINDOW_SHAPE)

    flipper = np.ones(FULL_WINDOW_SIZE, dtype=np.int32)
    flipper[(FULL_WINDOW_SIZE-1)//2] = -1
    flipped_spins = spin_windows * flipper
    flipped_factors = tf.reshape(
        model.factors(
            tf.reshape(flipped_spins, (NUM_SAMPLERS,)+FULL_WINDOW_SHAPE)),
        (NUM_SAMPLERS, -1))
    accept_prob = tf.pow(tf.abs(tf.exp(
        tf.reduce_sum(flipped_factors - factor_windows, 1))), 2)
    mask = accept_prob > accept_sample[i]
    current_samples = update_windows(current_samples, centers, flipped_spins,
                                     mask, SYSTEM_SHAPE, FULL_WINDOW_SHAPE)
    current_factors = update_windows(current_factors, centers, flipped_factors,
                                     mask, SYSTEM_SHAPE, HALF_WINDOW_SHAPE)

    def write_samples():
        """Write current_samples to samples."""
        with tf.variable_scope('sampler', reuse=True):
            current_samples = tf.get_variable('current_samples',
                                              dtype=tf.int32)
            samples = tf.get_variable('samples', dtype=tf.int32)
        j = (i - THERM_ITS)//ITS_PER_SAMPLE
        return tf.scatter_update(samples, j, current_samples)

    with tf.control_dependencies([current_samples, current_factors]):
        write_op = tf.cond(
            tf.logical_and(
                tf.greater_equal(i, THERM_ITS),
                tf.equal((i - THERM_ITS) % ITS_PER_SAMPLE, 0)),
            write_samples, lambda: samples)

    with tf.control_dependencies([write_op, mask]):
        return i+1


@scope_op()
def mcmc_op(model):
    """
    Compute MCMC Samples.

    Returns
    -------
    states : tensor of shape (N, system_size)
    """
    with tf.control_dependencies([mcmc_reset(model)]):
        loop = tf.while_loop(
            lambda i: i < SAMPLE_ITS,
            partial(mcmc_step, model),
            [tf.constant(0)],
            parallel_iterations=1,
            back_prop=False
        )
    with tf.control_dependencies([loop]):
        with tf.variable_scope('sampler', reuse=True):
            samples = tf.get_variable('samples', dtype=tf.int32)
        return tf.reshape(samples, [NUM_SAMPLES, NUM_SPINS])


@scope_op()
def optimize_op(model):
    """
    Perform optimization iteration.

    Returns
    -------
    energies : tensor of shape (N,)
        Energy of MCMC samples
    train_op : Optimization tensorflow op
    """
    with tf.device('/cpu:0'):
        samples = tf.stop_gradient(mcmc_op(model))
        # Compute energy in batches
        energies = tf.map_fn(
            fn=partial(energy_op, model),
            elems=tf.reshape(samples, (-1, ENERGY_BATCH_SIZE, NUM_SPINS)),
            dtype=tf.complex64,
            parallel_iterations=1,
            back_prop=False)
        energies = tf.reshape(energies, (-1,))
        energies = tf.stop_gradient(energies)

    with tf.device('/gpu:0'):
        samples_shaped = tf.reshape(samples, (NUM_SAMPLES,)+SYSTEM_SHAPE)
        samples_padded = pad(samples_shaped, SYSTEM_SHAPE, [(K-1)//2]*N_DIMS)
        loss = loss_op(model.factors(samples_padded), energies)
        # optimizer = tf.train.model(LEARNING_RATE)
        optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
        train_op = optimizer.minimize(loss)

    return energies, train_op


config = tf.ConfigProto(
    # log_device_placement=True,
    allow_soft_placement=True
)
with tf.Graph().as_default(), tf.Session(config=config) as sess:
    model = CRBM(K, ALPHA, N_DIMS)
    # model = DCRBM(3, [8, 8], N_DIMS)
    create_vars(model)
    op = optimize_op(model)
    sess.run(tf.global_variables_initializer())

    # writer = tf.summary.FileWriter('./logs', sess.graph)
    for it in range(OPTIMIZATION_ITS):
        start = time()
        e, _ = sess.run(op)
        e = np.real(e)
        print("It %d, E=%.5f (%.2e) %.1fs" %
              (it+1, e.mean(), e.std()/np.sqrt(NUM_SAMPLES), time()-start))
    # writer.flush()
