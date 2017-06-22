# This Python file uses the following encoding: utf-8
"""Test module."""
from __future__ import division
import tensorflow as tf
import numpy as np
from time import time
from helpers import scope_op, alignedness, all_windows, pad
from models import CRBM
from sampler import Sampler
from functools import partial


LEARNING_RATE = 3E-3
K = 5
ALPHA = 4
SCALE = 1E-2
SYSTEM_SHAPE = (40,)
N_DIMS = len(SYSTEM_SHAPE)
NUM_SPINS = np.prod(SYSTEM_SHAPE)
FULL_WINDOW_SHAPE = (K*2-1,)*N_DIMS
FULL_WINDOW_SIZE = np.prod(FULL_WINDOW_SHAPE)
HALF_WINDOW_SHAPE = (K,)*N_DIMS
HALF_WINDOW_SIZE = np.prod(HALF_WINDOW_SHAPE)
H = 1.0

NUM_SAMPLES = 100
OPTIMIZATION_ITS = 10000
ENERGY_BATCH_SIZE = 1000

NUM_EVAL_SAMPLES = 10000
EVAL_FREQ = 500


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
def batched_energy_op(model, states):
    """Do on-graph batched energy computation."""
    energies = tf.map_fn(
        fn=partial(energy_op, model),
        elems=tf.reshape(states, (-1, ENERGY_BATCH_SIZE, NUM_SPINS)),
        dtype=tf.complex64,
        parallel_iterations=1,
        back_prop=False)
    return tf.reshape(energies, (-1,))


@scope_op()
def optimize_op(sampler, model):
    """
    Perform optimization iteration.

    Returns
    -------
    energies : tensor of shape (N,)
        Energy of MCMC samples
    train_op : Optimization tensorflow op
    """
    with tf.device('/cpu:0'):
        samples = tf.stop_gradient(sampler.mcmc_op(model))
        energies = energy_op(model, samples)
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
    sampler = Sampler(SYSTEM_SHAPE, K, NUM_SAMPLES)
    eval_sampler = Sampler(SYSTEM_SHAPE, K, NUM_EVAL_SAMPLES)
    # model = DCRBM(3, [8, 8], N_DIMS)
    op = optimize_op(sampler, model)
    eval_energies = batched_energy_op(model, eval_sampler.mcmc_op(model))
    sess.run(tf.global_variables_initializer())

    # writer = tf.summary.FileWriter('./logs', sess.graph)
    for it in range(OPTIMIZATION_ITS):
        start = time()
        e, _ = sess.run(op)
        e = np.real(e)
        print("It %d, E=%.5f (%.2e) %.1fs" %
              (it+1, e.mean(), e.std()/np.sqrt(e.shape[0]), time()-start))

        if it > 0 and (it+1) % EVAL_FREQ == 0:
            start = time()
            e = sess.run(eval_energies)
            e = np.real(e)
            print("Eval It %d, E=%.5f (%.2e) %.1fs" %
                  (it+1, e.mean(), e.std()/np.sqrt(e.shape[0]), time()-start))

    # writer.flush()
