# This Python file uses the following encoding: utf-8
"""Test module."""
from __future__ import division, print_function
import tensorflow as tf
import numpy as np
from time import time
from helpers import scope_op, interactions, all_windows, pad
from models import CRBM
from sampler import Sampler
from functools import partial
from sys import stdout


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

NUM_SAMPLES = 100
OPTIMIZATION_ITS = 10000
ENERGY_BATCH_SIZE = 1000

NUM_EVAL_SAMPLES = 1000
EVAL_FREQ = 20


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
def ising_energy(model, states):
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
    alignedness = tf.reduce_sum(interactions(states, SYSTEM_SHAPE), [1, 2])
    energy = -H * tf.reduce_sum(tf.exp(log_pop), 1) - \
        tf.cast(alignedness, tf.complex64)
    return energy / NUM_SPINS


@scope_op()
def heisenberg_energy(model, states):
    """
    Compute local energies of AntiFerroMagneticHeisenberg model.

    Parameters
    ----------
    states : Tensor of shape (N, system_size)

    Returns
    -------
    energies : tensor of shape (N,)
    """
    FULL_WINDOW_SHAPE = (K*2-1+2,)*N_DIMS
    FULL_WINDOW_SIZE = np.prod(FULL_WINDOW_SHAPE)
    HALF_WINDOW_SHAPE = (K+2,)*N_DIMS
    HALF_WINDOW_SIZE = np.prod(HALF_WINDOW_SHAPE)

    batch_size = tf.shape(states)[0]
    states_shaped = tf.reshape(states, (batch_size,)+SYSTEM_SHAPE)
    states_padded = pad(states_shaped, SYSTEM_SHAPE, [(K-1)//2]*N_DIMS)
    factors = tf.reshape(model.factors(states_padded), (batch_size, -1))
    factor_windows = all_windows(factors, SYSTEM_SHAPE, HALF_WINDOW_SHAPE)
    spin_windows = all_windows(states, SYSTEM_SHAPE, FULL_WINDOW_SHAPE)

    flippers = np.zeros((N_DIMS, FULL_WINDOW_SIZE), dtype=np.int32)
    for d in range(N_DIMS):
        flipper = np.ones(FULL_WINDOW_SHAPE, dtype=np.int32)
        halfpoint = tuple((s-1)//2 for s in FULL_WINDOW_SHAPE)
        neighbour = tuple(x+1 if i == d else x
                          for i, x in enumerate(halfpoint))
        flipper[halfpoint] = -1
        flipper[neighbour] = -1
        flippers[d, :] = flipper.flatten()

    # [N, flip_direction=1..N_DIMS, spin_window=1..SYSTEM_SIZE, window_size]
    spins_flipped = spin_windows[:, None, :, :] * flippers[None, :, None, :]
    factors_flipped = tf.reshape(
        model.factors(tf.reshape(
            spins_flipped, (batch_size*N_DIMS*NUM_SPINS,)+FULL_WINDOW_SHAPE)),
        (batch_size, N_DIMS, NUM_SPINS, HALF_WINDOW_SIZE))

    log_pop = tf.reduce_sum(factors_flipped - factor_windows[:, None, :, :], 3)
    ints = tf.cast(interactions(states, SYSTEM_SHAPE), tf.complex64)

    terms = -(1 - ints) * tf.exp(log_pop) + ints
    energy = tf.reduce_sum(terms, [1, 2])

    return energy / NUM_SPINS


@scope_op()
def batched_op(fn, states, batch_size):
    """Do on-graph batched computation."""
    results = tf.map_fn(
        fn=fn,
        elems=tf.reshape(states, (-1, batch_size, NUM_SPINS)),
        dtype=tf.complex64,
        parallel_iterations=1,
        back_prop=False)
    return tf.reshape(results, (-1,))


@scope_op()
def optimize_op(sampler, model, energy_fn):
    """
    Perform optimization iteration.

    Returns
    -------
    energies : tensor of shape (N,)
        Energy of MCMC samples
    train_op : Optimization tensorflow op
    """
    with tf.device('/cpu:0'):
        samples = tf.stop_gradient(sampler.mcmc_op())
        energies = energy_fn(samples)
        energies = tf.stop_gradient(energies)

    with tf.device('/gpu:0'):
        samples_shaped = tf.reshape(samples, (NUM_SAMPLES,)+SYSTEM_SHAPE)
        samples_padded = pad(samples_shaped, SYSTEM_SHAPE, [(K-1)//2]*N_DIMS)
        loss = loss_op(model.factors(samples_padded), energies)
        optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
        train_op = optimizer.minimize(loss)

    return energies, train_op


@scope_op()
def eval_op(sampler, model, energy_fn):
    """
    Perform evaluation of energies.

    Returns
    -------
    energies : tensor of shape (N,)
        Energy of MCMC samples
    """
    with tf.device('/cpu:0'):
        return batched_op(
            energy_fn, sampler.mcmc_op(), ENERGY_BATCH_SIZE)


config = tf.ConfigProto(
    # log_device_placement=True,
    allow_soft_placement=True
)
with tf.Graph().as_default(), tf.Session(config=config) as sess:
    model = CRBM(K, (K-1)//2, ALPHA, N_DIMS)
    # energy_fn = partial(ising_energy, model)
    # num_flips = 1
    energy_fn = partial(heisenberg_energy, model)
    num_flips = 2

    sampler = Sampler(model, SYSTEM_SHAPE, K, NUM_SAMPLES, num_flips)
    eval_sampler = Sampler(model, SYSTEM_SHAPE, K, NUM_EVAL_SAMPLES, num_flips)

    optimize = optimize_op(sampler, model, energy_fn)
    eval_energies = batched_op(
        energy_fn, eval_sampler.mcmc_op(), ENERGY_BATCH_SIZE)

    sess.run(tf.global_variables_initializer())

    # writer = tf.summary.FileWriter('./logs', sess.graph)
    for it in range(OPTIMIZATION_ITS):
        start = time()
        e, _ = sess.run(optimize, feed_dict={
            sampler.new_samples: it == 0
        })
        e = np.real(e)
        print("It %d, E=%.5f (%.2e) %.1fs" %
              (it+1, e.mean(), e.std()/np.sqrt(e.shape[0]), time()-start),
              end="\r")
        stdout.flush()

        if it > 0 and (it+1) % EVAL_FREQ == 0:
            start = time()
            e = sess.run(eval_energies)
            e = np.real(e)
            print("Eval It %d, E=%.5f (%.2e) %.1fs" %
                  (it+1, e.mean(), e.std()/np.sqrt(e.shape[0]), time()-start))

    # writer.flush()
