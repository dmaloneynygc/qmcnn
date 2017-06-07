"""Test module."""
import tensorflow as tf
import numpy as np
from indices_numpy import create_index_matrix

tf.reset_default_graph()


def unpad(x, pad_size, n_dims):
    """Unpad tensor. Pad size is the padding in any direction."""
    size = tf.shape(x)[1:]
    slice_start = [0]+[pad_size]*n_dims
    slice_size = [-1]+[size[d]-pad_size*2 for d in range(n_dims)]
    return tf.slice(x, slice_start, slice_size)


def factors(x, n_dims, k, alpha, scale=1E-2):
    """Compute model factors."""
    with tf.variable_scope("factors"):
        bias_vis = tf.get_variable(
            "bias_vis", shape=[2],
            initializer=tf.random_normal_initializer(scale))
        filters = tf.get_variable(
            "filters", shape=[k]*n_dims+[1]+[2*alpha],
            initializer=tf.random_normal_initializer(scale))
        bias_hid = tf.get_variable(
            "bias_hid", shape=[2*alpha],
            initializer=tf.random_normal_initializer(scale))

    x_float = tf.cast(x, tf.float32)
    x_unpad = unpad(x_float, (k-1)/2, n_dims)
    x_expanded = x_float[..., None]

    if n_dims == 1:
        theta = tf.nn.conv1d(x_expanded, filters, 1, 'VALID')+bias_hid
    elif n_dims == 2:
        theta = tf.nn.conv2d(x_expanded, filters, [1]*4, 'VALID')+bias_hid
    theta = tf.complex(theta[..., :alpha], theta[..., alpha:])
    activation = tf.log(tf.exp(theta) + tf.exp(-theta))
    bias = tf.complex(bias_vis[0] * x_unpad, bias_vis[1] * x_unpad)
    return tf.reduce_sum(activation, n_dims+1) + bias


def loss(factor_fn, states, energies):
    """Compute loss."""
    batch_size = tf.shape(states)[0]
    n = tf.cast(batch_size, tf.complex64)
    n_dims = tf.rank(states)-1
    energies = tf.cast(energies, tf.complex64)
    factors = factor_fn(states)
    log_psi = tf.reduce_sum(factors, tf.range(1, n_dims+1))
    log_psi_conj = tf.conj(log_psi)
    energy_avg = tf.reduce_sum(energies)/n
    return tf.reduce_sum(energies*log_psi_conj, 0)/n - \
        energy_avg * tf.reduce_sum(log_psi_conj, 0)/n


def mcmc(states_var, spin_shape, k, batch_size, num_its, factor_fn):
    """Perform num_its MCMC iteratations and return new states."""
    full_window_shape = (k*2-1,)*1
    half_window_shape = (k,)*1
    full_index_matrix = tf.constant(
        create_index_matrix(spin_shape, full_window_shape))
    half_index_matrix = tf.constant(
        create_index_matrix(spin_shape, half_window_shape))
    num_spins = np.prod(spin_shape)
    factors = factor_fn(states_var)

    for _ in range(num_its):
        flip_positions = tf.random_uniform(batch_size, 0, num_spins)
        new_states = tf.Va


with tf.Graph().as_default(), tf.Session() as sess:
    x = np.arange(32).reshape(2, 4, 4)
    e = np.array([10, 15])
    loss_op = loss(lambda x: factors(x, n_dims=2, k=3, alpha=2), x, e)
    # factor_op = factors(x, 2, 3, 2)
    sess.run(tf.global_variables_initializer())
    print(sess.run(loss_op))
