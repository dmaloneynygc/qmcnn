"""Test module."""
from __future__ import division
import tensorflow as tf
import numpy as np
import functools


tf.reset_default_graph()

LEARNING_RATE = 1E-1
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

# NUM_SAMPLES = 4
# NUM_SAMPLERS = 2
NUM_SAMPLES = 10000
NUM_SAMPLERS = 1000
ITS_PER_SAMPLE = NUM_SPINS
SAMPLES_PER_SAMPLER = NUM_SAMPLES // NUM_SAMPLERS
THERM_ITS = SAMPLES_PER_SAMPLER // 2
SAMPLE_ITS = THERM_ITS + (SAMPLES_PER_SAMPLER-1) * ITS_PER_SAMPLE + 1
OPTIMIZATION_ITS = 10000


def create_index_matrix(data_shape, window_shape):
    """Return wrapped index matrix, shape (n, n_windows)."""
    n_data = np.prod(data_shape)
    n_window = np.prod(window_shape)
    box = np.indices(window_shape)
    index_matrix = np.zeros((n_data, n_window), dtype=np.int32)
    shifts = np.unravel_index(np.arange(n_data), data_shape)
    offset = (np.array(window_shape)-1)//2
    for i, shift in enumerate(zip(*shifts)):
        shift = np.array(shift)-offset
        window = (box.T + shift).T
        index_matrix[i] = np.ravel_multi_index(window, data_shape, 'wrap') \
            .flatten()
    return index_matrix


def scope_op(name=None):
    """Put tensorflow op in name_scope."""
    def decorator(function):
        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            with tf.name_scope(name or function.__name__):
                return function(*args, **kwargs)
        return wrapper
    return decorator


@scope_op()
def unpad(x, pad_size):
    """Unpad tensor. Pad size is the padding in any direction."""
    size = tf.shape(x)[1:]
    if isinstance(pad_size, int):
        pad_size = [pad_size]*N_DIMS
    slice_start = [0] + pad_size
    slice_size = [-1]+[size[d]-pad_size[d]*2 for d in range(N_DIMS)]
    return tf.slice(x, slice_start, slice_size)


@scope_op()
def pad(x, pad_size):
    """Add wrapped padding."""
    dtype = x.dtype
    res = unpad(tf.tile(tf.cast(x, tf.int32), [1]+[3]*N_DIMS),
                [s-pad_size for s in SYSTEM_SHAPE])
    return tf.cast(res, dtype)


@scope_op()
def gather_windows(x, centers, window_shape):
    """Gather windows around centers."""
    window_size = np.prod(window_shape)
    batch_size = tf.shape(x)[0]
    index_matrix = tf.constant(create_index_matrix(SYSTEM_SHAPE, window_shape))
    window_range = tf.range(batch_size, dtype=tf.int32)[:, None] * \
        tf.ones(window_size, dtype=tf.int32)[None, :]
    indices = tf.stack((window_range, tf.gather(index_matrix, centers)), 2)
    return tf.gather_nd(x, indices)


@scope_op()
def update_windows(x, centers, updates, mask, window_shape):
    """Update windows around centers at rows where mask is True."""
    window_size = np.prod(window_shape)
    batch_size = tf.shape(x)[0]
    index_matrix = tf.constant(create_index_matrix(SYSTEM_SHAPE, window_shape))
    window_range = tf.range(batch_size, dtype=tf.int32)[:, None] * \
        tf.ones(window_size, dtype=tf.int32)[None, :]
    indices = tf.stack((window_range, tf.gather(index_matrix, centers)), 2)
    return tf.scatter_nd_update(
        x, tf.boolean_mask(indices, mask), tf.boolean_mask(updates, mask))


@scope_op()
def all_windows(x, window_shape):
    """Get all windows."""
    index_matrix = tf.constant(create_index_matrix(SYSTEM_SHAPE, window_shape))
    return tf.transpose(
        tf.gather_nd(tf.transpose(x),
                     tf.expand_dims(index_matrix, 2)),
        [2, 0, 1])


@scope_op()
def alignedness(states):
    """Sum of product of neighbouring spins."""
    indices = np.arange(NUM_SPINS).reshape(SYSTEM_SHAPE)
    interactions = []
    for i in range(N_DIMS):
        shifted = tf.transpose(tf.gather(
            tf.transpose(states), np.roll(indices, 1, i).flatten()))
        interactions.append(tf.reduce_sum(states * shifted, 1))
    return tf.reduce_sum(tf.stack(interactions, 1), 1)


@scope_op()
def create_vars():
    """Add vars to graph."""
    with tf.variable_scope("factors"):
        tf.get_variable(
            "filters", shape=[K]*N_DIMS+[1]+[2*ALPHA],
            initializer=tf.random_normal_initializer(0., SCALE))
        tf.get_variable(
            "bias_vis", shape=[2],
            initializer=tf.random_normal_initializer(0., SCALE))
        tf.get_variable(
            "bias_hid", shape=[2*ALPHA],
            initializer=tf.random_normal_initializer(0., SCALE))

    with tf.variable_scope("sampler"):
        tf.get_variable("current_samples",
                        shape=[NUM_SAMPLERS, NUM_SPINS],
                        initializer=tf.constant_initializer(0),
                        dtype=tf.int8, trainable=False)
        tf.get_variable("current_factors",
                        shape=[NUM_SAMPLERS, NUM_SPINS],
                        initializer=tf.constant_initializer(0),
                        dtype=tf.complex64, trainable=False)
        tf.get_variable("samples",
                        shape=[SAMPLES_PER_SAMPLER, NUM_SAMPLERS, NUM_SPINS],
                        initializer=tf.constant_initializer(0),
                        dtype=tf.int8, trainable=False)
        tf.get_variable("flip_positions", shape=[SAMPLE_ITS, NUM_SAMPLERS],
                        initializer=tf.constant_initializer(0),
                        dtype=tf.int32, trainable=False)
        tf.get_variable("accept_sample", shape=[SAMPLE_ITS, NUM_SAMPLERS],
                        initializer=tf.constant_initializer(0.),
                        dtype=tf.float32, trainable=False)


@scope_op()
def factors_op(x):
    """Compute model factors."""
    with tf.variable_scope("factors", reuse=True):
        filters = tf.get_variable("filters")
        bias_vis = tf.get_variable("bias_vis")
        bias_hid = tf.get_variable("bias_hid")

    x_float = tf.cast(x, tf.float32)
    x_unpad = unpad(x_float, (K-1)//2)
    x_expanded = x_float[..., None]

    if N_DIMS == 1:
        theta = tf.nn.conv1d(x_expanded, filters, 1, 'VALID')+bias_hid
    elif N_DIMS == 2:
        theta = tf.nn.conv2d(x_expanded, filters, [1]*4, 'VALID')+bias_hid

    theta = tf.complex(theta[..., :ALPHA], theta[..., ALPHA:])
    activation = tf.log(tf.exp(theta) + tf.exp(-theta))
    bias = tf.complex(bias_vis[0] * x_unpad, bias_vis[1] * x_unpad)
    return tf.reduce_sum(activation, N_DIMS+1) + bias


@scope_op()
def loss_op(factors, energies):
    """Compute loss."""
    batch_size = tf.shape(factors)[0]
    n = tf.cast(batch_size, tf.complex64)
    energies = tf.cast(energies, tf.complex64)
    log_psi = tf.reduce_sum(factors, range(1, N_DIMS+1))
    log_psi_conj = tf.conj(log_psi)
    energy_avg = tf.reduce_sum(energies)/n
    loss = tf.reduce_sum(energies*log_psi_conj, 0)/n - \
        energy_avg * tf.reduce_sum(log_psi_conj, 0)/n
    return tf.real(loss)


@scope_op()
def energy_op(states):
    """Compute local energy of states."""
    batch_size = tf.shape(states)[0]
    states_shaped = tf.reshape(states, (batch_size,)+SYSTEM_SHAPE)
    factors = tf.reshape(factors_op(pad(states_shaped, K-1)), (batch_size, -1))
    factor_windows = all_windows(factors, HALF_WINDOW_SHAPE)
    spin_windows = all_windows(states, FULL_WINDOW_SHAPE)
    flipper = np.ones(FULL_WINDOW_SIZE, dtype=np.int8)
    flipper[(FULL_WINDOW_SIZE-1)//2] = -1
    spins_flipped = spin_windows * flipper
    factors_flipped = tf.reshape(
        factors_op(tf.reshape(
            spins_flipped, (batch_size*NUM_SPINS,)+FULL_WINDOW_SHAPE)),
        (batch_size, NUM_SPINS, -1))

    log_pop = tf.reduce_sum(factors_flipped - factor_windows, 2)
    energy = -H * tf.reduce_sum(tf.exp(log_pop), 1) - \
        tf.cast(alignedness(states), tf.complex64)
    return energy / NUM_SPINS


@scope_op()
def mcmc_reset():
    """Reset MCMC variables."""
    with tf.variable_scope('sampler', reuse=True):
        current_samples = tf.get_variable('current_samples', dtype=tf.int8)
        current_factors = tf.get_variable('current_factors',
                                          dtype=tf.complex64)
        samples = tf.get_variable('samples', dtype=tf.int8)
        flip_positions = tf.get_variable('flip_positions', dtype=tf.int32)
        accept_sample = tf.get_variable('accept_sample', dtype=tf.float32)

    states = tf.random_uniform(
        [NUM_SAMPLERS, NUM_SPINS], 0, 2, dtype=tf.int32)*2-1
    states = tf.cast(states, tf.int8)
    states_shaped = tf.reshape(states, (NUM_SAMPLERS,)+SYSTEM_SHAPE)
    factors = tf.reshape(
        factors_op(pad(states_shaped, (K-1)//2)),
        (NUM_SAMPLERS, -1))

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
def mcmc_step(i):
    """Do MCMC Step."""
    with tf.variable_scope('sampler', reuse=True):
        current_samples = tf.get_variable('current_samples', dtype=tf.int8)
        current_factors = tf.get_variable(
            'current_factors', dtype=tf.complex64)
        samples = tf.get_variable('samples', dtype=tf.int8)
        flip_positions = tf.get_variable('flip_positions', dtype=tf.int32)
        accept_sample = tf.get_variable('accept_sample', dtype=tf.float32)

    centers = flip_positions[i]
    spin_windows = gather_windows(
        current_samples, centers, FULL_WINDOW_SHAPE)
    factor_windows = gather_windows(
        current_factors, centers, HALF_WINDOW_SHAPE)

    flipper = np.ones(FULL_WINDOW_SIZE, dtype=np.int32)
    flipper[(FULL_WINDOW_SIZE-1)//2] = -1
    flipped_spins = spin_windows * flipper
    flipped_factors = tf.reshape(
        factors_op(
            tf.reshape(flipped_spins, (NUM_SAMPLERS,)+FULL_WINDOW_SHAPE)),
        (NUM_SAMPLERS, -1))
    accept_prob = tf.pow(tf.abs(tf.exp(
        tf.reduce_sum(flipped_factors - factor_windows, 1))), 2)
    mask = accept_prob > accept_sample[i]
    current_samples = update_windows(
        current_samples, centers, flipped_spins, mask, FULL_WINDOW_SHAPE)
    current_factors = update_windows(
        current_factors, centers, flipped_factors, mask, HALF_WINDOW_SHAPE)

    def write_samples():
        """Write current_samples to samples."""
        with tf.variable_scope('sampler', reuse=True):
            current_samples = tf.get_variable('current_samples', dtype=tf.int8)
            samples = tf.get_variable('samples', dtype=tf.int8)
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
def mcmc_op():
    """Get MCMC Samples."""
    with tf.control_dependencies([mcmc_reset()]):
        loop = tf.while_loop(
            lambda i: i < SAMPLE_ITS,
            mcmc_step,
            [tf.constant(0)],
            parallel_iterations=1,
            back_prop=False
        )
    with tf.control_dependencies([loop]):
        with tf.variable_scope('sampler', reuse=True):
            samples = tf.get_variable('samples', dtype=tf.int8)
        return tf.reshape(samples, [NUM_SAMPLES, NUM_SPINS])


@scope_op()
def optimize_op():
    """Perform optimization iteration."""
    samples = tf.stop_gradient(mcmc_op())
    samples_shaped = pad(
        tf.reshape(samples, (NUM_SAMPLES,)+SYSTEM_SHAPE), (K-1)//2)
    energies = tf.stop_gradient(energy_op(samples))
    loss = loss_op(factors_op(samples_shaped), energies)
    # optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE)
    optimizer = tf.train.AdamOptimizer(1E-3)

    return energies, optimizer.minimize(loss), [tf.shape(samples_shaped), tf.shape(factors_op(samples_shaped))]


with tf.Graph().as_default(), tf.Session() as sess:
    create_vars()
    op = optimize_op()
    sess.run(tf.global_variables_initializer())

    # x = tf.random_uniform((BATCH_SIZE, NUM_SPINS), 0, 2, dtype=tf.int32)*2-1
    # x = tf.cast(x, tf.int8)
    from time import time
    for it in range(1 or OPTIMIZATION_ITS):
        start = time()
        e, _, d = sess.run(op)
        print(d)
        e = np.real(e)
        print("It %d, E=%.5f (%.2e) %.1fs" %
              (it+1, e.mean(), e.std(), time()-start))
