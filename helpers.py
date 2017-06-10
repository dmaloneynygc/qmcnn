"""Helper graph building ops."""
import numpy as np
import tensorflow as tf
import functools


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
    slice_start = (0,) + pad_size
    slice_size = (-1,)+tuple(size[d]-p*2 for d, p in enumerate(pad_size))
    return tf.slice(x, slice_start, slice_size)


@scope_op()
def pad(x, system_shape, pad_size):
    """Add wrapped padding."""
    dtype = x.dtype
    res = unpad(tf.tile(tf.cast(x, tf.int32), (1,)+(3,)*len(pad_size)),
                tuple(s-p for s, p in zip(system_shape, pad_size)))
    return tf.cast(res, dtype)


@scope_op()
def gather_windows(x, centers, system_shape, window_shape):
    """Gather windows around centers."""
    window_size = np.prod(window_shape)
    batch_size = tf.shape(x)[0]
    index_matrix = tf.constant(create_index_matrix(system_shape, window_shape))
    window_range = tf.range(batch_size, dtype=tf.int32)[:, None] * \
        tf.ones(window_size, dtype=tf.int32)[None, :]
    indices = tf.stack((window_range, tf.gather(index_matrix, centers)), 2)
    return tf.gather_nd(x, indices)


@scope_op()
def update_windows(x, centers, updates, mask, system_shape, window_shape):
    """Update windows around centers at rows where mask is True."""
    window_size = np.prod(window_shape)
    batch_size = tf.shape(x)[0]
    index_matrix = tf.constant(create_index_matrix(system_shape, window_shape))
    window_range = tf.range(batch_size, dtype=tf.int32)[:, None] * \
        tf.ones(window_size, dtype=tf.int32)[None, :]
    indices = tf.stack((window_range, tf.gather(index_matrix, centers)), 2)
    return tf.scatter_nd_update(
        x, tf.boolean_mask(indices, mask), tf.boolean_mask(updates, mask))


@scope_op()
def all_windows(x, system_shape, window_shape):
    """Get all windows."""
    index_matrix = tf.constant(create_index_matrix(system_shape, window_shape))
    return tf.transpose(
        tf.gather_nd(tf.transpose(x),
                     tf.expand_dims(index_matrix, 2)),
        [2, 0, 1])


@scope_op()
def alignedness(states, system_shape):
    """Sum of product of neighbouring spins."""
    num_spins = np.prod(system_shape)
    indices = np.arange(num_spins).reshape(system_shape)
    interactions = []
    for i in range(len(system_shape)):
        shifted = tf.transpose(tf.gather(
            tf.transpose(states), np.roll(indices, 1, i).flatten()))
        interactions.append(tf.reduce_sum(states * shifted, 1))
    return tf.reduce_sum(tf.stack(interactions, 1), 1)
