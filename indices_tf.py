"""Test module."""
import tensorflow as tf
import numpy as np
from indices_numpy import create_index_matrix

tf.reset_default_graph()
sess = tf.Session()

shape = (4, 4)
window_shape = (3, 3)
n_dims = len(shape)
n = np.prod(shape)
n_window = np.prod(window_shape)
data = np.arange(n, dtype=np.int32).reshape(shape)
data_flattened = data.flatten()
index_matrix = tf.constant(create_index_matrix(shape, window_shape),
                           dtype=tf.int32)

centers = [0, 5, 15]
batch_size = len(centers)
datas = data_flattened[None, :] + np.arange(batch_size)[:, None] * 100
datas = tf.constant(datas, np.int32)

# %% Select a window from each batch
window_range = tf.constant(
    np.outer(np.arange(batch_size), np.ones(n_window)), dtype=tf.int32)
indices = tf.stack((window_range, tf.gather(index_matrix, centers)), 2)
sess.run(tf.gather_nd(datas, indices)).reshape((batch_size,)+window_shape)

# %% All windows
datas_t = tf.transpose(datas)
sess.run(tf.transpose(tf.gather_nd(datas_t, tf.expand_dims(index_matrix, 2)),
                      [2, 0, 1])).reshape((batch_size, n)+window_shape)

# %% Update windows
window_range = tf.constant(
    np.outer(np.arange(batch_size), np.ones(n_window)), dtype=tf.int32)
indices = tf.stack((window_range, tf.gather(index_matrix, centers)), 2)
windows = tf.gather_nd(datas, indices)
datas_var = tf.Variable(datas)
datas_var = tf.scatter_nd_update(datas_var, indices, -windows)
sess.run(tf.global_variables_initializer())
sess.run(datas_var).reshape((batch_size,)+shape)

# %% Conditional update
window_range = tf.constant(
    np.outer(np.arange(batch_size), np.ones(n_window)), dtype=tf.int32)
indices = tf.stack((window_range, tf.gather(index_matrix, centers)), 2)
windows = tf.gather_nd(datas, indices)
datas_var = tf.Variable(datas)
mask = tf.reduce_sum(windows, 1) > 1000
datas_var = tf.scatter_nd_update(
    datas_var, tf.boolean_mask(indices, mask), -tf.boolean_mask(windows, mask))
sess.run(tf.global_variables_initializer())
sess.run(datas_var).reshape((batch_size,)+shape)

# %% Spinflip
window_range = tf.constant(
    np.outer(np.arange(batch_size), np.ones(n_window)), dtype=tf.int32)
indices = tf.stack((window_range, tf.gather(index_matrix, centers)), 2)
windows = tf.cast(tf.gather_nd(datas, indices), tf.int32)
flipper = np.ones(n_window, dtype=np.int32)
flipper[(n_window-1)/2] = -1
flipper = tf.constant(flipper)
sess.run(windows * flipper).reshape((batch_size,)+window_shape)


# %% Loop of flips
sess = tf.Session()
with tf.variable_scope('iteration'):
    datas_var = tf.get_variable(
        'datas_var', initializer=datas, trainable=False)
sess.run(tf.global_variables_initializer())
num_its = 3
centers = tf.random_uniform([num_its, batch_size], 0, n, dtype=tf.int32)


def body(i):
    """Loop body."""
    indices = tf.stack((np.arange(batch_size), centers[i]), 1)
    with tf.variable_scope('iteration', reuse=True):
        datas_var = tf.get_variable('datas_var', dtype=np.int32)
    assign_op = tf.scatter_nd_update(datas_var, indices, -np.ones(batch_size))
    with tf.control_dependencies([assign_op]):
        return i+1


loop = tf.while_loop(
    lambda i: i < num_its,
    body,
    [tf.constant(0)],
    parallel_iterations=1,
    back_prop=False
)
sess.run([centers, loop, datas_var])

# %% alignedness
shape = (3, 3)
n_dims = len(shape)
n = np.prod(shape)
datas = tf.random_uniform([batch_size, n], 0, 2, dtype=tf.int32)*2-1
indices = np.arange(n).reshape(shape)
interactions = []
for i in range(n_dims):
    shifter = np.roll(indices, 1, i)
    shifted = tf.transpose(tf.gather(tf.transpose(datas), shifter.flatten()))
    interactions.append(tf.reduce_sum(datas * shifted, 1))
alignedness = tf.reduce_sum(tf.stack(interactions, 1), 1)
d, a = sess.run([datas, alignedness])
print(d.reshape((batch_size,)+shape))
print(a)
