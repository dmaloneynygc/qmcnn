"""Test module."""
import tensorflow as tf
import numpy as np
from indices_numpy import create_index_matrix

tf.reset_default_graph()
sess = tf.Session()

shape = (4, 4)
window_shape = (3, 3)
n = np.prod(shape)
n_window = np.prod(window_shape)
data = np.arange(n, dtype=np.int32).reshape(shape)
data_flattened = data.flatten()
index_matrix = tf.constant(create_index_matrix(shape, window_shape))

centers = [0, 5, 15]
batch_size = len(centers)
datas = data_flattened[None, :] + np.arange(batch_size)[:, None] * 100

# %% Select a window from each batch
window_range = tf.constant(
    np.outer(np.arange(batch_size), np.ones(n_window)), dtype=tf.int32)
indices = tf.stack((window_range, tf.gather(index_matrix, centers)), 2)
sess.run(tf.gather_nd(datas, indices))

# %% All windows
datas_t = tf.transpose(datas)
sess.run(tf.transpose(tf.gather_nd(datas_t, tf.expand_dims(index_matrix, 2)), [2, 0, 1]))
