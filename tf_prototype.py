"""Some tests."""
import tensorflow as tf
import numpy as np

sess = tf.InteractiveSession()

batch_size = 100
n_spins = 40
alpha = 4
init_scale = .01
r = int(2*((n_spins-1)//2)+1)  # Make odd sized
pad_size = int((r-1)/2)
data = np.random.randint(0, 2, size=(batch_size, n_spins))*2-1
data_e = np.random.rand(batch_size)


def random_complex(shape, scale):
    """Random complex valued array."""
    return scale * (np.random.randn(*shape) + 1j * np.random.randn(*shape))


bias_vis = tf.Variable(random_complex([], init_scale),
                       name='bias_vis',
                       dtype=tf.complex64)
bias_hid = tf.Variable(random_complex([alpha], init_scale),
                       name='bias_hid',
                       dtype=tf.complex64)
filters = tf.Variable(random_complex([r, alpha], init_scale),
                      name='filters',
                      dtype=tf.complex64)

state = tf.placeholder(tf.float32, shape=(None, n_spins), name='state')
energy = tf.placeholder(tf.complex64, shape=(None,), name='energy')

tiled = tf.tile(state, [1, 3])
padded = tiled[:, n_spins-pad_size:-n_spins+pad_size]
padded_3d = padded[:, None, None, :, None]
filters_3d = filters[None, None, :, None, :]
filters_3d_re = tf.real(filters_3d)
filters_3d_im = tf.imag(filters_3d)
theta_re = tf.nn.conv3d(
    padded_3d,
    filters_3d_re,
    (1,)*5, 'VALID')
theta_im = tf.nn.conv3d(
    padded_3d,
    filters_3d_im,
    (1,)*5, 'VALID')
theta = tf.complex(theta_re, theta_im)
theta += bias_hid[None, None, None, None, :]
A = tf.log(tf.exp(theta)+tf.exp(-theta))
log_psi = bias_vis * tf.reduce_sum(tf.cast(state, tf.complex64), 1)
log_psi += tf.reduce_sum(A, [1, 2, 3, 4])

num_samples = tf.cast(tf.shape(state)[0], tf.complex64)
energy_avg = tf.reduce_sum(energy)/num_samples
f = tf.reduce_sum(energy[:, None]*log_psi, 0)/num_samples
f -= energy_avg * tf.reduce_sum(log_psi, 0)/num_samples

sess.run(tf.global_variables_initializer())
sess.run([f, tf.gradients(f, [bias_vis, bias_hid, filters])],
         feed_dict={state: data, energy: data_e})
