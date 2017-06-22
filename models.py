"""Factor models."""
import tensorflow as tf
from helpers import unpad, scope_op


class CRBM(object):
    """Convolutional marginalised RBM."""

    SCALE = 1E-2

    def __init__(self, k, alpha, n_dims):
        """Initialise."""
        self.alpha = alpha
        self.k = k
        self.n_dims = n_dims

        self.get_variables()

    def get_variables(self):
        """Get variables usedl by model."""
        with tf.variable_scope("factors"):
            return [
                tf.get_variable(
                    "filters", shape=[self.k]*self.n_dims+[1]+[2*self.alpha],
                    dtype=tf.float32,
                    initializer=tf.random_normal_initializer(0., self.SCALE)),
                tf.get_variable(
                    "bias_vis", shape=[2], dtype=tf.float32,
                    initializer=tf.random_normal_initializer(0., self.SCALE)),
                tf.get_variable(
                    "bias_hid", shape=[2*self.alpha], dtype=tf.float32,
                    initializer=tf.random_normal_initializer(0., self.SCALE))
            ]

    @scope_op('crbm-factors')
    def factors(self, x):
        """
        Compute model factors.

        Performs 'VALID' convolution. Output is (K-1)//2 smaller in both
        directions in all dimensions.

        Parameters
        ----------
        x : Tensor of shape (N,) + system_shape

        Returns
        -------
        factors : tensor of shape (N,) + unpadded shape of x
        """
        with tf.variable_scope("factors", reuse=True):
            filters = tf.get_variable("filters")
            bias_vis = tf.get_variable("bias_vis")
            bias_hid = tf.get_variable("bias_hid")

        x_float = tf.cast(x, tf.float32)
        x_unpad = unpad(x_float, ((self.k-1)//2,)*self.n_dims)
        x_expanded = x_float[..., None]

        with tf.device('/gpu:0'):
            if self.n_dims == 1:
                theta = tf.nn.conv1d(x_expanded, filters, 1, 'VALID')
            elif self.n_dims == 2:
                theta = tf.nn.conv2d(x_expanded, filters, [1]*4, 'VALID')
            elif self.n_dims == 3:
                theta = tf.nn.conv3d(x_expanded, filters, [1]*5, 'VALID')
            theta = theta + bias_hid

        theta = tf.complex(theta[..., :self.alpha], theta[..., self.alpha:])
        activation = tf.log(tf.exp(theta) + tf.exp(-theta))
        bias = tf.complex(bias_vis[0] * x_unpad, bias_vis[1] * x_unpad)
        return tf.reduce_sum(activation, self.n_dims+1) + bias


class DCRBM:
    """Deep Convolutional marginalised RBM."""

    SCALE = 1E-2

    def __init__(self, k, layers, n_dims):
        """Initialise."""
        self.layers = layers
        l = [1]+layers
        self.shapes = zip(l, l[1:])
        self.k = k
        self.n_dims = n_dims

    def get_variables(self):
        """Get variables usedl by model."""
        vars = []
        shape = [self.k]*self.n_dims
        with tf.variable_scope("factors"):
            for l, (in_, out) in enumerate(self.shapes):
                filters = tf.get_variable(
                    "filters_%d" % l, shape=shape + [in_, out],
                    dtype=tf.float32,
                    initializer=tf.random_normal_initializer(0., self.SCALE)),
                bias = tf.get_variable(
                    "bias_%d" % l, shape=[out], dtype=tf.float32,
                    initializer=tf.random_normal_initializer(0., self.SCALE))
                vars += [filters, bias]
        return vars

    @scope_op('dcrbm-factors')
    def factors(self, x):
        """
        Compute model factors.

        Performs 'VALID' convolution. Output is (K-1)//2 smaller in both
        directions in all dimensions.

        Parameters
        ----------
        x : Tensor of shape (N,) + system_shape

        Returns
        -------
        factors : tensor of shape (N,) + unpadded shape of x
        """
        x = tf.cast(x, tf.float32)[..., None]

        with tf.device('/gpu:0'):
            for l in range(len(self.layers)):
                with tf.variable_scope("factors", reuse=True):
                    filters = tf.get_variable("filters_%d" % l)
                    bias = tf.get_variable("bias_%d" % l)

                if self.n_dims == 1:
                    x = tf.nn.conv1d(x, filters, 1, 'VALID')
                elif self.n_dims == 2:
                    x = tf.nn.conv2d(x, filters, [1]*4, 'VALID')
                elif self.n_dims == 3:
                    x = tf.nn.conv3d(x, filters, [1]*5, 'VALID')
                x = x + bias
                if l != len(self.layers)-1:
                    x = tf.tanh(x)

        sep = self.layers[-1]//2
        theta = tf.complex(x[..., :sep], x[..., sep:])
        act = tf.log(tf.exp(theta) + tf.exp(-theta))
        return tf.reduce_sum(act, self.n_dims+1)
        # re = tf.reduce_sum(x[..., :sep], self.n_dims+1)
        # im = tf.reduce_sum(x[..., sep:], self.n_dims+1)
        # return tf.complex(re, im)
