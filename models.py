"""Representation models."""
import numpy as np
import tensorflow as tf


def log2cosh(x):
    """Compute log * 2 * cosh(x) approx sign(re(x))*x for |re x|>>0."""
    re = np.real(x)
    s = np.sign(re)
    threshold = 10
    y = s * x
    selector = re * s < threshold
    y[selector] = np.log(2*np.cosh(x[selector]))
    return y


def random_complex(shape, scale):
    """Random complex valued array."""
    return scale * (np.random.randn(*shape) + 1j * np.random.randn(*shape))


class CRBM():
    """Marginalised CRBM model. Return log psi and derivatives of logs."""

    INIT_PARAM_SCALE = 1E-2

    def __init__(self, n_spins, n_hid):
        """Initialise model."""
        self.n_spins = n_spins
        self.n_hid = n_hid
        self.n_params = (n_spins + 1) * (n_hid + 1) - 1
        self.set_params(self.INIT_PARAM_SCALE *
                        (np.random.randn(self.n_params) +
                         1j * np.random.randn(self.n_params)))

    def set_params(self, params):
        """Set params."""
        self.params = params
        self.bias_vis = self.params[:self.n_spins]
        self.bias_hid = self.params[self.n_spins:self.n_spins+self.n_hid]
        self.weights = self.params[self.n_spins+self.n_hid:].reshape(
            (self.n_hid, self.n_spins))

    def theta(self, states):
        """Compute thetas for one or more spin states."""
        return self.bias_hid + np.dot(states, self.weights.T)

    def forward(self, states):
        """Compute log of wavefn for one or more spin states."""
        thetas = self.theta(states)
        return np.dot(states, self.bias_vis) + \
            np.sum(log2cosh(thetas), axis=-1)

    def backward(self, states):
        """Compute log derivatives from batch of state."""
        t = self.theta(states)
        d_bias_vis = states
        d_bias_hid = t
        d_weights = (t[:, :, np.newaxis] * states[:, np.newaxis, :]).reshape(
            (states.shape[0], -1))
        return np.hstack((d_bias_vis, d_bias_hid, d_weights))

    @classmethod
    def from_carleo_params(cls, path):
        """Init model parameters from Carleo format."""
        with open(path, 'rt') as f:
            n_spins = int(f.readline())
            n_hid = int(f.readline())
            n_params = (n_spins + 1) * (n_hid + 1) - 1
            data = np.zeros((n_params, 2))
            for i, line in enumerate(f):
                parts = [float(p) for p in line[1:-2].split(',')]
                data[i, :] = parts

        params = data[:, 0] + 1j * data[:, 1]
        model = cls(n_spins, n_hid)
        model.set_params(params)
        return model


class ConvCRBM:
    """Marginalised convolutional CRBM model in TF.

    Return log psi and derivatives of logs.
    """

    INIT_PARAM_SCALE = 1E-2
    MAX_BATCH_SIZE = 10000

    def __init__(self, n_spins, alpha, r):
        """Initialise.

        Alpha is number of hidden nodes per spin.
        R is the size of each conv filter.
        dims is number of dimensions.
        """
        self.n_spins = n_spins
        self.n_dims = len(n_spins)
        self.n_spins_3d = n_spins + [1]*(3-self.n_dims)
        self.alpha = alpha
        self.r = r
        self.filter_shape = [r]*self.n_dims+[1]*(3-self.n_dims)+[1, self.alpha]
        self.build_graph()
        self.writer = tf.summary.FileWriter('./logs', self.session.graph)

    def forward(self, states):
        """Compute log of wavefn for one or more spin states."""
        n = states.shape[0]
        results = []
        num_its = int(np.ceil(n/self.MAX_BATCH_SIZE))
        for i in range(num_its):
            batch = states[i*self.MAX_BATCH_SIZE:(i+1)*self.MAX_BATCH_SIZE]
            res = self.session.run(self.log_psi, feed_dict={
                self.state_placeholder: batch})
            results.append(res)
        return np.hstack(results)

    def optimize(self, states, energies):
        """Perform one optimization step."""
        _, sums = self.session.run([self.train_op, self.summaries], feed_dict={
            self.state_placeholder: states,
            self.energy_placeholder: energies})
        self.writer.add_summary(sums)

    def _complex_var(self, value, dtype, name=None):
        """Create complex variable out of 2 reals."""
        re = tf.Variable(np.real(value), name=("%s_re" % name), dtype=dtype)
        im = tf.Variable(np.imag(value), name=("%s_im" % name), dtype=dtype)
        return tf.complex(re, im, name=name)

    def _complex_wrapped_conv(self, x, filters, n_dims):
        """Do complex n-D convolution with periodic bounadry conditions.

        Works through tiling and slicing.
        x: [batch, z, y, x, in_channels]
        filters: [z, y, x, in_channels, out_channels]
        n_dims: the number of spatial dimensions to wrap
        """
        shape = tf.shape(x)
        filter_shape = tf.shape(filters)
        size = shape[1:-1]  # Spatial size
        filter_size = filter_shape[:3]

        # Make periodic boundary conditions by tiling the space
        tiled = tf.tile(x, [1]+[3]*n_dims+[1]*(3-n_dims+1))
        pad_size = tf.to_int32((filter_size-1)/2)

        # Slice s.t. VALID convolution yields original size
        slice_start = [0]+[size[d]-pad_size[d] if d < n_dims else 0
                           for d in range(3)]+[0]
        slice_size = [shape[0]]+[size[d]+2*pad_size[d]
                                 if d < n_dims else size[d]
                                 for d in range(3)]+[shape[4]]
        sliced = tf.slice(tiled, slice_start, slice_size)
        filters_re = tf.real(filters)
        filters_im = tf.imag(filters)
        res_re = tf.nn.conv3d(
            sliced,
            filters_re,
            (1,)*5, 'VALID')
        res_im = tf.nn.conv3d(
            sliced,
            filters_im,
            (1,)*5, 'VALID')
        return tf.complex(res_re, res_im)

    def build_graph(self):
        """Init TF graph."""
        with tf.Graph().as_default():
            # Init placeholders and variables
            self.state_placeholder = tf.placeholder(
                tf.float32, shape=[None]+self.n_spins, name='state')
            self.energy_placeholder = tf.placeholder(
                tf.complex64, shape=[None], name='energy')
            batch_size = tf.shape(self.state_placeholder)[0]
            state_5d = tf.reshape(
                self.state_placeholder, [batch_size]+self.n_spins_3d+[1])

            self.bias_vis = self._complex_var(
                random_complex([], self.INIT_PARAM_SCALE),
                tf.float32, name='bias_vis')
            self.bias_hid = self._complex_var(
                random_complex([self.alpha], self.INIT_PARAM_SCALE),
                tf.float32, name='bias_hid')
            self.filters = self._complex_var(
                random_complex(self.filter_shape, self.INIT_PARAM_SCALE),
                tf.float32, name='filters')

            theta = self._complex_wrapped_conv(
                state_5d,
                self.filters,
                self.n_dims)

            theta += self.bias_hid[None, None, None, None, :]
            A = tf.log(tf.exp(theta)+tf.exp(-theta))
            self.log_psi = self.bias_vis * tf.reduce_sum(  # Bias
                tf.cast(state_5d, tf.complex64), [1, 2, 3, 4])
            self.log_psi += tf.reduce_sum(A, [1, 2, 3, 4])
            n = tf.cast(batch_size, tf.complex64)
            energy_avg = tf.reduce_sum(self.energy_placeholder)/n
            log_psi_conj = tf.conj(self.log_psi)
            loss = tf.reduce_sum(
                self.energy_placeholder*log_psi_conj, 0)/n
            loss -= energy_avg * tf.reduce_sum(log_psi_conj, 0)/n
            self.loss = tf.real(loss)

            optimizer = tf.train.AdamOptimizer(1E-2)
            self.train_op = optimizer.minimize(self.loss)

            energy_mean, energy_var = tf.nn.moments(
                tf.real(self.energy_placeholder), [0])
            tf.summary.scalar('energy_mean', energy_mean)
            tf.summary.scalar('energy_var', energy_var)
            tf.summary.scalar('loss', self.loss)
            self.summaries = tf.summary.merge_all()

            self.session = tf.Session()
            self.session.run(tf.global_variables_initializer())
