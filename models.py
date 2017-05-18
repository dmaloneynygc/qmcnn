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
    FORWARD_BATCH_SIZE = 1000
    ENERGY_BATCH_SIZE = 1000
    OPTIMIZE_BATCH_SIZE = 1000

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

    def batch(self, fn, X, batch_size):
        """Create mini-batches from X."""
        n = X.shape[0]
        results = []
        num_its = int(np.ceil(n/batch_size))
        for i in range(num_its):
            batch = X[i*batch_size:(i+1)*batch_size]
            res = fn(batch)
            results.append(res)
        return np.concatenate(results)

    def forward(self, states):
        """Compute log of wavefn for batch of half-pad spin states."""
        return self.batch(
            lambda batch: self.session.run(self.log_psi, feed_dict={
                self.state_placeholder: batch}),
            states, self.FORWARD_BATCH_SIZE
        )

    def forward_factors(self, states):
        """Compute log factors for batch of spin states.

        If spin states full-pad, factors half-pad.
        If spin states half-pad, factors unpad.
        """
        result = self.batch(
            lambda batch: self.session.run(self.factors, feed_dict={
                self.state_placeholder: batch}),
            states, self.FORWARD_BATCH_SIZE
        )
        return np.squeeze(result, list(range(self.n_dims+1, 4)))

    def optimize(self, states, energies):
        """Perform one optimization step on half-pad states."""
        batch_size = self.OPTIMIZE_BATCH_SIZE
        n = states.shape[0]
        num_its = int(np.ceil(n/batch_size))
        for i in range(num_its):
            states_batch = states[i*batch_size:(i+1)*batch_size]
            energies_batch = energies[i*batch_size:(i+1)*batch_size]
            if i == num_its-1:
                op = [self.accumulate_gradients, self.train_op, self.reset_op]
            else:
                op = [self.accumulate_gradients]
            self.session.run(op, feed_dict={
                self.state_placeholder: states_batch,
                self.energy_placeholder: energies_batch})

    def _complex_var(self, value, dtype, name=None):
        """Create complex variable out of 2 reals."""
        re = tf.Variable(np.real(value), name=("%s_re" % name), dtype=dtype)
        im = tf.Variable(np.imag(value), name=("%s_im" % name), dtype=dtype)
        return tf.complex(re, im, name=name)

    def pad(self, states, mode):
        """Pad batch wrapped.

        Half: pad s.t. factors are shape of system (pad with (R-1)/2).
        Full: pad s.t. factors cover all dependencies (pad with (R-1)).
        """
        if mode == 'full':
            s = self.r-1
        elif mode == 'half':
            s = (self.r-1)//2
        else:
            raise ValueError('Padding mode not supported')

        return np.pad(states, [(0, 0)]+[(s, s)]*self.n_dims, 'wrap')

    def unpad(self, states, mode):
        """Remove a half-pad or full-pad."""
        if mode == 'full':
            s = self.r-1
        elif mode == 'half':
            s = (self.r-1)//2
        else:
            raise ValueError('Padding mode not supported')
        return states[(Ellipsis,)+(slice(s, -s),)*self.n_dims]

    def build_graph(self):
        """Init TF graph."""
        with tf.Graph().as_default():
            # Init placeholders and variables
            self.state_placeholder = tf.placeholder(
                tf.float32, shape=[None]*(self.n_dims+1), name='state')
            self.energy_placeholder = tf.placeholder(
                tf.complex64, shape=[None], name='energy')
            batch_size = tf.shape(self.state_placeholder)[0]

            state_5d = self.state_placeholder
            for _ in range(4-self.n_dims):
                state_5d = tf.expand_dims(state_5d, -1)
            state_5d_c = tf.cast(state_5d, tf.complex64)

            # Create unpadded state (from half-pad)
            size = tf.shape(self.state_placeholder)[1:]
            slice_start = [0]+[(self.r-1)//2 if d < self.n_dims else 0
                               for d in range(3)]+[0]
            slice_size = [-1]+[size[d]-(self.r-1)
                               if d < self.n_dims else -1
                               for d in range(3)]+[-1]
            state_5d_c_unpadded = tf.slice(state_5d_c, slice_start, slice_size)

            self.bias_vis = self._complex_var(
                random_complex([], self.INIT_PARAM_SCALE),
                tf.float32, name='bias_vis')
            self.bias_hid = self._complex_var(
                random_complex([self.alpha], self.INIT_PARAM_SCALE),
                tf.float32, name='bias_hid')
            self.filters_re = tf.Variable(
                np.random.randn(*self.filter_shape).astype(np.float32) *
                self.INIT_PARAM_SCALE, name='filters_re')
            self.filters_im = tf.Variable(
                np.random.randn(*self.filter_shape).astype(np.float32) *
                self.INIT_PARAM_SCALE, name='filters_im')
            theta_re = tf.nn.conv3d(
                state_5d,
                self.filters_re,
                (1,)*5, 'VALID')
            theta_im = tf.nn.conv3d(
                state_5d,
                self.filters_im,
                (1,)*5, 'VALID')
            theta = tf.complex(theta_re, theta_im)

            theta += self.bias_hid[None, None, None, None, :]
            A = tf.log(tf.exp(theta)+tf.exp(-theta))
            self.factors = tf.reduce_sum(
                A + self.bias_vis * state_5d_c_unpadded, 4)
            self.log_psi = tf.reduce_sum(self.factors, [1, 2, 3])

            n = tf.cast(batch_size, tf.complex64)
            energy_avg = tf.reduce_sum(self.energy_placeholder)/n
            log_psi_conj = tf.conj(self.log_psi)
            loss = tf.reduce_sum(
                self.energy_placeholder*log_psi_conj, 0)/n
            loss -= energy_avg * tf.reduce_sum(log_psi_conj, 0)/n
            self.loss = tf.real(loss)

            optimizer = tf.train.AdamOptimizer(3E-3)
            self.train_op = optimizer.minimize(self.loss)

            self.session = tf.Session()
            self.session.run(tf.global_variables_initializer())

            vars = tf.trainable_variables()
            vars_zeros = []
            grad_accums = []
            for v in vars:
                shape = self.session.run(tf.shape(v))
                zeros = np.zeros(shape)
                vars_zeros.append(zeros)
                grad_accums.append(tf.Variable(
                    zeros, trainable=False, dtype=v.dtype.base_dtype))

            gradients = optimizer.compute_gradients(self.loss, vars)
            for i, (grad, var) in enumerate(gradients):
                grad_accums[i] = tf.assign_add(grad_accums[i], grad)
            self.accumulate_gradients = grad_accums

            with tf.control_dependencies(grad_accums):
                self.train_op = optimizer.apply_gradients(
                    list(zip(grad_accums, vars)))
                with tf.control_dependencies([self.train_op]):
                    reset_ops = []
                    for i, acc in enumerate(grad_accums):
                        reset_ops.append(tf.assign(acc, vars_zeros[i]))
                    self.reset_op = reset_ops

            self.session.run(tf.global_variables_initializer())
            energy_mean, energy_var = tf.nn.moments(
                tf.real(self.energy_placeholder), [0])
            tf.summary.scalar('energy_mean', energy_mean)
            tf.summary.scalar('energy_var', energy_var)
            tf.summary.scalar('loss', self.loss)
            self.summaries = tf.summary.merge_all()
