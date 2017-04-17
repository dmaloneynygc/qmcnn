"""Representation models."""
import numpy as np


def log2cosh(x):
    """Compute log * 2 * cosh(x) approx sign(re(x))*x for |re x|>>0."""
    re = np.real(x)
    s = np.sign(re)
    threshold = 10
    y = s * x
    selector = re * s < threshold
    y[selector] = np.log(2*np.cosh(x[selector]))
    return y


class CRBM():
    """Marginalised CRBM model. Return log psi and derivatives of logs."""

    INIT_PARAM_SCALE = 1E-4

    def __init__(self, n_spins, n_hid):
        """Initialise model."""
        self.n_spins = n_spins
        self.n_hid = n_hid
        self.n_params = (n_spins + 1) * (n_hid + 1) - 1
        self.params = self.INIT_PARAM_SCALE * \
            (np.random.randn(self.n_params) +
             1j * np.random.randn(self.n_params))
        self.bias_vis = self.params[:n_spins]
        self.bias_hid = self.params[n_spins:n_spins+n_hid]
        self.weights = self.params[n_spins+n_hid:].reshape((n_hid, n_spins))

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
        model.bias_vis = params[:n_spins]
        model.bias_hid = params[n_spins:n_hid+n_spins]
        model.weights = params[n_spins+n_hid:].reshape((n_hid, n_spins))
        return model
