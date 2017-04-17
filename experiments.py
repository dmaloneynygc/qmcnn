"""Experiments module."""
import numpy as np
from scipy.stats import bernoulli


class Model():
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

    def log2cosh(self, x):
        """Compute log * 2 * cosh(x) approx sign(re(x))*x for |re x|>>0."""
        re = np.real(x)
        s = np.sign(re)
        threshold = 10
        y = s * x
        selector = re * s < threshold
        y[selector] = np.log(2*np.cosh(x[selector]))
        return y

    def theta(self, states):
        """Compute thetas for one or more spin states."""
        return self.bias_hid + np.dot(states, self.weights.T)

    def forward(self, states):
        """Compute log of wavefn for one or more spin states."""
        thetas = self.theta(states)
        return np.dot(states, self.bias_vis) + \
            np.sum(self.log2cosh(thetas), axis=-1)

    def backward(self, states):
        """Compute log derivatives from batch of state."""
        t = self.theta(states)
        d_bias_vis = states
        d_bias_hid = t
        d_weights = (t[:, :, np.newaxis] * states[:, np.newaxis, :]).reshape(
            (states.shape[0], -1))
        return np.hstack(d_bias_vis, d_bias_hid, d_weights)

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


class Ising1D:
    """1D Transverse field Ising model."""

    H = 1

    def __init__(self, n_spins, model):
        """Initialise."""
        self.n_spins = n_spins
        self.model = model

    def local_energy(self, state):
        """Compute energy for batch of states."""
        flipper = (1-2*np.eye(self.n_spins))[np.newaxis, :, :]
        # S_{nfi}, n: number in batch, f: spin flipped, i: spin in basis
        connected_states = state[:, np.newaxis, :] * flipper
        log_pop = self.model.forward(connected_states) - \
            self.model.forward(state)[:, np.newaxis]

        interaction = np.sum(state * np.roll(state, 1, axis=1), axis=1)
        return -interaction - self.H * np.sum(np.exp(log_pop), axis=1)


class Sampler:
    """Model class. 1D Ising, CRBM."""

    THERMFACTOR = 0.1
    SWEEPFACTOR = 1
    N_SAMPLERS = 100

    def __init__(self, n_spins, model):
        """Initialise."""
        self.n_spins = n_spins
        self.model = model
        self.system = Ising1D(n_spins, model)
        self.init_sample_state()
        self.moves, self.accepted = 0, 0

    def init_sample_state(self):
        """Random init sample state and lookup tables."""
        self.sample_states = -1+2*bernoulli.rvs(
            .5, size=(self.N_SAMPLERS, self.n_spins))
        self.sample_log_psis = self.model.forward(self.sample_states)

    def sample_step(self):
        """Do Metropolis-Hastings spinflip sampling step."""
        spinflips = np.random.randint(self.n_spins, size=self.N_SAMPLERS)
        states_new = self.sample_states.copy()
        states_new[np.arange(self.N_SAMPLERS), spinflips] *= -1
        log_psis_new = self.model.forward(states_new)
        log_pop = log_psis_new - self.sample_log_psis
        A = np.power(np.absolute(np.exp(log_pop)), 2)

        accepted = A > np.random.rand(self.N_SAMPLERS)
        self.sample_states[accepted] = states_new[accepted]
        self.sample_log_psis[accepted] = log_psis_new[accepted]
        self.moves += self.N_SAMPLERS
        self.accepted += np.sum(accepted)

    def sample(self, num):
        """Sample num states."""
        print("Thermalising")
        n_sweeps = int(num / self.N_SAMPLERS)
        for _ in range(int(self.THERMFACTOR*self.SWEEPFACTOR*n_sweeps)):
            self.sample_step()

        samples = np.zeros((num, self.n_spins), dtype=np.int8)

        self.moves, self.accepted = 0, 0
        print("Sampling")
        for i in range(n_sweeps):
            for _ in range(int(self.SWEEPFACTOR * self.n_spins)):
                self.sample_step()
            n_sample = i*self.N_SAMPLERS
            samples[n_sample:self.N_SAMPLERS+n_sample, :] = \
                self.sample_states
        print("Accepted %d / %d = %f %%" %
              (self.accepted, self.moves, self.accepted / self.moves))

        return samples

    def energy(self, num_its, samples=None):
        """Compute expected energy per spin of quantum state."""
        if samples is None:
            samples = self.sample(num_its)
        energies = np.real(self.system.local_energy(samples))
        return energies/self.n_spins

    def energy_from_sample_file(self, path):
        """Compute energy from Carleo sampled state."""
        samples = np.loadtxt('../Nqs/states.txt', dtype=np.int8)
        return self.energy(None, samples)


path = '../Nqs/Ground/Ising1d_40_1_1.wf'
model = Model.from_carleo_params(path)
sampler = Sampler(model.n_spins, model)
sampler.energy(10000).mean()
# sampler.energy_from_sample_file('../Nqs/states.txt').mean()
