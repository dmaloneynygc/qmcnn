"""Experiments module."""
import numpy as np
from scipy.stats import bernoulli


class Model:
    """Model class. 1D Ising, CRBM."""

    INIT_PARAM_SCALE = 1E-4
    THERMFACTOR = 0.1
    SWEEPFACTOR = 1

    H = 1

    def __init__(self, n_vis, n_hid):
        """Initialise model."""
        self.n_vis = n_vis
        self.n_hid = n_hid
        self.n_params = (n_vis + 1) * (n_hid + 1) - 1
        self.params = self.INIT_PARAM_SCALE * \
            (np.random.randn(self.n_params) +
             1j * np.random.randn(self.n_params))
        self.bias_vis = self.params[:n_vis]
        self.bias_hid = self.params[n_vis:n_vis+n_hid]
        self.weights = self.params[n_vis+n_hid:].reshape((n_hid, n_vis))

        self.init_sample_state()
        self.moves, self.accepted = 0, 0

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

    def log_psi(self, states, thetas=None):
        """Compute log of wavefn for one or more spin states."""
        if thetas is None:
            thetas = self.theta(states)
        return np.dot(states, self.bias_vis) + \
            np.sum(self.log2cosh(thetas), axis=-1)

    def init_sample_state(self):
        """Random init sample state and lookup tables."""
        self.sample_state = bernoulli.rvs(.5, size=self.n_vis) * 2 - 1
        self.sample_thetas = self.theta(self.sample_state)
        self.sample_log_psi = self.log_psi(
            self.sample_state, self.sample_thetas)

    def sample_step(self):
        """Do Metropolis-Hastings spinflip sampling step."""
        flip = np.random.randint(self.n_vis)
        state_new = self.sample_state.copy()
        state_new[flip] *= -1
        log_psi_new = self.log_psi(state_new)
        A = np.power(np.absolute(np.exp(log_psi_new - self.sample_log_psi)), 2)

        if A > np.random.rand():
            # self.sample_thetas = thetas_new
            self.sample_state = state_new
            self.sample_log_psi = log_psi_new
            self.accepted += 1
        self.moves += 1

    def sample(self, num):
        """Sample num states."""
        print("Thermalising")
        for _ in range(int(num*self.THERMFACTOR*self.SWEEPFACTOR*self.n_vis)):
            self.sample_step()

        samples = np.zeros((num, self.n_vis), dtype=np.int8)

        self.moves, self.accepted = 0, 0
        print("Sampling")
        for i in range(num):
            for _ in range(int(self.SWEEPFACTOR * self.n_vis)):
                self.sample_step()
            samples[i, :] = self.sample_state
        print("Accepted %d / %d = %f %%" %
              (self.accepted, self.moves, self.accepted / self.moves))

        return samples

    def local_energy(self, state):
        """Compute energy for batch of states."""
        flipper = (1-2*np.eye(self.n_vis))[np.newaxis, :, :]
        # S_{nfi}, n: number in batch, f: spin flipped, i: spin in basis
        connected_states = state[:, np.newaxis, :] * flipper
        log_pop = self.log_psi(connected_states) - \
            self.log_psi(state)[:, np.newaxis]

        interaction = np.sum(state * np.roll(state, 1, axis=1), axis=1)
        return -interaction - self.H * np.sum(np.exp(log_pop), axis=1)

    def energy(self, num_its, samples=None):
        """Compute expected energy per spin of quantum state."""
        if samples is None:
            samples = self.sample(num_its)
        print("Computing energy")
        # energies = np.real([self.local_energy(state) for state in samples])
        energies = np.real(self.local_energy(samples))
        energies /= self.n_vis
        return np.mean(energies), np.std(energies)

    def energy_from_sample_file(self, path):
        """Compute energy from Carleo sampled state."""
        samples = np.loadtxt('../Nqs/states.txt', dtype=np.int8)
        return self.energy(None, samples)

    @classmethod
    def from_carleo_params(cls, path):
        """Init model parameters from Carleo format."""
        with open(path, 'rt') as f:
            n_vis = int(f.readline())
            n_hid = int(f.readline())
            n_params = (n_vis + 1) * (n_hid + 1) - 1
            data = np.zeros((n_params, 2))
            for i, line in enumerate(f):
                parts = [float(p) for p in line[1:-2].split(',')]
                data[i, :] = parts

        params = data[:, 0] + 1j * data[:, 1]
        model = cls(n_vis, n_hid)
        model.bias_vis = params[:n_vis]
        model.bias_hid = params[n_vis:n_hid+n_vis]
        model.weights = params[n_vis+n_hid:].reshape((n_hid, n_vis))
        model.init_sample_state()
        return model


path = '../Nqs/Ground/Ising1d_40_1_1.wf'
model = Model.from_carleo_params(path)
model.energy(100)
# model.energy_from_sample_file('../Nqs/states.txt')
