"""Sampler."""
import numpy as np
from scipy.stats import bernoulli


class Sampler:
    """Model class. 1D Ising, CRBM."""

    THERMFACTOR = 1
    SWEEPFACTOR = 1
    N_SAMPLERS = 100

    def __init__(self, n_spins, model):
        """Initialise."""
        self.n_spins = n_spins
        self.model = model
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
        self.init_sample_state()
        n_sweeps = int(num / self.N_SAMPLERS)
        for _ in range(int(self.THERMFACTOR*self.SWEEPFACTOR*n_sweeps)):
            self.sample_step()

        samples = np.zeros((num, self.n_spins), dtype=np.int8)

        self.moves, self.accepted = 0, 0
        for i in range(n_sweeps):
            for _ in range(int(self.SWEEPFACTOR * self.n_spins)):
                self.sample_step()
            n_sample = i*self.N_SAMPLERS
            samples[n_sample:self.N_SAMPLERS+n_sample, :] = \
                self.sample_states

        return samples
