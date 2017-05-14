"""Sampler."""
import numpy as np
from scipy.stats import bernoulli
from utils.utils import flip_wrapped, replace_wrapped, window


class Sampler:
    """Model class. 1D Ising, CRBM."""

    THERMFACTOR = 1
    SWEEPFACTOR = 1
    N_SAMPLERS = 100

    def __init__(self, n_spins, model):
        """Initialise."""
        self.n_spins = n_spins
        self.n_dims = len(n_spins)
        self.total_spins = np.prod(n_spins)
        self.model = model
        self.init_sample_state()
        self.moves, self.accepted = 0, 0

    def init_sample_state(self):
        """Random init sample state and lookup tables."""
        sample_states = -1+2*bernoulli.rvs(
            .5, size=[self.N_SAMPLERS]+self.n_spins).astype(np.int8)
        self.sample_states_padded = self.model.pad(sample_states, 'full')
        self.sample_factors_padded = self.model.forward_factors(
            self.sample_states_padded)

    def sample_step(self):
        """Do Metropolis-Hastings spinflip sampling step."""
        spinflips = np.transpose(
            [np.random.randint(s, size=self.N_SAMPLERS) for s in self.n_spins])
        window_size = np.array([self.model.r]*self.n_dims)
        states_new_padded = self.sample_states_padded.copy()
        flip_wrapped(states_new_padded, window_size-1, spinflips)
        spin_windows = window(states_new_padded, window_size*2-1, spinflips)
        factor_windows = self.model.forward_factors(spin_windows)
        old_factor_windows = window(
            self.sample_factors_padded, window_size, spinflips)
        log_pop = np.sum(factor_windows, tuple(range(1, self.n_dims+1))) - \
            np.sum(old_factor_windows, tuple(range(1, self.n_dims+1)))
        A = np.power(np.absolute(np.exp(log_pop)), 2)

        accepted = A > np.random.rand(self.N_SAMPLERS)
        self.sample_states_padded[accepted] = states_new_padded[accepted]

        new_factors = self.sample_factors_padded[accepted].copy()
        replace_wrapped(new_factors, factor_windows[accepted],
                        spinflips[accepted])
        np.set_printoptions(precision=3)
        self.sample_factors_padded[accepted] = new_factors
        self.moves += self.N_SAMPLERS
        self.accepted += np.sum(accepted)

    def sample(self, num):
        """Sample num states."""
        self.init_sample_state()
        n_sweeps = int(num / self.N_SAMPLERS)
        for _ in range(int(self.THERMFACTOR*self.SWEEPFACTOR*n_sweeps)):
            self.sample_step()

        padded_size = [d+(self.model.r-1)*2 for d in self.n_spins]
        samples = np.zeros([num]+padded_size, dtype=np.int8)

        self.moves, self.accepted = 0, 0
        for i in range(n_sweeps):
            for _ in range(int(self.SWEEPFACTOR * self.total_spins)):
                self.sample_step()
            n_sample = i*self.N_SAMPLERS
            samples[n_sample:self.N_SAMPLERS+n_sample, ...] = \
                self.sample_states_padded

        return samples
