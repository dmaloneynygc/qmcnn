"""Physical systems."""
from __future__ import absolute_import
import numpy as np
from utils import all_windows


class Ising:
    """Transverse field Ising model."""

    def __init__(self, h, n_spins, model):
        """Initialise."""
        self.h = h
        self.n_spins = n_spins  # List of spins in dimensions
        self.total_spins = np.prod(n_spins)
        self.n_dims = len(n_spins)
        self.model = model

    def local_energy(self, state):
        """Compute energy per spin for batch of full-pad states."""
        batch_size = state.shape[0]
        half_window_size = np.array([self.model.r]*self.n_dims)
        full_window_size = half_window_size * 2 - 1

        spin_windows = all_windows(state, full_window_size)
        factors = self.model.forward_factors(state)
        factor_windows = all_windows(factors, half_window_size).reshape(
            [batch_size, self.total_spins]+list(half_window_size))

        windows_flattened = spin_windows.reshape(
            [batch_size*self.total_spins]+list(full_window_size))
        template = np.ones(full_window_size)
        template[(self.model.r-1,)*self.n_dims] *= -1
        spins_flipped = windows_flattened * template[np.newaxis, ...]
        factors_flipped = self.model.forward_factors(spins_flipped).reshape(
            [batch_size, self.total_spins]+list(half_window_size))
        log_pop = np.sum(factors_flipped-factor_windows,
                         tuple(range(2, self.n_dims+2)))

        unpadded = self.model.unpad(state, 'full')
        interaction = np.zeros(batch_size)
        for dim, size in enumerate(self.n_spins):
            interaction += np.sum(unpadded * np.roll(unpadded, 1, axis=dim+1),
                                  axis=tuple(range(1, self.n_dims+1)))
        energy = -interaction - self.h * np.sum(np.exp(log_pop), axis=1)

        return energy / self.total_spins
