"""Physical systems."""
import numpy as np


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
        """Compute energy per spin for batch of states."""
        batch_size = state.shape[0]
        flipper = (1-2*np.eye(self.total_spins))[np.newaxis, :, :]
        # S_{nfi}, n: number in batch, f: spin flipped, i: spin
        state_1d = state.reshape((batch_size, 1, self.total_spins))
        # S_{n*f,i_1,...}
        flattened = (flipper * state_1d) \
            .reshape([batch_size, self.total_spins]+self.n_spins) \
            .reshape([batch_size*self.total_spins]+self.n_spins)
        # log_psi_{n*f}
        forwarded = self.model.forward(flattened)
        # log_psi_{n,f}
        forwarded = forwarded.reshape((batch_size, self.total_spins))
        log_pop = forwarded - self.model.forward(state)[:, np.newaxis]

        interaction = np.zeros(batch_size)
        for dim, size in enumerate(self.n_spins):
            interaction += np.sum(state * np.roll(state, 1, axis=dim+1),
                                  axis=tuple(range(1, self.n_dims+1)))
        energy = -interaction - self.h * np.sum(np.exp(log_pop), axis=1)
        return energy / self.total_spins
