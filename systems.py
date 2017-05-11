"""Physical systems."""
import numpy as np
from utils.utils import flip_wrapped


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
        spin_padding = (np.array([self.model.r]*self.n_dims)-1)//2

        spinflips = np.stack(
            np.meshgrid(*[np.arange(x) for x in self.n_spins]), 0
            ).reshape(self.n_dims, -1).T

        spinflips = np.tile(spinflips, [batch_size, 1])

        states_flipped = np.repeat(state, self.total_spins, 0)
        flip_wrapped(states_flipped, spin_padding, spinflips)

        # print(state[0])
        # print(states_flipped.reshape([batch_size, self.total_spins, -1])[0])

        forwarded = self.model.forward(states_flipped)
        # log_psi_{n,f}
        forwarded = forwarded.reshape((batch_size, self.total_spins))
        log_pop = forwarded - self.model.forward(state)[:, np.newaxis]

        unpadded = self.model.unpad(state)
        interaction = np.zeros(batch_size)
        for dim, size in enumerate(self.n_spins):
            interaction += np.sum(unpadded * np.roll(unpadded, 1, axis=dim+1),
                                  axis=tuple(range(1, self.n_dims+1)))
        energy = -interaction - self.h * np.sum(np.exp(log_pop), axis=1)
        return energy / self.total_spins
