"""Physical systems."""
import numpy as np


class Ising1D:
    """1D Transverse field Ising model."""

    def __init__(self, h, n_spins, model):
        """Initialise."""
        self.h = h
        self.n_spins = n_spins
        self.model = model

    def local_energy(self, state):
        """Compute energy per spin for batch of states."""
        flipper = (1-2*np.eye(self.n_spins))[np.newaxis, :, :]
        # S_{nfi}, n: number in batch, f: spin flipped, i: spin in basis
        connected_states = state[:, np.newaxis, :] * flipper
        flattened = connected_states.reshape((-1, self.n_spins))
        forwarded = self.model.forward(flattened).reshape((-1, self.n_spins))
        log_pop = forwarded - self.model.forward(state)[:, np.newaxis]

        interaction = np.sum(state * np.roll(state, 1, axis=1), axis=1)
        energy = -interaction - self.h * np.sum(np.exp(log_pop), axis=1)
        return energy / self.n_spins
