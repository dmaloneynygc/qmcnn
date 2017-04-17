"""Optimizer."""
import numpy as np


class Optimizer:
    """Ground state optimizer."""

    def __init__(self, model, sampler, system):
        """Initialise."""
        self.model = model
        self.sampler = sampler
        self.system = system

    def optimize(self, iterations, num_samples):
        """Perform optimization."""
        for it in range(iterations):
            samples = self.sampler.sample(num_samples)
            E = self.system.local_energy(samples)
            O = self.model.backward(samples)
            O_conj = np.conj(O)

            S = np.mean(O_conj[:, :, np.newaxis] * O[:, np.newaxis, :],
                        axis=0)
            S -= np.mean(O_conj[:, :, np.newaxis], axis=0) * \
                np.mean(O[:, :, np.newaxis], axis=0)

            F = np.mean(E[:, np.newaxis] * O_conj, axis=0)
            F -= np.mean(E[:, np.newaxis], axis=0)*np.mean(O_conj, axis=0)

            delta = -1E-3 * np.dot(np.linalg.pinv(S), F)

            self.model.params += delta

            print("Iteration %d/%d, E=%f", it, iterations, np.mean(E))
