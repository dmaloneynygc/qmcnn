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

            S = np.dot(O_conj.T, O)/num_samples
            S -= np.mean(O_conj, axis=0)[:, np.newaxis] * \
                np.mean(O, axis=0)[np.newaxis, :]
            S_iv = np.linalg.pinv(S)

            F = np.mean(E[:, np.newaxis] * O_conj, axis=0)
            F -= np.mean(E[:, np.newaxis], axis=0)*np.mean(O_conj, axis=0)

            delta = -1E-5 * np.dot(S_iv, F)

            self.model.params += delta

            print("Iteration %d/%d, E=%f (%f)" %
                  (it, iterations, np.real(np.mean(E)), np.std(E)))
