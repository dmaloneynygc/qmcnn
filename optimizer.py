"""Optimizer."""
import numpy as np
from time import time


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
            start = time()
            samples = self.sampler.sample(num_samples)
            time_sample, start = time()-start, time()
            E = self.system.local_energy(samples)
            O = self.model.backward(samples)
            O_conj = np.conj(O)

            F = np.dot(O_conj.T, E) / num_samples
            F -= np.mean(E, axis=0)*np.mean(O_conj, axis=0)
            time_model, start = time()-start, time()

            # S = np.dot(O_conj.T, O)/num_samples - \
            #     np.outer(O_conj.mean(axis=0), O.mean(axis=0))
            # delta = -1E-3 * np.dot(np.linalg.pinv(S), F)
            delta = -1 * F

            self.model.set_params(self.model.params + delta)
            time_solve, start = time()-start, time()

            print(("Iteration %d/%d, E=%f (%.2f), w=%.2E delta=%.2E "
                   "(%.2fs, %.2fs, %.2fs)") %
                  (it, iterations, np.real(np.mean(E)),
                   np.std(E)/np.sqrt(num_samples),
                   np.mean(np.absolute(self.model.params)),
                   np.mean(np.absolute(delta)),
                   time_sample, time_model, time_solve))


class TFOptimizer:
    """Ground state optimizer."""

    def __init__(self, model, sampler, system):
        """Initialise."""
        self.model = model
        self.sampler = sampler
        self.system = system

    def optimize(self, iterations, num_samples):
        """Perform optimization."""
        for it in range(iterations):
            start = time()
            samples = self.sampler.sample(num_samples)
            time_sample, start = time()-start, time()
            E = self.system.local_energy(samples)
            time_energy, start = time()-start, time()
            self.model.optimize(samples, E)
            time_optimize, start = time()-start, time()

            print(("Iteration %d/%d, E=%f (%.2E), "
                   "(%.2fs, %.2fs, %.2fs)") %
                  (it, iterations, np.real(np.mean(E)),
                   np.std(E)/np.sqrt(num_samples),
                   time_sample, time_energy, time_optimize))
