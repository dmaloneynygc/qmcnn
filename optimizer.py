"""Optimizer."""
import numpy as np
from minresQLP import minresQLP
import theano
import theano.tensor as T
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
            # delta = -1E-3
            # delta = -1E2 * self._solve(O, O_conj, F)

            self.model.set_params(self.model.params + delta)
            time_solve, start = time()-start, time()

            print(("Iteration %d/%d, E=%f (%.2f), w=%.2E delta=%.2E "
                   "(%.2fs, %.2fs, %.2fs)") %
                  (it, iterations, np.real(np.mean(E)), np.std(E),
                   np.mean(np.absolute(self.model.params)),
                   np.mean(np.absolute(delta)),
                   time_sample, time_model, time_solve))

    def _solve(self, O_conj, O, f):
        """Solve (<O*O>-<O*><O>)x=f for x through minresQLP."""
        tf = theano.shared(f)
        tO = theano.shared(O)
        tO_conj = theano.shared(O_conj)
        tO_mean = theano.shared(np.mean(O, axis=0))
        tO_conj_mean = theano.shared(
            np.mean(O_conj, axis=0))
        num_samples = O.shape[0]
        num_params = f.shape[0]

        def compute_Ax(x):
            """Compute Ax."""
            Ax = T.dot(tO_conj.T, T.dot(tO, x))/num_samples - \
                tO_conj_mean * T.dot(tO_mean, x)
            return ([Ax], {})

        sol, flag, iters, relres, Anorm, Acond = minresQLP(
            compute_Ax, tf, param_shapes=(num_params,)
        )
        mqlp = theano.function([], [sol, flag, iters, relres, Anorm, Acond])
        sol, flag, iters, relres, Anorm, Acond = mqlp()
        sol = np.array(sol)

        sol = np.array(sol)
        # print('flag', flag)
        # print('iters', iters)
        # print('relres', relres)
        # print('Anorm', Anorm)
        # print('Acond', Acond)
        # print('Solution', sol)

        return sol
