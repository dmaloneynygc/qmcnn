"""Experiments module."""
from __future__ import division
from models import ConvCRBM
from systems import Ising
from sampler import Sampler
from optimizer import TFOptimizer


# import numpy as np

# path = '../Nqs/Ground/Ising1d_40_1_1.wf'
# model = CRBM.from_carleo_params(path)
# system = Ising1D(1., model.n_spins, model)
# sampler = Sampler(model.n_spins, model)
# print(np.mean(system.local_energy(sampler.sample(10000))))
# print(np.mean(system.local_energy(sampler.sample(10000))))

# n_spins = 40
# path = '../Nqs/Ground/Ising1d_40_1_1.wf'
# model = CRBM.from_carleo_params(path)
# model = CRBM(n_spins, 40)
# system = Ising1D(1., n_spins, model)
# sampler = Sampler(n_spins, model)
# optimizer = Optimizer(model, sampler, system)
# optimizer.optimize(100, 10000)

n_spins = [10, 10]
model = ConvCRBM(n_spins, 4, 5)
system = Ising(1., n_spins, model)
sampler = Sampler(n_spins, model)
# import numpy as np
# s = np.random.randint(2, size=[1000, 40+25-1], dtype=np.int8)*2-1
# print(model.forward_factors(s).mean())
# from time import time
# s = np.random.randint(2, size=[10, 20+2*(15-1), 20+2*(15-1)], dtype=np.int8)*2-1
# start = time()
# print(system.local_energy(s).mean())
# print(time()-start)
optimizer = TFOptimizer(model, sampler, system)
optimizer.optimize(1000, 5000)

# n_spins = [6]
# model = ConvCRBM(n_spins, 4, 3)
# system = Ising(1., n_spins, model)
# sampler = Sampler(n_spins, model)
# optimizer = TFOptimizer(model, sampler, system)
# optimizer.optimize(2, 2)
