"""Experiments module."""
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

n_spins = [40]
model = ConvCRBM(n_spins, 4, 9)
system = Ising(1., n_spins, model)
sampler = Sampler(n_spins, model)
optimizer = TFOptimizer(model, sampler, system)
optimizer.optimize(100, 1000)

# n_spins = [6]
# model = ConvCRBM(n_spins, 4, 3)
# system = Ising(1., n_spins, model)
# sampler = Sampler(n_spins, model)
# optimizer = TFOptimizer(model, sampler, system)
# optimizer.optimize(8, 2)
