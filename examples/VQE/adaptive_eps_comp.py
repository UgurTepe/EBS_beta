import sys
import numpy as np
import matplotlib.pyplot as plt


'''
Configuring matplotlib
'''
plt.rcParams["figure.autolayout"] = True
plt.rcParams['mathtext.default'] = 'regular'
plt.rcParams.update({'font.size': 15})

# Importing own functions
sys.path.append("./")
from dep.qm_vqe import *
#eps_bern = 1.6e-3
eps_bern = 0.1

g = [0.2252, 0.3435, -0.4347, 0.5716, 0.0910, 0.0910] # d = 0.75 Å
arr_par, arr_energy,arr_var, arr_est_energy, arr_est_var, arr_steps, arr_höf,arr_max_flag,arr_grad1,arr_momentum,arr_epsilon = vqe_adam_h2_test(eps_bern=eps_bern,delta=0.1,hamiltonian_coeff=g)
arr_par2, arr_energy2,arr_var2, arr_est_energy2, arr_est_var2, arr_steps2, arr_höf2,arr_max_flag2,arr_grad2,arr_momentum2,arr_epsilon2 = vqe_adam_h2(eps_bern=eps_bern,delta=0.1,hamiltonian_coeff=g)

header_file = 'Parameter,Energy,Variance,EstimatedEnergy,EstimatedVariance,SamplesEBS,SamplesHoeffding,ConvergenceBool,Gradient,Momentum,Epsilon'

np.savetxt(f'data.txt', [arr_par, arr_energy,arr_var, arr_est_energy, arr_est_var, arr_steps, arr_höf,arr_max_flag,arr_grad1,arr_momentum,arr_epsilon], delimiter=',',header=header_file)
np.savetxt(f'data2.txt', [arr_par2, arr_energy2,arr_var2, arr_est_energy2, arr_est_var2, arr_steps2, arr_höf2,arr_max_flag2,arr_grad2,arr_momentum2,arr_epsilon2], delimiter=',',header=header_file)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

ax1.plot(arr_par, arr_est_energy, '.--', label='Adaptive Epsilon')
#ax1.plot(arr_par2, arr_est_energy2, '.--', label='Constant Epsilon')
ax2.plot(arr_par, arr_steps, 'x-', label=f'Adaptive Epsilon | Total Steps: {np.sum(arr_steps)}')
#ax2.plot(arr_par2, arr_steps2, 'x-', label=f'Constant Epsilon | Total Steps: {np.sum(arr_steps2)}')
ax1.set_xlabel(r'$\theta$')
ax1.set_ylabel(r'Energy (Hartree)')
ax2.set_xlabel(r'$\theta$')
ax2.set_ylabel(r'Samples')

ground_energy = np.linalg.eigvalsh(h2_op(g))[0]
ax1.axhline(y=ground_energy, color='r', linestyle='--',label = r'Ground Energy for $d = 0.75 \AA$')
ax1.fill_between(arr_par, ground_energy-eps_bern, ground_energy+eps_bern,facecolor = 'red',alpha = 0.4,label = r'$\pm \epsilon$')
ax1.grid()
ax2.grid()
ax1.legend()
ax2.legend()
plt.show()

