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


epsilon = 0.01

paras = np.loadtxt(fname='dep/h2_config/h2_parameter.csv',dtype=None,delimiter=',',skiprows=1)
bond_length = paras[:, 0]

g = [0.2252, 0.3435, -0.4347, 0.5716, 0.0910, 0.0910]


arr_par, arr_energy,arr_var, arr_est_energy, arr_est_var, arr_steps, arr_höf,arr_max_flag = vqe_adam_h2(eps_bern=epsilon,delta=0.1,hamiltonian_coeff=g)
arr_par1, arr_energy1, arr_var1, arr_est_energy1, arr_est_var1, arr_steps1, arr_höf1, arr_max_flag1 = vqe_adam_h2_test(eps_bern=epsilon, delta=0.1, hamiltonian_coeff=g)
fig, (ax1, ax2) = plt.subplots(1, 2)

ax1.plot(arr_par, arr_est_energy, '.-', label='Energy Classic')
ax1.plot(arr_par, arr_est_var, '.-', label='Variance Classic')
ax1.plot(arr_par1, arr_est_energy1, '.-', label='Energy Mod')
ax1.plot(arr_par1, arr_est_var1, '.-', label='Variance Mod')

ax2.plot(arr_par, np.cumsum(arr_steps), '.-', label='Steps Classic')
ax2.plot(arr_par1, np.cumsum(arr_steps1), '.-', label='Steps Mod')

ax1.set_xlabel(r'$\theta$')
ax2.set_xlabel(r'$\theta$')
ax1.set_ylabel('Energy/Variance')
ax2.set_ylabel('Steps')
ax1.grid()
ax2.grid()
ax1.legend()
ax2.legend()

# plt.plot(arr_par, arr_est_var, '.-')
# plt.plot(arr_par, arr_max_flag, '.-', label='epsilon')
# plt.legend()
# plt.grid()

plt.show()
