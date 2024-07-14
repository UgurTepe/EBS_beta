import sys
import numpy as np
import matplotlib.pyplot as plt
#from sklearn.preprocessing import normalize
from glob import glob
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
eps_bern = 0.01

folders = glob('results-6/*')
arr = [float(string.split('\\')[1]) for string in folders] # Windows
#arr = [float(string.split('/')[1]) for string in folders] # Mac 
n75 = np.argwhere(np.array(arr) == 0.75)[0][0]

"""
For D = 0.75
"""
subfolders = folders[n75]
print(subfolders)
file_normal = subfolders+'/data_adam_epsilon_{}_0.txt'.format(str(eps_bern))
file_test = subfolders+'/data_adam_test_epsilon_{}_0.txt'.format(str(eps_bern))
arr_par, arr_energy,arr_var, arr_est_energy, arr_est_var, arr_steps, arr_höf,arr_max_flag,arr_grad,arr_momentum,arr_epsilon = np.loadtxt(file_normal, delimiter=',', skiprows=1)
arr_par2, arr_energy2,arr_var2, arr_est_energy2, arr_est_var2, arr_steps2, arr_höf2,arr_max_flag2,arr_grad2,arr_momentum2,arr_epsilon2 = np.loadtxt(file_test, delimiter=',', skiprows=1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

ax1.plot(arr_par, arr_est_energy, '.-', label='Constant Epsilon')
ax1.plot(arr_par2, arr_est_energy2, '.-', label='Adaptive Epsilon')
# ax2.plot(arr_par,arr_epsilon, 'x-', label=f'Constant Epsilon | Total Steps: {np.sum(arr_steps)}')
# ax2.plot(arr_par2, arr_epsilon2  , 'o-', label=f'Adaptive Epsilon | Total Steps: {np.sum(arr_steps2)}')
ax2.plot(arr_par,arr_steps, 'x-', label=f'EBS Constant Epsilon {np.sum(arr_steps):.3e} Steps= {len(arr_steps)}')
ax2.plot(arr_par2, arr_steps2  , 'o-', label=f'EBS Adaptive Epsilon{np.sum(arr_steps2):.3e} Steps= {len(arr_steps2)}')
ax2.plot(arr_par, arr_höf, 'x-', label=f'Höffding  {np.sum(arr_höf):.3e} Steps= {len(arr_höf)}')
ax1.set_xlabel(r'$\theta$')
ax1.set_ylabel(r'Energy (Hartree)')
ax2.set_xlabel(r'$\theta$')
ax2.set_ylabel(r'Samples')

ground_energy = np.linalg.eigvalsh(h2_op([0.2252, 0.3435, -0.4347, 0.5716, 0.0910, 0.0910]))[0]
ax1.axhline(y=ground_energy, color='r', linestyle='--',label = r'Ground Energy for $d = 0.75 \AA$')
ax1.fill_between(arr_par, ground_energy-eps_bern, ground_energy+eps_bern,facecolor = 'red',alpha = 0.4,label = r'$\pm \epsilon$')
ax1.grid()
ax2.grid()
ax1.legend()
ax2.legend()
plt.show()

