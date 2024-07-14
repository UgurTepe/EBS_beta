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
#arr = [float(string.split('\\')[1]) for string in folders]
arr = [float(string.split('/')[1]) for string in folders]


"""
Distance Plot
"""
N_höf = 0
N_steps = 0
for n,folderX in enumerate(folders):
    #subfolders = glob(folderX + '/*')
    file_normal = folderX+'/data_adam_epsilon_{}_0.txt'.format(str(eps_bern))
    file_test = folderX+'/data_adam_test_epsilon_{}_0.txt'.format(str(eps_bern))
    arr_par, arr_energy,arr_var, arr_est_energy, arr_est_var, arr_steps, arr_höf,arr_max_flag,arr_grad,arr_momentum,arr_epsilon = np.loadtxt(file_normal, delimiter=',', skiprows=1)
    arr_par2, arr_energy2,arr_var2, arr_est_energy2, arr_est_var2, arr_steps2, arr_höf2,arr_max_flag2,arr_grad2,arr_momentum2,arr_epsilon2 =  np.loadtxt(file_test, delimiter=',', skiprows=1)
    N_höf += np.sum(arr_höf)
    N_steps += np.sum(arr_steps)
    plt.plot(arr[n], np.mean(arr_höf), 'gx',label = f'Höffding' if n == 0 else '')
    plt.plot(arr[n], np.mean(arr_steps2),'ro',label = f'EBS' if n == 0 else '')
plt.plot([], [], ' ', label='Total Samples: Höf={:.2e} / Bern = {:.2e}'.format(N_höf, N_steps))
print(N_höf, N_steps)
plt.xlabel(r'$d [\AA]$')
plt.ylabel(r'Average #Samples per Distance')
plt.legend()
plt.grid()
plt.show()