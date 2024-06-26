import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
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

folders = glob('results/*')
arr = [float(string.split('\\')[1]) for string in folders]
n75 = np.argwhere(np.array(arr) == 0.75)[0][0]
"""
Distance Plot
"""
# for n,folderX in enumerate(folders):
#     subfolders = glob(folderX + '/*')
#     data_NonTest = np.transpose(np.loadtxt(subfolders[0], delimiter=',', skiprows=1))
#     data_Test =  np.transpose(np.loadtxt(subfolders[1], delimiter=',', skiprows=1))

#     plt.semilogy(arr[n], np.sum(data_NonTest[5]), 'x')
#     plt.semilogy(arr[n], np.sum(data_Test[5]),'o')

# plt.grid()
# plt.show()

"""
For D = 0.75
"""
subfolders = glob(folders[n75] + '/*')
data_NonTest = np.transpose(np.loadtxt(subfolders[0], delimiter=',', skiprows=1))
data_Test = np.transpose(np.loadtxt(subfolders[1], delimiter=',', skiprows=1))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
print(data_NonTest[:,0])
ax1.plot(data_NonTest[:,0], data_NonTest[:,1], '.--', label='Adaptive Epsilon')
ax1.plot(data_Test[:,0], data_Test[:,1], '.--', label='Constant Epsilon')
ax2.plot(data_NonTest[:,0],data_NonTest[:,5], 'x-', label=f'Adaptive Epsilon | Total Steps: {np.sum(data_NonTest[:,5])}')
ax2.plot(data_Test[:,0], data_Test[:,5], 'o-', label=f'Constant Epsilon | Total Steps: {np.sum(data_Test[:,5])}')
ax1.set_xlabel(r'$\theta$')
ax1.set_ylabel(r'Energy (Hartree)')
ax2.set_xlabel(r'$\theta$')
ax2.set_ylabel(r'Samples')

ground_energy = np.linalg.eigvalsh(h2_op([0.2252, 0.3435, -0.4347, 0.5716, 0.0910, 0.0910]))[0]
ax1.axhline(y=ground_energy, color='r', linestyle='--',label = r'Ground Energy for $d = 0.75 \AA$')
ax1.fill_between(data_Test[:,0], ground_energy-eps_bern, ground_energy+eps_bern,facecolor = 'red',alpha = 0.4,label = r'$\pm \epsilon$')
ax1.grid()
ax2.grid()
ax1.legend()
ax2.legend()
plt.show()

