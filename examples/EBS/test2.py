import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import betabinom

'''
Configuring matplotlib
'''
plt.rcParams["figure.autolayout"] = True
plt.rcParams['mathtext.default'] = 'regular'
plt.rcParams.update({'font.size': 15})

# Importing own functions
sys.path.append("./")
from src.algorithms import eba_geo_marg as eba
from src.test_alg import *

epsilon = 0.001
eba_classic = eba(delta=0.1, epsilon=epsilon, range_of_rndvar=1, beta=1.1)
eba_new = eba_geo_marg(delta=0.1, epsilon=epsilon, range_of_rndvar=1, beta=1.05)

while eba_classic.cond_check():
    sample = np.random.beta(5, 2)
    eba_classic.add_sample(sample)

while eba_new.cond_check():
    sample = np.random.beta(5, 2)
    eba_new.add_sample(sample)

print(eba_classic.get_step(), eba_new.get_step())