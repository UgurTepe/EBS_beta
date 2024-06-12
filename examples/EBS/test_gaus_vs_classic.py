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

epsilon = 0.1
eba_classic = eba(delta=0.1, epsilon=epsilon, range_of_rndvar=1, beta=1.1)
eba_new = eba_geo_marg(delta=0.1, epsilon=epsilon, range_of_rndvar=1, beta=1.1)

while eba_classic.cond_check():
    sample = np.random.beta(5, 2)
    eba_classic.add_sample(sample)

while eba_new.cond_check():
    sample = np.random.beta(5, 2)

    eba_new.add_sample(sample)
    print(eba_new.get_ct())
    print(eba_new.x)
    print(eba_new.alpha)

print(eba_classic.get_step(), eba_new.get_step())
plt.plot(np.arange(eba_classic.current_k),eba_classic.ct,label="Classic")
plt.plot(np.arange(eba_new.current_k),eba_new.ct,label="Gauss")
plt.grid()
plt.legend()
plt.show()
