import numpy as np
def hoeffding_bound(delta, epsilon, rng):
    return 0.5*np.log(2/delta)*rng**2/epsilon**2

x  = 0.1/np.log(hoeffding_bound(0.1,0.1,1)*np.log(1.1))
ar = np.arange(1,hoeffding_bound(0.1,0.1,1))      
print(np.sum(np.ones_like(ar)*x))