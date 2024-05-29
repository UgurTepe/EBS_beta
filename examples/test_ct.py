import numpy as np  
import matplotlib.pyplot as plt
import scipy.integrate as integrate



def hoeffding_bound(delta, epsilon, rng):
    return 0.5*np.log(2/delta)*rng**2/epsilon**2

delta = 0.1
epsilon = 0.1

t_arr = np.arange(1,hoeffding_bound(delta, epsilon, 1))
ac = delta/np.sqrt(2*np.pi)
var = 100
mu = hoeffding_bound(delta, epsilon, 1)/16
a = ac/np.sqrt(var)

gauss = lambda x: 0.5*a*np.exp(-0.5*(x-mu)**2/(2*var))
c = delta*(1.1-1)/1.1
expo = lambda x: np.exp(-x/c)
def func(x):
    return np.insert(np.exp(-x[1:]/c),0,delta)
d_t = lambda x: c/(x**1.1)
# result = np.sum(gauss(t_arr))
# print(result)

plt.plot(t_arr,-np.log(gauss(t_arr))/t_arr,label="Gaussian")
#plt.plot(t_arr,func(t_arr),label="Exponential")
plt.plot(t_arr,-np.log(d_t(t_arr))/t_arr,label="Classic")
plt.plot(t_arr,-np.log(0.1/np.log(hoeffding_bound(0.1,0.1,1)*np.log(1.1)))/t_arr,label="gleich")
plt.xlabel("Number of samples")
plt.ylabel(r"$d_t$")
plt.axvline(x=mu, color='r', linestyle='--',label=r"$t_{min}/4$")
plt.grid()
plt.legend()
plt.show()
