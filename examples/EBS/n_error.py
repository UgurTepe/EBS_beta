import numpy as np
import matplotlib.pyplot as plt


sample = lambda n: np.mean(np.random.beta(a, b,n))
mean = lambda a,b: a/(a+b)

a = 5   
b = 2
narr = np.logspace(1,3,100,dtype=int)

arr = []

for n in narr:
    summ = 0
    for i in range(100):
        if np.abs(mean(a,b)-sample(n))<= 0.1: summ += 1
    arr.append(summ/100)

#plt.loglog(narr,arr,label="Error")
plt.plot(narr,np.ones_like(arr)-arr,label="Error")
plt.grid()
plt.legend()
plt.show()
