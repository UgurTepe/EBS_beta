import numpy as np


def hoeffding_bound(delta, epsilon, rng):
    '''
    Hoeffding bound solved for t_min. Calculates the minimum number of samples needed to achieve a given accuracy and confidence.
    '''
    return 0.5*np.log(2/delta)*rng**2/epsilon**2

class Welford():

    """
    Class for calculating the mean and standard deviation using the Welford's method.

    Attributes:
        n (int): The number of data points.
        M (float): The current mean.
        S (float): The current sum of squared differences from the mean.

    Methods:
        update(x): Updates the mean and sum of squared differences with a new data point.
        mean: Returns the current mean.
        std: Returns the current standard deviation.
    """

    def __init__(self, a_list=None):
        self.n = 0
        self.M = 0
        self.S = 0

    def update(self, x):
        """
        Updates the mean and sum of squared differences with a new data point.

        Args:
            x (float): The new data point.

        Returns:
            None
        """
        self.n += 1
        newM = self.M + (x - self.M) / self.n
        newS = self.S + (x - self.M) * (x - newM)
        self.M = newM
        self.S = newS

    @property
    def mean(self):
        """
        Returns the current mean.

        Returns:
            float: The current mean.
        """
        return self.M

    @property
    def std(self):
        """
        Returns the current standard deviation.

        Returns:
            float: The current standard deviation.
        """
        if self.n == 1:
            return 0
        return np.sqrt(self.S / (self.n - 1))

class dt_classic():
    def __init__(self,delta = 0.1, p = 1.1):
        self.c = delta*(p-1)/p
        self.p = p
    def inner_func(self,x):
        temp_val = self.c/(x**self.p)
        return temp_val
    
class dt_new():
    def __init__(self,delta = 0.1,rng = 1,epsilon = 0.1):   
        self.max_sample = hoeffding_bound(delta,epsilon,rng)
        self.delta = delta
        self.val = self.delta/np.log(self.max_sample*np.log(1.1))
    def inner_func(self):
        return self.val
    
class dt_new():
    def __init__(self,delta = 0.1,rng = 1,epsilon = 0.1):   
        self.max_sample = hoeffding_bound(delta,epsilon,rng)
        self.delta = delta
        self.val = self.delta/np.log(self.max_sample*np.log(1.1))
    def inner_func(self):
        return self.val       
class dt_gaus():
    def __init__(self,delta = 0.1,epsilon=0.1,rng=1):
        self.max_sample = hoeffding_bound(delta,epsilon,rng)/4
        self.var = 100
        self.const = delta/np.sqrt(2*np.pi*self.var)
  
    def inner_func(self,x):
        return 0.5*self.const*np.exp(-0.5*(x-self.max_sample)**2/(2*self.var))

class eba_geo_marg():
    def __init__(self, delta=0.1, epsilon=0.1, range_of_rndvar=1, beta=1.1,p=1.1):
        self.delta = delta
        self.epsilon = epsilon
        self.range_of_rndvar = range_of_rndvar
        self.samples = []
        self.p = p
        self.running_mean = [0]
        self.sample_sum = 0
        self.running_variance = [0]
        self.ct = []
        #self.dt_class = dt_class(delta = self.delta,epsilon = self.epsilon,rng = self.range_of_rndvar)
        self.dt_class = dt_new(delta = self.delta,epsilon = self.epsilon,rng = self.range_of_rndvar)
        self.dt = self.dt_class.inner_func
        self.dt_arr = []
        self.beta = beta
        self.x = 0
        self.alpha = 0
        self.current_k = 0
        self.current_t = 1
        self.welf = Welford()

    def add_sample(self, sample):
        """
        Adds a sample to the list of samples and updates the parameters.

        Parameters:
        - sample (float): The sample value.
        """
        self.samples.append(sample)
        self.sample_sum += sample
        cur_mean = np.divide(self.sample_sum, self.current_t)
        self.running_mean.append(cur_mean)
        self.welf.update(sample)
        self.running_variance.append(np.square(self.welf.std))
        self.current_t = self.current_t + 1

        # Inner loop condition check
        self.inner_cond_check()

    def cond_check(self):
        """
        Checks if the EBA should stop or continue.

        Returns:
        - bool: True if EBA should continue, False if EBA should stop.
        """
        if self.current_k == 0:
            return True
        if self.ct[-1] > self.epsilon:
            return True
        else:
            return False

    def inner_cond_check(self):
        """
        Check if the inner loop condition is satisfied.

        Returns:
        - none
        updates ct if the condition is satisfied
        """
        if self.current_t > np.floor(self.beta**self.current_k):
            self.update_ct()

    def calc_ct(self):
        """
        Calculates the c_t value for a given time t.

        Returns:
        - float: The c_t value.
        """
        if self.current_t <= 1:
            return 99
        else:
            return np.sqrt(2*self.running_variance[-1]*self.x/self.current_t)+3*self.range_of_rndvar*self.x/self.current_t

    def update_ct(self):
        """
        Updates the c_t value.
        """
        self.current_k += 1
        self.alpha = np.floor(self.beta**self.current_k) / \
            np.floor(self.beta**(self.current_k-1))
        self.dt_arr.append(self.dt())
        self.x = -self.alpha*np.log(self.dt()/3)
        self.ct.append(self.calc_ct())

    def get_ct(self):
        """
        Returns the array of c_t values.

        Returns:
        - numpy.ndarray: The array of c_t values.
        """
        return np.asarray(self.ct)

    def get_estimate(self):
        """
        Returns the latest estimated mean.

        Returns:
        - float: The latest estimated mean.
        """
        return self.running_mean[-1]

    def get_mean(self):
        """
        Returns the array of estimated means.

        Returns:
        - numpy.ndarray: The array of estimated means.
        """
        return np.asarray(self.running_mean)

    def get_var(self):
        """
        Returns the array of variances.

        Returns:
        - numpy.ndarray: The array of variances.
        """
        return np.asarray(self.running_variance)

    def get_step(self):
        """
        Returns the current iteration/step.

        Returns:
        - int: The current iteration/step.
        """
        return self.current_t
    def get_dt(self):
        return np.asarray(self.dt_arr)