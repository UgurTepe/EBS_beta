
import numpy as np
import sys
import matplotlib.pyplot as plt 
sys.path.append("./")
from src.algorithms import eba_geo_marg as eba
from src.test_alg import eba_geo_marg as eba_mod
from src.test_alg import *

def load_samples(filename):
    params = {}
    with open(filename, "r") as f:
        line = f.readline().strip()
        while line.find("#") >= 0:
            key, val = line.split("=")
            params[key[2:]] = float(val)
            line = f.readline().strip()
    samples = np.loadtxt(filename, skiprows=len(params.keys()))
    return samples, params

def main(molecule_name="H2",mapping_name="JW",epsilon=0.1,delta=0.1):
    samples, params = load_samples("dep/h2_config/EBS_samples/{}_molecule_{}_samples.txt".format(molecule_name, mapping_name))
    np.random.shuffle(samples)
    ebs = eba(epsilon=0.1, delta=0.1, range_of_rndvar=params["R"])

    ind = 0
    while ebs.cond_check():
        if ind == len(samples):
            np.random.shuffle(samples)
            ind = 0
        ebs.add_sample(samples[ind])
        ind += 1
    print(ebs.get_step()*params["Ngroups"],np.ceil(1.1**(ebs.current_k-1))*params["Ngroups"],ebs.current_k-1,params["Ngroups"])
    return ebs.get_step()*params["Ngroups"],ebs.get_estimate()

main(molecule_name="H2",mapping_name="JW",epsilon=0.1,delta=0.1)