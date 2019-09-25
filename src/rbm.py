import numpy as np
from math import exp

# RBM = Restricted Boltzman Machine 

def sigma(x):
  return 1 / (1 + exp(-x))
         
def probability_neurons(bj, Wj, ns):
  return sigma(
    bj + (np.array(Wj) @ np.array(ns))
  )