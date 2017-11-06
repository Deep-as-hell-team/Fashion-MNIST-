import numpy as np
import math

np.random.seed(1337)  # for reproducibility
np.set_printoptions(precision=2)

def step_decay(epoch):
    '''Learning rate step decay following the original paper.'''
    initial_lrate = 0.001
    drop = 0.99
    epochs_drop = 1
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate
