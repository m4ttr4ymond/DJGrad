import numpy as np


npzfile = np.load('gradients.npz', allow_pickle=True)['grads']
print(npzfile[0])