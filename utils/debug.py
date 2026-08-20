import numpy as np
import scipy.io as sio

# print .mat files
test = sio.loadmat('data/raw/delay_t001_tx000_r000.mat')['delay']

# print .npy files
# test = np.load('../data/processed/CSI.npy')

print(test.shape)
print(test)