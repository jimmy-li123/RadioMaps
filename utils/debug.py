import numpy as np
import scipy.io as sio

test = sio.loadmat('data/raw/delay_t001_tx000_r000.mat')['delay']


print(test)