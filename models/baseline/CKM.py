# Channel Knowledge Map method
# Uses a database of location measurements

import sys
from pathlib import Path
import numpy as np
import math
from scipy.spatial.distance import cdist

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from utils.metrics import *

# Using default parameters
loc_std = config.DEFAULT_LOC_STD
fading_ratio = config.DEFAULT_FADING_RATIO
SNR = config.DEFAULT_SNR

# Load data arrays explicitly
UEloc = np.load(config.PROCESSED_DATA_DIR / 'UEloc.npy')
AoD   = np.load(config.PROCESSED_DATA_DIR / 'AoD.npy')
LoS   = np.load(config.PROCESSED_DATA_DIR / 'LoS.npy')
CSI   = np.load(config.PROCESSED_DATA_DIR / 'CSI.npy')    # Already normalised
Nc    = config.NC
Nt    = config.NT
d     = config.ANTENNA_SPACING

CSI_fading = add_fading(CSI, fading_ratio)

# Diving training and test set
x_train, x_test, y_train, y_test, AoD_train, AoD_test, LoS_train, LoS_test = train_test_split(
    UEloc, CSI_fading, AoD, LoS,
    test_size= config.TEST_SIZE, 
    random_state=config.RANDOM_SEED
)
N_train, N_test = AoD_train.shape[0], AoD_test.shape[0]

# Adding position error to test data
x_test = add_error(x_test, loc_std)

# Calculate distance by IDW (inverse distance weighting)
dist = cdist(x_test, x_train, metric='euclidean')   # (N, n_dist)

# k-Nearest Neighbours (k=3) + IDW
k = 3
k_idx = np.argpartition(dist, kth=k, axis=1)[:, :k] # Sorts rows in O(N) time by grabbing smallest k distances at once
k_dist = np.take_along_axis(dist, k_idx, axis=1)   # extracts elements from an array - k_dist[i, j] = arr[i, indices(i, j)]

weights = 1/k_dist
weights = weights / np.sum(weights)

# Circular angle interpolation - averages angles across nearest neighbours without boundary wrapping
k_AoD = AoD_train[k_idx]
sin_interp = np.sum(weights * np.sin(np.deg2rad(k_AoD)), axis=1)
cos_interp = np.sum(weights * np.cos(np.deg2rad(k_AoD)), axis=1)
AoD_interp = np.arctan2(sin_interp, cos_interp)

theta = AoD_interp[:, None, None, None]
lamda = (c / (f + np.arange(Nc) * B / Nc))[None, :, None, None]
antenna_index = np.arange(Nt)[None, None, :, None]
phase = (-2 * np.pi * antenna_index * d * np.sin(theta) / lamda)

V_CKM = normalise_V(np.exp(1j * phase).astype(np.complex64))

power = (np.linalg.norm(CSI_fading))**2 / np.prod(CSI_fading.shape)
noise = power / (10 ** (SNR / 10))

## Calculate spectral efficiency and optimum
CKM_SE = cal_SE(y_test, V_CKM, noise)
opt_SE = cal_opt_SE(y_test, noise)



# Calculate ratio
CKM2opt = np.mean(CKM_SE) / np.mean(opt_SE) * 100                                           
                                                                                                        
# Compute LoS and NLoS metrics                                                                       
CKM_NLoS, CKM_NLoS_SE = compute(CKM_SE, LoS_test)                                                  
opt_LoS_SE, opt_NLoS_SE = compute(opt_SE, LoS_test)                                                  
                                                                                                        
CKM2opt_LoS = (CKM_NLoS / opt_LoS_SE) * 100 if opt_LoS_SE > 0 else 0.0                             
CKM2opt_NLoS = (CKM_NLoS_SE / opt_NLoS_SE) * 100 if opt_NLoS_SE > 0 else 0.0  

# Print result
if __name__ == '__main__':
    print('CKM2opt:', np.round(CKM2opt, 3), '%')
    print('CKM LoS:', np.round(CKM2opt_LoS, 3), '%')
    print('CKM NLoS:', np.round(CKM2opt_NLoS, 3), '%')

# CKM2opt: 94.034 %
# CKM LoS: 94.514 %
# CKM NLoS: 86.585 %



