# Geometry based method
# Estimates the beamforming vector by trig

import sys
from pathlib import Path
import numpy as np
import math

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from utils.metrics import *

# parameters
def main():
    args = parse_args(description="Evaluate Geometry-Based Method (GBM) baseline")

    loc_std      = args.loc_std
    fading_ratio = args.fading_ratio
    SNR          = args.snr
    Nc           = config.NC
    Nt           = config.NT

    # Load data arrays
    UEloc, CSI, LoS, BSloc = load_dataset(args.dataset, return_bsloc=True)

    # Process data
    UEloc, CSI = eliminate_block(UEloc, CSI)
    CSI = CSI / np.max(np.abs(CSI))
    CSI_fading = add_fading(CSI, fading_ratio)

    # Divide train and test set
    x_train, x_test, y_train, y_test, LoS_train, LoS_test = train_test_split(
        UEloc, CSI_fading, LoS, 
        test_size=config.TEST_SIZE, 
        random_state=config.RANDOM_SEED
    ) # Set rng seed as 1
    N_test = x_test.shape[0]

    # Add position error to test to make it more realistic
    x_test = add_error(x_test, loc_std)

    # Calculate AoD based on locations
    # Doesn't pull from precalculated values
    AoD = np.arctan2(x_test[:, 1] - BSloc[0, 1], x_test[:, 0] - BSloc[0, 0]) # returns in radians

    # Calculate transmit precoding vector - vectorised (N_test, Nc, Nt, 1)
    sin_AoD = np.sin(AoD)[:, None, None, None]
    j = np.arange(Nc)
    lamda = c / (f + j * B / Nc)[None, :, None, None]
    k = np.arange(Nt)[None, None, :, None]
    phase = -2 * np.pi * k * d * sin_AoD / lamda  # Transmit array steering phase

    V_GBM = np.exp(1j * phase).astype(np.complex64)
            
    # Normalise beamforming vector
    V_GBM = normalise_V(V_GBM)

    # Noise
    power = (np.linalg.norm(CSI_fading))**2/np.prod(CSI_fading.shape)
    noise = power / (10**(SNR/10))

    # Calculate SE and optimum of GBM
    GBM_SE = cal_SE(y_test, V_GBM, noise)
    opt_SE = cal_opt_SE(y_test, noise)

    # Calculate ratio
    GBM2opt = np.mean(GBM_SE) / np.mean(opt_SE) * 100                                           
                                                                                                            
    # Compute LoS and NLoS metrics                                                                       
    GBM_LoS_SE, GBM_NLoS_SE = compute(GBM_SE, LoS_test)                                                  
    opt_LoS_SE, opt_NLoS_SE = compute(opt_SE, LoS_test)                                                  
                                                                                                            
    GBM2opt_LoS = (GBM_LoS_SE / opt_LoS_SE) * 100 if opt_LoS_SE > 0 else 0.0                             
    GBM2opt_NLoS = (GBM_NLoS_SE / opt_NLoS_SE) * 100 if opt_NLoS_SE > 0 else 0.0  

    # Print result
    print(f"=== GBM Baseline ({args.dataset}) ===")
    print('GBM2opt: ', np.round(GBM2opt, 3), '%')
    print('GBM LoS: ', np.round(GBM2opt_LoS, 3), '%')
    print('GBM NLoS:', np.round(GBM2opt_NLoS, 3), '%')

if __name__ == '__main__':
    main()

# GBM2opt: 91.612 %
# GBM LoS: 95.472 %
# GBM NLoS: 31.746 %