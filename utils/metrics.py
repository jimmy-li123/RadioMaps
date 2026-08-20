"""
utils/metrics.py

Purpose:
Provides core mathematical and channel impairment utilities, beamforming vector power normalization, spectral efficiency calculation, and the PyTorch unsupervised loss function.

Steps for Implementation:
- Implement channel perturbation functions: add_error (GPS noise), normalise_loc, add_fading (Rayleigh scattering), and add_noise (AWGN).
- Implement normalise_V to enforce the unit-power constraint ||V||^2 = 1 on beamforming vectors.
- Implement cal_SE for per-user and per-subcarrier Shannon capacity rate calculation: log2(1 + SNR).
- Implement compute to separate spectral efficiency results into LoS vs NLoS user groups.
- Implement optimal_svd_beamforming via SVD on perfect CSI as the theoretical upper-bound benchmark.
- Implement the PyTorch custom unsupervised loss su_loss that minimizes negative mean spectral efficiency: -E[log2(1 + |HV|^2 / sigma^2)].
"""


import os
import sys
from pathlib import Path
import torch
import numpy as np
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config

# =====================================================================      
# Configuration & Communication Parameters                                   
# ===================================================================== 
c = config.C
f = config.FC
B = config.B
lamda = config.LAMBDA_VAL
d = config.ANTENNA_SPACING


# =====================================================================      
# Data Preprocessing Functions                                               
# =====================================================================  
# Filters out blocked or unreachable points (should already be filtered)
def eliminate_block(UEloc, CSI):
    """                                                                      
    Eliminates blocked users (where CSI power is zero or negligible).        
                                                                                
    input:  UEloc (N, 2), CSI (N, Nc, Nr, Nt)                                
    output: UEloc (N_valid, 2), CSI (N_valid, Nc, Nr, Nt)                    
    """    
    power = np.abs(np.sum(CSI.reshape(CSI.shape[0], -1), axis=1))
    mask = power > 1e-15
    return UEloc[mask], CSI[mask]

def add_error(loc, std):
    """                                                                      
    Adds Gaussian GPS positioning error to user coordinates.                 
                                                                                
    input:  loc (N, 2), std: GPS standard deviation                          
    output: loc_error (N, 2)                                                 
    """
    np.random.seed(1) # ? Make variable global
    error = np.random.normal(0, std/np.sqrt(2), loc.shape)
    loc_error = loc + error
    return loc_error

def normalise_loc(loc):
    """                                                                      
    Normalises user coordinates to [0, 1] range.                             
                                                                                
    input:  loc (N, 2)                                                       
    output: loc_norm (N, 2)                                                  
    """       
    loc_norm = np.empty(loc.shape)
    x_max = np.max(loc[:,0])
    x_min = np.min(loc[:,0])
    x_len = x_max - x_min
    loc_norm[:,0] = (loc[:,0] - x_min) / x_len
    y_max = np.max(loc[:,1])
    y_min = np.min(loc[:,1])
    y_len = y_max - y_min
    loc_norm[:,1] = (loc[:,1] - y_min) / y_len
    return loc_norm

def add_fading(CSI, K):
    """                                                                      
    Adds Rayleigh fading to CSI based on scattering power ratio K.           
                                                                                
    input:  CSI (N, Nc, Nr, Nt), K: random scattering power ratio            
    output: CSI_fading (N, Nc, Nr, Nt)                                       
    """    
    fading = np.empty(CSI.shape, dtype=np.complex64)        
                     
    for i in range(CSI.shape[0]):                                            
        CSI_power = (np.linalg.norm(CSI[i, :]))**2 / np.prod(CSI[i, :].shape)
        fading_std = np.sqrt(K * CSI_power / 2)                              
        np.random.seed(1)                                                    
        fading_real = np.random.normal(0, fading_std, CSI[i, :].shape)       
        np.random.seed(2)                                                    
        fading_imag = np.random.normal(0, fading_std, CSI[i, :].shape)       
        fading[i, :] = fading_real + 1j * fading_imag                        
                                                                                
    CSI_fading = CSI + fading                                                
    return CSI_fading     

def add_noise(CSI, SNR):                                                     
    """                                                                      
    Adds AWGN noise to CSI given SNR in dB.                                  
                                                                                
    input:  CSI (N, Nc, Nr, Nt), SNR (dB)                                    
    output: CSI_noise (N, Nc, Nr, Nt)                                        
    """                                                                      
    power = (np.linalg.norm(CSI))**2 / np.prod(CSI.shape)                    
    noise_std = np.sqrt(power / (10 ** (SNR / 10)) / 2)                      
    np.random.seed(1)                                                        
    noise_real = np.random.normal(0, noise_std, CSI.shape)                   
    np.random.seed(2)                                                        
    noise_imag = np.random.normal(0, noise_std, CSI.shape)                   
    noise_comp = noise_real + 1j * noise_imag                                
    CSI_noise = CSI + noise_comp                                             
    return CSI_noise                                                         
                                                                                
                                                                                
def normalise_V(V):                                                          
    """                                                                      
    Normalises beamforming vector to meet unit power constraint: V / sqrt(V^H * V).                                                                          
                                                                                
    input:  V (N, Nc, Nt, 1) or (N, Nc, Nt)                                  
    output: V_norm (same shape)                                              
    """         
    # ? Fix                                                                                                                            
    if V.ndim == 4 and V.shape[-1] == 1:                                 
        V_conj_tran = np.transpose(V, (0, 1, 3, 2)).conjugate()          
        V_power = np.matmul(V_conj_tran, V)                              
        V_norm = V / np.sqrt(np.real(V_power) + 1e-12)                   
    else:                                                                
        power = np.sum(np.abs(V) ** 2, axis=-1, keepdims=True)           
        V_norm = V / np.sqrt(power + 1e-12)                              
    return V_norm  


# =====================================================================      
# Spectral Efficiency Calculation & Evaluation                               
# =====================================================================      
def cal_SE(H, V, noise):    
    """                                                                      
    Calculates spectral efficiency for all users across subcarriers.         
                                                                                
    input:  H (N, Nc, Nr, Nt), V (N, Nc, Nt, 1), noise (float)               
    output: rate_Nc (N, )                                                    
    """                                                                   
    HV = np.matmul(H, V)                                                 
    HV_gain = np.matmul(np.transpose(np.conj(HV), (0, 1, 3, 2)), HV)     
    HV_gain = np.squeeze(np.abs(HV_gain))               # (N, Nc)        
    SNR = HV_gain / noise                                                
    rate = np.log2(1.0 + SNR)                                            
    rate_Nc = np.mean(rate, axis=1)                     # Average over subcarriers                                                                    
    return rate_Nc       



def compute(rate_Nc, LoS_test=None):
    """                                                                      
    Computes separate Spectral Efficiency for LoS and NLoS conditions.       

    input:  rate_Nc (N_test, ), LoS_test (optional (N_test, ))               
    output: LoS_SE, NLoS_SE                                                  
    """   
    if LoS_test is None:                                                     
        LoS = np.load(config.PROCESSED_DATA_DIR / 'LoS.npy')                                
        LoS = LoS[LoS != -1]                                                 
        _, LoS_test = train_test_split(LoS, test_size=0.2, random_state=1)   
                                                                                
    if isinstance(rate_Nc, torch.Tensor):                                    
        rate_Nc = rate_Nc.detach().cpu().numpy()                             
    if isinstance(LoS_test, torch.Tensor):                                   
        LoS_test = LoS_test.detach().cpu().numpy()                           
                                                                                
    los_mask = (LoS_test == 1)                                               
    nlos_mask = (LoS_test == 0)                                              
                                                                                
    LoS_SE = float(rate_Nc[los_mask].mean()) if np.any(los_mask) else 0.0    
    NLoS_SE = float(rate_Nc[nlos_mask].mean()) if np.any(nlos_mask) else 0.0 
                                                                                
    return LoS_SE, NLoS_SE      


# =====================================================================      
# Custom Loss Function & PyTorch Training Loop                               
# =====================================================================      
#! TODO