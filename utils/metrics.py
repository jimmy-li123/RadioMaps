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
import time
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config

# =====================================================================      
# Configuration & Communication Parameters                                   
# ===================================================================== 
c     = config.C
f     = config.FC
B     = config.B
lamda = config.LAMBDA_VAL
d     = config.ANTENNA_SPACING

# ? Fix 
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

# ? Faster SVD - V_opt = H^H/abs(H)
def cal_opt_SE(H, noise):
    """
    Calculates optimal spectral efficiency given appropriate CSI
    
    input: H (N, Nc, Nr, Nt) , noise (float)
    output: rate_Nc (N, )
    """   
    H_H = np.transpose(H, (0, 1, 3, 2)).conjugate()
    H_norm = np.linalg.norm(H, axis = -1, keepdims = True) 
    V_opt = H_H / (H_norm + 1e-12)
    return cal_SE(H, V_opt, noise)


# Computes SE of Los and NLos conditions
# form a 1D array of SE of each user
def compute(rate_Nc, LoS_test=None):
    """                                                                      
    Computes separate Spectral Efficiency for LoS and NLoS conditions.       

    input:  rate_Nc (N_test, ), LoS_test (optional (N_test, ))               
    output: LoS_SE, NLoS_SE                                                  
    """   
    if LoS_test is None:                                                     
        LoS = np.load(config.PROCESSED_DATA_DIR / 'LoS.npy')                                
        LoS = LoS[LoS != -1]                                                 
        _, LoS_test = train_test_split(LoS, test_size=config.TEST_SIZE, random_state=config.RANDOM_SEED)   
                                                                        
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
# Apparently classes should be in CamelCase
class SELoss(nn.Module):
    """
    Unsupervised SE Loss
    Minimises negative Shannon rate
    """
    def __init__(self, noise):
        super().__init__()
        self.noise = noise
    
    # nn.module always calls foward when invoked with criterion
    def forward(self, H, V):
        # H: (N, Nc, 1, Nt)
        # V: (N, Nc, Nt, 1)

        # Normalising
        V_power = torch.sum(torch.abs(V)**2, dim=-2, keepdim=True)  # targets Nt
        V_norm = V / torch.sqrt(V_power + 1e-12)

        # Received gain: |H * V_norm|^2
        HV = torch.matmul(H, V_norm)    # (N, Nc, 1, 1)
        HV_gain = torch.abs(torch.squeeze(HV, dim=-1))**2   # (N, Nc, 1)
        
        # Shannon's capacity: log2(1 + SNR)
        SNR = HV_gain / self.noise
        rate = torch.log2(1.0 + SNR)
        return -torch.mean(rate)
        
# ! Overfitting test w 5 samples
def train_model(model, x_train, y_train, x_test, y_test, noise, 
          epochs=1000, batch_size=128, lr=1e-3, save_path=None, device=None):
    """
    Universal training loop for models
    """
    if device is None:
        device = config.DEVICE
    
    # Initialise
    model = model.to(device)                                            # Use GPU if available
    criterion = SELoss(noise).to(device)                               # Initialises criterion object
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)             # model.parameters holds weights and biases
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(             # Dynamically changes optim factor
        optimiser, mode='min', factor=0.1, patience=50, min_lr=1e-6
    )
    
    # Convert to PyTorch Tensors
    x_tr = torch.as_tensor(x_train, dtype=torch.float32)    # x is real valued
    y_tr = torch.as_tensor(y_train, dtype=torch.complex64)  # y is im
    x_vl = torch.as_tensor(x_test, dtype=torch.float32)
    y_vl = torch.as_tensor(y_test, dtype=torch.complex64)

    # Separate into batches
    train_dataset = TensorDataset(x_tr, y_tr)       # Separates input into paired samples
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)   # Split into shuffled mini batches

    best_val_loss = float('inf')    # Initial loss
    history = {'train_loss': [], 'val_loss': []}
    
    print(f"Starting training on {device} ({epochs} epochs, batch_size={batch_size})...")

    t0_total = time.perf_counter()
    t0_interval = time.perf_counter()

    try:
        for epoch in range(1, epochs + 1):  # [1, epoch]
            model.train()                   # Set to training mode, zeros some activations to prevent overfitting
            total_train_loss = 0.0
            
            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)
                optimiser.zero_grad()           # Reset gradients from parameters
                
                v_pred = model(bx)              # Forward pass
                loss = criterion(by, v_pred)    # Compute loss
                loss.backward()                 # Populate .grad buffers with fresh gradients
                optimiser.step()                # Updates weights using new gradients

                total_train_loss += loss.item() * bx.size(0)    # Accumulates losses of each batch
            
            train_loss = total_train_loss / len(train_dataset)
            
            # Validation
            model.eval()
            with torch.no_grad():
                v_val = model(x_vl.to(device))
                val_loss = criterion(y_vl.to(device), v_val).item()
                
            scheduler.step(val_loss)
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if save_path:
                    torch.save(model.state_dict(), save_path)
            
            if epoch == 1 or epoch % 50 == 0 or epoch == epochs:
                current_lr = optimiser.param_groups[0]['lr']
                elapsed_50 = time.perf_counter() - t0_interval
                print(f"\rEpoch [{epoch:4d}/{epochs:4d}] - Train SE: {-train_loss:.3f} bps/Hz | Val SE: {-val_loss:.3f} bps/Hz | LR: {current_lr:.1e} | Time: {elapsed_50:.2f}s")
                t0_interval = time.perf_counter()
            else:
                print(f"\rTraining Epoch: {epoch:4d}/{epochs:4d}...", end="", flush=True)
        print()
        total_time = time.perf_counter() - t0_total
        print(f"[INFO] Training finished in {total_time:.2f}s ({total_time / epochs * 1000:.1f} ms/epoch)")
    except KeyboardInterrupt:
        total_time = time.perf_counter() - t0_total
        print(f"\n\n[WARNING] Training interrupted by user (Ctrl + C) after {total_time:.2f}s!")
        print(f"[INFO] Preserving best model checkpoint saved up to this point.")

    # Save the overall best model
    if save_path and Path(save_path).exists():
        model.load_state_dict(torch.load(save_path, weights_only=True))

    return model, history       

# ! Change to toml and uv.lock
