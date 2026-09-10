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
import argparse
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

# =====================================================================      
# Dataset Loading & CLI Parsing                                              
# ===================================================================== 
def load_dataset(dataset_name=None, return_aod=False, return_bsloc=False, eliminate_blocked=True):
    """
    Loads dataset arrays (UEloc, CSI, LoS, and optionally BSloc, AoD) from 'sydney' or any extra simulation (e.g. 'o1').
    Ensures standard array shapes across all simulations:
      - UEloc: (N, 2)
      - CSI:   (N, Nc, 1, Nt)
      - LoS:   (N,)
      - BSloc: (1, 2) [optional]
      - AoD:   (N,)   [optional]
    """
    dataset_dir = config.get_dataset_dir(dataset_name)

    UEloc = np.load(dataset_dir / "UEloc.npy")
    CSI   = np.load(dataset_dir / "CSI.npy")
    LoS   = np.load(dataset_dir / "LoS.npy")

    # Standardize CSI shape to (N, Nc, 1, Nt) if needed
    if CSI.ndim == 4 and CSI.shape[1] == config.NC and CSI.shape[2] == 1 and CSI.shape[3] == config.NT:
        pass
    elif CSI.ndim == 4 and CSI.shape[1] == 1 and CSI.shape[2] == config.NT and CSI.shape[3] == config.NC:
        CSI = np.transpose(CSI, (0, 3, 1, 2))

    BSloc = None
    if return_bsloc or (dataset_dir / "BSloc.npy").exists():
        try:
            BSloc = np.load(dataset_dir / "BSloc.npy")
            if BSloc.ndim == 1:
                BSloc = BSloc[None, :]
            if BSloc.shape[-1] > 2:
                BSloc = BSloc[:, :2]
        except Exception:
            BSloc = None

    AoD = None
    if return_aod or (dataset_dir / "AoD.npy").exists():
        try:
            AoD = np.load(dataset_dir / "AoD.npy")
        except Exception:
            AoD = None

    if eliminate_blocked:
        valid_mask = (LoS != -1)
        power = np.abs(np.sum(CSI.reshape(CSI.shape[0], -1), axis=1))
        valid_mask = valid_mask & (power > 1e-15)
        if not np.all(valid_mask):
            UEloc = UEloc[valid_mask]
            CSI = CSI[valid_mask]
            LoS = LoS[valid_mask]
            if AoD is not None:
                AoD = AoD[valid_mask]

    ret = [UEloc, CSI, LoS]
    if return_bsloc:
        ret.append(BSloc)
    if return_aod:
        ret.append(AoD)

    return tuple(ret) if len(ret) > 1 else ret[0]


def parse_args(description="Train and evaluate MIMO beamforming model", 
               default_epochs=None, 
               default_eta=None,
               default_dataset=None):
    """
    Unified CLI parser reused across all training and baseline models.
    """
    if default_dataset is None:
        default_dataset = config.DEFAULT_DATASET
    if default_epochs is None:
        default_epochs = config.EPOCHS_RM

    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("--dataset", type=str, default=default_dataset,
                        help=f"Dataset name to train/evaluate on (e.g. 'sydney', 'o1'). Default: '{default_dataset}'.")
    parser.add_argument("--force-train", action="store_true", 
                        help="Force retraining even if saved weights exist.")
    parser.add_argument("--epochs", type=int, default=default_epochs, 
                        help=f"Number of training epochs. Default: {default_epochs}.")
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE, 
                        help=f"Training batch size. Default: {config.BATCH_SIZE}.")
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE, 
                        help=f"Learning rate. Default: {config.LEARNING_RATE}.")
    parser.add_argument("--snr", type=float, default=config.DEFAULT_SNR, 
                        help=f"SNR in dB. Default: {config.DEFAULT_SNR}.")
    parser.add_argument("--loc_std", type=float, default=config.DEFAULT_LOC_STD, 
                        help=f"GPS positioning error std in meters. Default: {config.DEFAULT_LOC_STD}.")
    parser.add_argument("--fading_ratio", type=float, default=config.DEFAULT_FADING_RATIO, 
                        help=f"Rayleigh scattering fading ratio K. Default: {config.DEFAULT_FADING_RATIO}.")
    
    eta_val = default_eta if default_eta is not None else config.DEFAULT_ETA
    parser.add_argument("--eta", type=int, default=eta_val, choices=[25, 50, 75, 100], 
                        help=f"Reduced pilot density percentage (25, 50, 75, 100). Default: {eta_val}.")

    return parser.parse_args()


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
        
def train_model(model, x_train, y_train, x_val=None, y_val=None, noise=None, 
                epochs=1000, batch_size=128, lr=1e-3, save_path=None, device=None,
                args=None, x_test=None, y_test=None):
    """
    Universal training loop for models with support for unified CLI args,
    device selection, validation tracking, ETA estimation, and model checkpointing.
    """
    # Override defaults if args Namespace is provided
    if args is not None:
        if hasattr(args, 'epochs') and args.epochs is not None:
            epochs = args.epochs
        if hasattr(args, 'batch_size') and args.batch_size is not None:
            batch_size = args.batch_size
        if hasattr(args, 'lr') and args.lr is not None:
            lr = args.lr

    # Handle alias x_test/y_test for backward compatibility
    if x_val is None and x_test is not None:
        x_val = x_test
    if y_val is None and y_test is not None:
        y_val = y_test

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
    x_vl = torch.as_tensor(x_val, dtype=torch.float32)
    y_vl = torch.as_tensor(y_val, dtype=torch.complex64)

    # Separate into batches
    train_dataset = TensorDataset(x_tr, y_tr)       # Separates input into paired samples
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)   # Split into shuffled mini batches

    best_val_loss = float('inf')    # Initial loss
    history = {'train_loss': [], 'val_loss': []}
    
    print(f"Starting training on {device} ({epochs} epochs, batch_size={batch_size}, lr={lr:.1e})...")

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

            elapsed_total = time.perf_counter() - t0_total
            sec_per_epoch = elapsed_total / epoch
            ms_per_epoch = sec_per_epoch * 1000.0
            remaining_sec = sec_per_epoch * (epochs - epoch)
            if remaining_sec >= 60:
                eta_str = f"{int(remaining_sec // 60)}m {int(remaining_sec % 60):02d}s"
            else:
                eta_str = f"{remaining_sec:.1f}s"
            
            if epoch == 1 or epoch % 50 == 0 or epoch == epochs:
                current_lr = optimiser.param_groups[0]['lr']
                elapsed_50 = time.perf_counter() - t0_interval
                print(f"\rEpoch [{epoch:4d}/{epochs:4d}] - Train SE: {-train_loss:.3f} bps/Hz | Val SE: {-val_loss:.3f} bps/Hz | LR: {current_lr:.1e} | {ms_per_epoch:.1f}ms/ep | ETA: {eta_str}")
                t0_interval = time.perf_counter()
            else:
                print(f"\rEpoch [{epoch:4d}/{epochs:4d}] ({ms_per_epoch:.1f}ms/ep, ETA: {eta_str})...", end="", flush=True)
        print()
        total_time = time.perf_counter() - t0_total
        print(f"[INFO] Training finished in {total_time:.2f}s ({total_time / epochs * 1000:.1f} ms/epoch)")
    except KeyboardInterrupt:
        total_time = time.perf_counter() - t0_total
        print(f"\n\n[WARNING] Training interrupted by user (Ctrl + C) after {total_time:.2f}s!")
        print(f"[INFO] Preserving best model checkpoint saved up to this point.")

    # Save the overall best model
    if save_path and Path(save_path).exists():
        model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))

    return model, history       
