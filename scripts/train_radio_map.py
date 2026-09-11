# Radio Maps model script
# Note this is worse than CKM, but is much faster, smaller and more versatile

# ! Time and verify Apparently much faster (12x ish) by torch c++ execution and avoids svd

import sys
import argparse
from pathlib import Path
import numpy as np
import math
import torch
import json
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from utils.metrics import *
from models.radio_map import RadioMapNet

def main():
    args = parse_args(description="Train and evaluate Radio Map model", default_epochs=config.EPOCHS_RM)
    
    # Parameters
    loc_std      = args.loc_std
    fading_ratio = args.fading_ratio
    SNR          = args.snr
    Nc           = config.NC
    Nt           = config.NT
    device       = config.DEVICE

    # Load data arrays
    UEloc, CSI, LoS = load_dataset(args.dataset)

    # Format and add realistic errors
    UEloc, CSI = eliminate_block(UEloc, CSI)
    loc_norm = prepare_loc_features(UEloc, loc_std)
    CSI_fading = add_fading(CSI, fading_ratio)
    CSI_noise = add_noise(CSI_fading, SNR).astype(np.complex64)

    # Deterministic train/val/test split
    train_idx, val_idx, test_idx = split_indices(len(UEloc))

    x_train, x_val = loc_norm[train_idx], loc_norm[val_idx]
    y_train, y_val = CSI_noise[train_idx], CSI_noise[val_idx]           # Noisy CSI for training loss
    x_test = loc_norm[test_idx]
    CSI_train, CSI_test = CSI_fading[train_idx], CSI_fading[test_idx]   # True physical CSI for eval
    LoS_test = LoS[test_idx]

    # Calculate power and noise
    power = (np.linalg.norm(CSI_train))**2/np.prod(CSI_train.shape)     # Changed to CSI_train to avoid leaking into test
    noise = power / (10 ** (SNR / 10))

    # Check if weights are already computed
    model = RadioMapNet(Nc, Nt).to(device)
    save_path, history_path = get_model_paths("RM", args.dataset)

    if not load_weights_if_available(model, save_path, history_path, args.force_train, device):
        print(f"Training RadioMapNet ({args.dataset}) for {args.epochs} epochs...")
        model, history = train_model(
            model = model,
            x_train = x_train,
            y_train = y_train,
            x_val = x_val,
            y_val = y_val,
            noise = noise,
            args = args,
            save_path = save_path,
            device = device
        )
        # Save training history to outputs
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)
        rel_hist = history_path.relative_to(PROJECT_ROOT) if history_path.is_relative_to(PROJECT_ROOT) else history_path
        print(f"Training complete: saved training history to: {rel_hist}")
    
    # Evaluate model
    model.eval()
    with torch.no_grad():
        x_test_tensor = torch.as_tensor(x_test, dtype=torch.float32).to(device)
        V_RM_test = model(x_test_tensor).cpu().numpy()
        
    V_RM_test = normalise_V(V_RM_test)  # Should already be normalised

    # Calculate SE and optimum
    RM_SE = cal_SE(CSI_test, V_RM_test, noise)
    opt_SE = cal_opt_SE(CSI_test, noise)
    
    RM2opt = np.mean(RM_SE) / np.mean(opt_SE) * 100
    
    # Compute LoS and NLoS metrics                                                                       
    RM_LoS_SE, RM_NLoS_SE = compute(RM_SE, LoS_test)                                                  
    opt_LoS_SE, opt_NLoS_SE = compute(opt_SE, LoS_test)                                                  
                                                                                                            
    RM2opt_LoS = (RM_LoS_SE / opt_LoS_SE) * 100 if opt_LoS_SE > 0 else 0.0                             
    RM2opt_NLoS = (RM_NLoS_SE / opt_NLoS_SE) * 100 if opt_NLoS_SE > 0 else 0.0  

    # Print result
    print("=" * 50)
    print(f"Radio Map Results ({args.dataset.upper()} @ SNR={SNR}dB, loc_std={loc_std}m)")
    print("=" * 50)
    print(f"Pilot overhead rho: 0.00% (zero overhead)")
    print(f"Raw RM SE:    {np.mean(RM_SE):.3f} bps/Hz")
    print(f"opt_SE:       {np.mean(opt_SE):.3f} bps/Hz")
    print(f"RM2opt (Raw): {RM2opt:.3f} %")
    print(f"RM LoS:       {RM2opt_LoS:.3f} %")
    print(f"RM NLoS:      {RM2opt_NLoS:.3f} %")

if __name__ == '__main__':
    main()
    
"""
==================================================                                                                                     
Radio Map Results (SYDNEY @ SNR=15.0dB, loc_std=1.0m)                                                                                  
==================================================                                                                                     
Pilot overhead rho: 0.00% (zero overhead)                                                                                              
Raw RM SE:    7.229 bps/Hz                                                                                                             
opt_SE:       7.853 bps/Hz                                                                                                             
RM2opt (Raw): 92.055 % 
RM LoS:       93.079 %                                                                                                                 
RM NLoS:      76.206 %                  
"""