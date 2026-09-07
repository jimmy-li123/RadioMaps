# Radio Maps model script
# ? add argument when running script
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

PROJECT_ROOT = Path(__file__).resolve().parents[0]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from utils.metrics import *
from models.radio_map import RadioMapNet

def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate Radio Map model")

    parser.add_argument("--force-train", action="store_true", help="Force retraining even if saved weights exist.")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS_RM, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE, help="Training batch size.")
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE, help="Learning rate.")

    return parser.parse_args()

def main():
    args = parse_args()
    
    # parameters
    loc_std      = config.DEFAULT_LOC_STD
    fading_ratio = config.DEFAULT_FADING_RATIO
    SNR          = config.DEFAULT_SNR
    Nc           = config.NC
    Nt           = config.NT
    device       = config.DEVICE

    # Load data arrays explicitly
    UEloc = np.load(config.PROCESSED_DATA_DIR / 'UEloc.npy')                                             
    BSloc = np.load(config.PROCESSED_DATA_DIR / 'BSloc.npy')                                             
    CSI   = np.load(config.PROCESSED_DATA_DIR / 'CSI.npy')                                               
    LoS   = np.load(config.PROCESSED_DATA_DIR / 'LoS.npy') 

    # Format and add realistic errors
    UEloc, CSI = eliminate_block(UEloc, CSI)
    UEloc_error = add_error(UEloc, loc_std)
    loc_norm = normalise_loc(UEloc_error)
    CSI_fading = add_fading(CSI, fading_ratio)
    CSI_noise = add_noise(CSI_fading, SNR).astype(np.complex64)

    # Divide train, valdiation and test set 
    N_samples = UEloc.shape[0]

    # First split into clean 20% test set
    train_val_idx, test_idx = train_test_split(
        np.arange(N_samples), 
        test_size=config.TEST_SIZE, 
        random_state=config.RANDOM_SEED
    )

    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=config.VAL_SIZE,
        random_state=config.RANDOM_SEED
    )   

    x_train, x_val = loc_norm[train_idx], loc_norm[val_idx]
    y_train, y_val = CSI_noise[train_idx], CSI_noise[val_idx]           # Noisy CSI for training loss
    x_test = loc_norm[test_idx]
    CSI_train, CSI_test = CSI_fading[train_idx], CSI_fading[test_idx]   # True physical CSI for eval
    LoS_test = LoS[test_idx]

    # Calculate power and noise
    power = (np.linalg.norm(CSI_train))**2/np.prod(CSI_train.shape)     # Changed to CSI_train to avoid leaking into test
    noise = power / (10 ** (SNR / 10))

    # Check if weights are already computed
    config.SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    save_path = config.SAVED_MODELS_DIR / "RM.pth"
    
    model = RadioMapNet(Nc, Nt)
    
    history_path = config.OUTPUTS_DIR / "RM_history.json"

    if save_path.exists() and not args.force_train:
        rel_save = save_path.relative_to(PROJECT_ROOT) if save_path.is_relative_to(PROJECT_ROOT) else save_path
        print(f"Found pre-trained weights at: {rel_save} (use --force-train to retrain)...")
        model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
        model = model.to(device)
        if history_path.exists() and history_path.stat().st_size > 0:
            try:
                with open(history_path, "r") as f:
                    hist = json.load(f)
                    if "val_loss" in hist and len(hist["val_loss"]) > 0:
                        best_val = min(hist["val_loss"])
                        print(f"Loaded model best validation SE: {-best_val:.3f} bps/Hz")
            except (json.JSONDecodeError, Exception):
                pass
    else:
        print(f"Training RadioMapNet for {args.epochs} epochs...")
        model, history = train_model(
            model = model,
            x_train = x_train,
            y_train = y_train,
            x_test = x_val,
            y_test = y_val,
            noise = noise,
            epochs = args.epochs,
            batch_size = args.batch_size,
            lr = args.lr,
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
    print('RM2opt:', np.round(RM2opt, 3), '%')
    print('RM LoS:', np.round(RM2opt_LoS, 3), '%')
    print('RM NLoS:', np.round(RM2opt_NLoS, 3), '%')

if __name__ == '__main__':
    main()
    
    
# RM2opt: 92.933 %
# RM LoS: 94.02 %
# RM NLoS: 76.127 %