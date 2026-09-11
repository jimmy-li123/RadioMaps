# Reduced Pilots model script

import sys
import argparse
from pathlib import Path
import numpy as np
import torch
import json
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
import config
from utils.metrics import *
from models.reduced_pilot import ReducedPilotNet

def main():
    args = parse_args(description="Train and evaluate Reduced Pilots model", default_epochs=config.EPOCHS_RP, default_eta=config.DEFAULT_ETA)
    
    # Parameters
    fading_ratio = args.fading_ratio
    SNR = args.snr
    eta = args.eta
    Ns = config.NS
    Nt = config.NT
    Nc = config.NC
    device = config.DEVICE
    
    # Load arrays
    UEloc, CSI, LoS = load_dataset(args.dataset)
    
    # Adding errors
    UEloc, CSI = eliminate_block(UEloc, CSI)
    CSI_fading = add_fading(CSI, fading_ratio)
    CSI_noise = add_noise(CSI_fading, SNR).astype(np.complex64)

    # Pilot Measurement & Features via reusable helper
    x_pilot = prepare_pilot_features(CSI_fading, SNR, eta, Nc, Ns, Nt)

    # Deterministic train/val/test split
    train_idx, val_idx, test_idx = split_indices(len(UEloc))

    x_train, x_val = x_pilot[train_idx], x_pilot[val_idx]
    y_train, y_val = CSI_noise[train_idx], CSI_noise[val_idx]
    x_test = x_pilot[test_idx]
    CSI_train, CSI_test = CSI_fading[train_idx], CSI_fading[test_idx]
    LoS_test = LoS[test_idx]
    N_train, N_test = x_train.shape[0], x_test.shape[0]

    # Noise
    power = (np.linalg.norm(CSI_train))**2/np.prod(CSI_train.shape)
    noise = power / (10**(SNR/10))

    # Check if weights are already computed
    model = ReducedPilotNet(Nc, Nt).to(device)
    save_path, history_path = get_model_paths("RP", args.dataset, eta=eta)

    if not load_weights_if_available(model, save_path, history_path, args.force_train, device):
        print(f"Training ReducedPilotNet ({args.dataset}, eta={eta}%) for {args.epochs} epochs...")
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

        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)
        rel_hist = history_path.relative_to(PROJECT_ROOT) if history_path.is_relative_to(PROJECT_ROOT) else history_path
        print(f"Training complete: saved training history to {rel_hist}")

    # Evaluate model
    model.eval()
    with torch.no_grad():
        x_test_tensor = torch.as_tensor(x_test, dtype=torch.float32).to(device)
        V_RP_test = model(x_test_tensor).cpu().numpy()
    
    V_RP_test = normalise_V(V_RP_test)

    # Calculate raw SE and optimum SE
    RP_SE = cal_SE(CSI_test, V_RP_test, noise)
    opt_SE = cal_opt_SE(CSI_test, noise)
    
    # ? Pilot overhead penalty: rho = (eta/100) * (16 / (Nc * Ns))
    # 16 pilot REs for 32 antennas (2 antennas share 1 RE) out of 168 total REs
    rho = (eta / 100.0) * (16.0 / (Nc * Ns))
    RP_net_SE = RP_SE * (1.0 - rho)

    RP2opt_raw = np.mean(RP_SE) / np.mean(opt_SE) * 100
    RP2opt_net = np.mean(RP_net_SE) / np.mean(opt_SE) * 100

    # Compute LoS and NLoS metrics
    RP_net_LoS, RP_net_NLoS = compute(RP_net_SE, LoS_test)
    RP_raw_LoS, RP_raw_NLoS = compute(RP_SE, LoS_test)
    opt_LoS_SE, opt_NLoS_SE = compute(opt_SE, LoS_test)

    RP2opt_raw_LoS = (RP_raw_LoS / opt_LoS_SE) * 100 if opt_LoS_SE > 0 else 0.0
    RP2opt_raw_NLoS = (RP_raw_NLoS / opt_NLoS_SE) * 100 if opt_NLoS_SE > 0 else 0.0
    RP2opt_net_LoS = (RP_net_LoS / opt_LoS_SE) * 100 if opt_LoS_SE > 0 else 0.0
    RP2opt_net_NLoS = (RP_net_NLoS / opt_NLoS_SE) * 100 if opt_NLoS_SE > 0 else 0.0

    # Print results
    print("=" * 50)
    print(f"Reduced Pilots Results ({args.dataset.upper()} @ SNR={SNR}dB, eta={eta}%)")
    print("=" * 50)
    print(f"Pilot overhead rho (eta={eta}%): {rho*100:.2f}%")
    print(f"Raw RP SE:    {np.mean(RP_SE):.3f} bps/Hz")
    print(f"Net RP SE:    {np.mean(RP_net_SE):.3f} bps/Hz")
    print(f"opt_SE:       {np.mean(opt_SE):.3f} bps/Hz")
    print(f"Raw RP2opt:   {RP2opt_raw:.3f} %")
    print(f"Net RP2opt:   {RP2opt_net:.3f} %")
    print(f"Net RP LoS:   {RP2opt_net_LoS:.3f} %")
    print(f"Net RP NLoS:  {RP2opt_net_NLoS:.3f} %")

if __name__ == "__main__":
    main()

# ! Need to verify somehow
"""
==================================================                                                                                     
Reduced Pilots Results (SYDNEY @ SNR=15.0dB, eta=50%)                                                                                  
==================================================                                                                                     
Pilot overhead rho (eta=50%): 4.76%                                                                                                    
Raw RP SE:    5.105 bps/Hz                                                                                                             
Net RP SE:    4.862 bps/Hz                                                                                                             
opt_SE:       7.853 bps/Hz                                                                                                             
Raw RP2opt:   65.004 %                                                                                                                 
Net RP2opt:   61.908 %   
Net RP LoS:   64.528 %
Net RP NLoS:  21.390 %

Compared to the original paper:
    - Lower SNR
    - the RP2opt doesn't include the resource penalty
    - Different scenario

Therefore:
    - RM is better when memorising 3D building layout, but fails when theres GPS positioning error or dynamic fading
    - RP measures real-time channel fading, but suffers in complex multipath as 16 causes spatial alisasing and pilot overhead penalties

    - The best will be a combination of both

"""
