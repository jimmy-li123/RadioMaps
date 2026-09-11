# Reduced Pilots and Radio Map integration script
# Data from both user location and subset of pilot signal to build a better beamforming vector

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
from models.radio_map import RadioMapNet
from models.reduced_pilot import ReducedPilotNet
from models.integrate import IntegrationNet

def main():
    args = parse_args(description="Train and evaluate integration model", default_epochs=config.EPOCHS_INT)

    # Parameters
    dataset = args.dataset
    loc_std = args.loc_std
    fading_ratio = args.fading_ratio
    SNR = args.snr
    Nc = config.NC
    Nt = config.NT
    eta = args.eta
    device = config.DEVICE
    
    # Load Arrays
    UEloc, CSI, LoS = load_dataset(dataset)
    UEloc, CSI = eliminate_block(UEloc, CSI)
    CSI_fading = add_fading(CSI, fading_ratio)

    x_loc = prepare_loc_features(UEloc, loc_std)
    x_pilot = prepare_pilot_features(CSI_fading, SNR, eta)
    train_idx, val_idx, test_idx = split_indices(len(UEloc))
    
    CSI_noise = add_noise(CSI_fading, SNR).astype(np.complex64)
    y_train = CSI_noise[train_idx]
    y_val = CSI_noise[val_idx]
    CSI_test = CSI_fading[test_idx]
    LoS_test = LoS[test_idx]

    power = (np.linalg.norm(CSI_fading[train_idx]))**2 / np.prod(CSI_fading[train_idx].shape)
    noise = power / (10 ** (SNR / 10))

    # Load pretrained upstream component models
    model_rm = load_pretrained_model(RadioMapNet, "RM", dataset, device=device)
    model_rp = load_pretrained_model(ReducedPilotNet, "RP", dataset, eta=eta, device=device)

    x_train_int = prepare_integration_features(model_rm, model_rp, x_loc[train_idx], x_pilot[train_idx], device)
    x_val_int = prepare_integration_features(model_rm, model_rp, x_loc[val_idx], x_pilot[val_idx], device)
    x_test_int = prepare_integration_features(model_rm, model_rp, x_loc[test_idx], x_pilot[test_idx], device)

    # Check if IntegrationNet weights are already computed
    model = IntegrationNet(Nc, Nt).to(device)
    save_path, history_path = get_model_paths("INT", dataset, eta=eta)

    if not load_weights_if_available(model, save_path, history_path, args.force_train, device):
        print(f"Training IntegrationNet ({dataset}, eta={eta}%) for {args.epochs} epochs...")
        model, history = train_model(
            model=model,
            x_train=x_train_int,
            y_train=y_train,
            x_val=x_val_int,
            y_val=y_val,
            noise=noise,
            args=args,
            save_path=save_path,
            device=device,
        )
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)
        rel_hist = history_path.relative_to(PROJECT_ROOT) if history_path.is_relative_to(PROJECT_ROOT) else history_path
        print(f"Training complete: saved training history to {rel_hist}")

    # Evaluate model on test set
    model.eval()
    with torch.no_grad():
        x_test_tensor = torch.as_tensor(x_test_int, dtype=torch.float32).to(device)
        V_INT_test = model(x_test_tensor).cpu().numpy()

    V_INT_test = normalise_V(V_INT_test)

    # Calculate SE and optimum
    INT_SE = cal_SE(CSI_test, V_INT_test, noise)
    opt_SE = cal_opt_SE(CSI_test, noise)

    # Pilot overhead penalty: rho = (eta/100) * (16 / (Nc * Ns))
    rho = (eta / 100.0) * (16.0 / (Nc * config.NS))
    INT_net_SE = INT_SE * (1.0 - rho)

    INT2opt_raw = np.mean(INT_SE) / np.mean(opt_SE) * 100
    INT2opt_net = np.mean(INT_net_SE) / np.mean(opt_SE) * 100

    INT_LoS_SE, INT_NLoS_SE = compute(INT_net_SE, LoS_test)
    opt_LoS_SE, opt_NLoS_SE = compute(opt_SE, LoS_test)

    INT2opt_LoS = (INT_LoS_SE / opt_LoS_SE) * 100 if opt_LoS_SE > 0 else 0.0
    INT2opt_NLoS = (INT_NLoS_SE / opt_NLoS_SE) * 100 if opt_NLoS_SE > 0 else 0.0

    print("=" * 50)
    print(f"Integration Net Results ({dataset.upper()} @ SNR={SNR}dB, eta={eta}%)")
    print("=" * 50)
    print(f"Pilot overhead rho (eta={eta}%): {rho * 100:.2f}%")
    print(f"Raw INT SE:   {np.mean(INT_SE):.3f} bps/Hz")
    print(f"Net INT SE:   {np.mean(INT_net_SE):.3f} bps/Hz")
    print(f"opt_SE:       {np.mean(opt_SE):.3f} bps/Hz")
    print(f"Raw INT2opt:  {INT2opt_raw:.3f} %")
    print(f"Net INT2opt:  {INT2opt_net:.3f} %")
    print(f"Net INT LoS:  {INT2opt_LoS:.3f} %")
    print(f"Net INT NLoS: {INT2opt_NLoS:.3f} %")

if __name__ == "__main__":
    main()
    
    
"""
==================================================
Integration Net Results (SYDNEY @ SNR=15.0dB, eta=50%)
==================================================
Pilot overhead rho (eta=50%): 4.76%
Raw INT SE:   7.265 bps/Hz
Net INT SE:   6.919 bps/Hz
opt_SE:       7.853 bps/Hz
Raw INT2opt:  92.506 %
Net INT2opt:  88.101 %
Net INT LoS:  89.080 %
Net INT NLoS: 72.965 %
"""

# Radio maps comtributes much more than reduced pilots did here, as it learns the 3D building morphology better