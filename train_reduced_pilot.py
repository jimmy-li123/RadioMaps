# Reduced Pilots model script

import sys
import argparse
from pathlib import Path
import numpy as np
import torch
import json
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[0]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
import config
from utils.metrics import *
from models.reduced_pilot import ReducedPilotNet

def mask_pilots(CSI_pilot, eta):
    masked = CSI_pilot.copy()
    if eta == 100:
        return masked
    elif eta == 75:
        masked[:, 3::4] = 0
    elif eta == 50:
        masked[:, 1::2] = 0
    elif eta == 25: # Keep every 4th antenna
        masked[:, 1::2] = 0
        masked[:, 2::4] = 0
        masked[:, 3::4] = 0
    else:
        raise ValueError('Please choose from 100,75,50,25.')

    return masked

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
    CSI_RB = np.repeat(CSI_fading, Ns, axis=2)    # Resource block = (N, Nc, Ns = OFDM symbols, Nt)
    CSI_noise = add_noise(CSI_RB, SNR)

    # Pilot Measurement
    Nc_pos = np.array([2, 3, 4, 5, 8, 9, 10, 11])     # Active subcarrier indices
    
    ant_idx = np.arange(Nt)     # 4 * j + k
    j = ant_idx // 4
    k = ant_idx % 4

    sub_idx = Nc_pos[j]
    sym_idx = k + 3
    
    CSI_pilot = CSI_noise[:, sub_idx, sym_idx, ant_idx]

    CSI_pilot = mask_pilots(CSI_pilot, eta)
    norm = np.linalg.norm(CSI_pilot, axis=-1, keepdims=True)
    norm = np.where(norm == 0, 1e-12, norm)
    reduced_pilot = (CSI_pilot / norm)[:, None, :].repeat(Nc, axis=1)
    pilot_comp = np.concatenate([
        np.real(reduced_pilot[:, :, :, None]),
        np.imag(reduced_pilot[:, :, :, None]),
    ], axis=-1)  # (N, Nc, Nt, 2)
    
    CSI_noise = add_noise(CSI_fading, SNR).astype(np.complex64)
    
    # Divide into train, validation and test set
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

    x_train, x_val = pilot_comp[train_idx], pilot_comp[val_idx]
    y_train, y_val = CSI_noise[train_idx], CSI_noise[val_idx]
    x_test = pilot_comp[test_idx]
    CSI_train, CSI_test = CSI_fading[train_idx], CSI_fading[test_idx]
    LoS_test = LoS[test_idx]
    N_train, N_test = x_train.shape[0], x_test.shape[0]

    # Noise
    power = (np.linalg.norm(CSI_train))**2/np.prod(CSI_train.shape)
    noise = power / (10**(SNR/10))

    # Check if weights are already computed
    config.SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    save_path = config.SAVED_MODELS_DIR / f"RP_{args.dataset}_{int(eta)}.pth"
    if not save_path.exists() and args.dataset == "sydney" and (config.SAVED_MODELS_DIR / f"RP_{int(eta)}.pth").exists():
        save_path = config.SAVED_MODELS_DIR / f"RP_{int(eta)}.pth"

    model = ReducedPilotNet(Nc, Nt)
    history_path = config.OUTPUTS_DIR / f"RP_{args.dataset}_{int(eta)}_history.json"
    if not history_path.exists() and args.dataset == "sydney" and (config.OUTPUTS_DIR / f"RP_{int(eta)}_history.json").exists():
        history_path = config.OUTPUTS_DIR / f"RP_{int(eta)}_history.json"

    if save_path.exists() and not args.force_train:
        model = model.to(device)
        rel_save = save_path.relative_to(PROJECT_ROOT) if save_path.is_relative_to(PROJECT_ROOT) else save_path
        print(f"Found pre-trained weights at: {rel_save} (use --force-train to retrain)...")
        model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
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
    RP_LoS_SE, RP_NLoS_SE = compute(RP_net_SE, LoS_test)
    opt_LoS_SE, opt_NLoS_SE = compute(opt_SE, LoS_test)

    RP2opt_LoS = (RP_LoS_SE / opt_LoS_SE) * 100 if opt_LoS_SE > 0 else 0.0
    RP2opt_NLoS = (RP_NLoS_SE / opt_NLoS_SE) * 100 if opt_NLoS_SE > 0 else 0.0

    # Print results
    print(f'Pilot overhead rho (eta={eta}%): {rho*100:.2f}%')
    print('Raw RP SE:   ', np.round(np.mean(RP_SE), 3), 'bps/Hz')
    print('Net RP SE:   ', np.round(np.mean(RP_net_SE), 3), 'bps/Hz')
    print('opt_SE:      ', np.round(np.mean(opt_SE), 3), 'bps/Hz')
    print('Net RP2opt:  ', np.round(RP2opt_net, 3), '%')
    print('Net RP LoS:  ', np.round(RP2opt_LoS, 3), '%')
    print('Net RP NLoS: ', np.round(RP2opt_NLoS, 3), '%')

if __name__ == "__main__":
    main()

# ! Need to verify somehow
"""
With SNR = 15:
Pilot overhead rho (eta=50%): 4.76%
Raw RP SE:    5.105 bps/Hz
Net RP SE:    4.862 bps/Hz
opt_SE:       7.853 bps/Hz
Net RP2opt:   61.908 %
Net RP LoS:   64.528 %
Net RP NLoS:  21.39 %

Compared to the original paper:
    - Lower SNR
    - the RP2opt doesn't include the resource penalty
    - Different scenario

Therefore:
    - RM is better when memorising 3D building layout, but fails when theres GPS positioning error or dynamic fading
    - RP measures real-time channel fading, but suffers in complex multipath as 16 causes spatial alisasing and pilot overhead penalties

    - The best will be a combination of both

"""
