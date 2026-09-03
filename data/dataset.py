"""
data/dataset.py

Purpose:
Parses raw ray-tracing .mat files from data/raw/ and synthesises the frequency-domain MIMO channel matrix CSI.npy alongside user coordinates, base station positions, AoD, and LoS labels in data/processed/.

Steps for Implementation:
- Load configuration constants and paths from config.py (RAW_DATA_DIR, PROCESSED_DATA_DIR, FC, B, NT, NR, NC, etc.).
- Load raw .mat arrays (power, phase, delay, AoD, rx_pos, tx_pos, inter).
- Filter active receivers where power is non-NaN and convert power from dBm to linear scale.
- Extract 2D coordinates (UEloc, BSloc) and dominant path Angle-of-Departure (AoD).
- Classify LoS vs NLoS using interaction counts from inter_t001_tx000_r000.mat (0 interactions = direct LoS).
- Synthesise multi-carrier MIMO channel tensors CSI of shape (N, Nc, 1, Nt) using steering vectors and subcarrier delays.
- Normalise CSI by its maximum absolute value and save CSI.npy, UEloc.npy, BSloc.npy, AoD.npy, and LoS.npy to data/processed/.
"""
# ! TODO plot overlay?

import os
import sys
import json
from pathlib import Path
import scipy.io as sio
import numpy as np

# Ensure project root is in sys.path for direct script execution
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config


def load_mat(file_path):
    """Load the first non-metadata array from a .mat file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Missing file: {file_path}")
    mat = sio.loadmat(file_path)
    for key in mat:
        if not key.startswith('__'):
            return mat[key]
    raise ValueError(f"No valid data in {file_path}")


def process_dataset():
    # Directories from config
    raw_dir = config.RAW_DATA_DIR
    out_dir = config.PROCESSED_DATA_DIR

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(config.BASE_DIR / "data", exist_ok=True)

    # Save Experiment Sweep Benchmarks from config
    np.save(out_dir / 'fading_ratio.npy', np.array(config.SWEEP_FADING_RATIO))
    np.save(out_dir / 'loc_std.npy', np.array(config.SWEEP_LOC_STD))
    np.save(out_dir / 'SNR.npy', np.array(config.SWEEP_SNR))

    # Also save to data/ root for backward compatibility
    np.save(config.BASE_DIR / 'data' / 'fading_ratio.npy', np.array(config.SWEEP_FADING_RATIO))
    np.save(config.BASE_DIR / 'data' / 'loc_std.npy', np.array(config.SWEEP_LOC_STD))
    np.save(config.BASE_DIR / 'data' / 'SNR.npy', np.array(config.SWEEP_SNR))

    # Physical parameters from config or params.json
    params_path = raw_dir / 'params.json'
    if params_path.exists():
        with open(params_path, 'r') as f:
            params = json.load(f)
        fc = params.get('rt_params', {}).get('raw_params', {}).get('waveform', {}).get('CarrierFrequency', config.FC)
        B = params.get('rt_params', {}).get('raw_params', {}).get('waveform', {}).get('bandwidth', config.B)
    else:
        fc = config.FC
        B = config.B

    Nc = config.NC
    Nt = config.NT
    Nr = config.NR
    c = config.C
    lambda_val = c / fc
    d = config.ANTENNA_SPACING

    print(f"System Configuration: fc = {fc / 1e9:.2f} GHz, Bandwidth = {B / 1e6:.1f} MHz, Nt = {Nt}, Nr = {Nr}, Nc = {Nc}")

    # Load raw data files
    power_raw = load_mat(raw_dir / 'power_t001_tx000_r000.mat')
    phase_raw = load_mat(raw_dir / 'phase_t001_tx000_r000.mat')
    delay_raw = load_mat(raw_dir / 'delay_t001_tx000_r000.mat')
    aod_raw   = load_mat(raw_dir / 'aod_az_t001_tx000_r000.mat')
    rx_pos    = load_mat(raw_dir / 'rx_pos_t001_tx000_r000.mat')
    tx_pos    = load_mat(raw_dir / 'tx_pos_t001_tx000_r000.mat')

    try:
        inter_raw = load_mat(raw_dir / 'inter_t001_tx000_r000.mat')
    except (FileNotFoundError, ValueError):
        inter_raw = None

    # Filter out inactive receivers (where all paths are NaN)
    active_mask = np.any(~np.isnan(power_raw), axis=1)
    power_raw = power_raw[active_mask]
    phase_raw = phase_raw[active_mask]
    delay_raw = delay_raw[active_mask]
    aod_raw   = aod_raw[active_mask]
    rx_pos    = rx_pos[active_mask]
    if inter_raw is not None:
        inter_raw = inter_raw[active_mask]

    N_ue, L_max = power_raw.shape
    print(f"Active User Points: {N_ue}, Max Multipath per User: {L_max}")

    # Coordinates
    UEloc = rx_pos[:, :2]
    BSloc = tx_pos[:, :2]

    # Dominant path Angle-of-Departure (AoD)
    dominant_idx = np.nanargmax(power_raw, axis=1)
    AoD = aod_raw[np.arange(N_ue), dominant_idx]

    # Line-of-Sight (LoS) classification (0 interactions on dominant path = LoS)
    if inter_raw is not None and not np.all(np.isnan(inter_raw)):
        dominant_inter = inter_raw[np.arange(N_ue), dominant_idx]
        LoS = (dominant_inter == 0).astype(np.int64)
    else:
        LoS = np.ones(N_ue, dtype=np.int64)

    # Multi-carrier MIMO Channel Synthesis
    # Baseband subcarrier frequency offsets (phase_raw already contains carrier phase -2*pi*fc*tau)
    subcarrier_offsets = np.arange(Nc) * (B / Nc)
    antenna_indices = np.arange(Nt)

    # Valid path mask and conversion to linear power (dBm -> linear)
    valid_paths = ~np.isnan(power_raw)
    power_lin = np.zeros_like(power_raw)
    power_lin[valid_paths] = 10.0 ** (power_raw[valid_paths] / 10.0)

    phase_rad = np.zeros_like(phase_raw)
    phase_rad[valid_paths] = phase_raw[valid_paths] * (np.pi / 180.0)

    aod_rad = np.zeros_like(aod_raw)
    aod_rad[valid_paths] = aod_raw[valid_paths] * (np.pi / 180.0)

    delay_val = np.zeros_like(delay_raw)
    delay_val[valid_paths] = delay_raw[valid_paths]

    # Complex path gain alpha: sqrt(power) * exp(j * phase)
    alpha = np.where(valid_paths, np.sqrt(power_lin) * np.exp(1j * phase_rad), 0.0 + 0.0j)

    # Subcarrier delay matrix: exp(-j * 2 * pi * delta_f * tau) -> shape (N_ue, L_max, Nc)
    delay_phase = -2j * np.pi * np.einsum('f,ul->ulf', subcarrier_offsets, delay_val)
    delay_matrix = np.where(valid_paths[:, :, None], np.exp(delay_phase), 0.0 + 0.0j)

    # Base station array steering matrix: exp(-j * 2 * pi * m * d * sin(AoD) / lambda) -> shape (N_ue, L_max, Nt)
    steering_phase = 2j * np.pi * (d / lambda_val) * np.einsum('t,ul->ult', antenna_indices, np.sin(aod_rad))
    steering_matrix = np.where(valid_paths[:, :, None], np.exp(steering_phase), 0.0 + 0.0j)

    # Channel tensor per user: sum across L_max paths -> shape (N_ue, Nc, Nt)
    CSI_2d = np.einsum('ul,ulf,ult->uft', alpha, delay_matrix, steering_matrix)

    # Reshape to (N_ue, Nc, Nr, Nt) with Nr = 1
    CSI = CSI_2d[:, :, None, :]

    # Normalize CSI by max absolute value
    max_c = np.max(np.abs(CSI))
    if max_c > 0:
        CSI = CSI / max_c
    CSI = CSI.astype(np.complex64)

    # Save to data/processed/
    np.save(out_dir / 'CSI.npy', CSI)
    np.save(out_dir / 'UEloc.npy', UEloc)
    np.save(out_dir / 'BSloc.npy', BSloc)
    np.save(out_dir / 'AoD.npy', AoD)
    np.save(out_dir / 'LoS.npy', LoS)

    print(f"\nProcessing Complete. Saved to: {out_dir}")
    print(f"  CSI.npy   : Shape {CSI.shape}, Dtype {CSI.dtype}")
    print(f"  UEloc.npy : Shape {UEloc.shape}")
    print(f"  BSloc.npy : Shape {BSloc.shape}, Value {BSloc}")
    print(f"  AoD.npy   : Shape {AoD.shape}")
    print(f"  LoS.npy   : Line-of-sight users = {np.sum(LoS)} / {N_ue} ({np.mean(LoS)*100:.1f}%)")


if __name__ == "__main__":
    process_dataset()
