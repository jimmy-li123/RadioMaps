# Helper functions for training scripts

from pathlib import Path
import numpy as np
import torch
from sklearn.model_selection import train_test_split

import config
from utils.metrics import (
    add_error,
    normalise_loc,
    add_noise,
    normalise_V,
)
# =====================================================================
# Data and Feature Processing
# =====================================================================

def split_indices(n_samples: int,
                  test_size: float = config.TEST_SIZE,
                  val_size: float = config.VAL_SIZE,
                  random_state: int = config.RANDOM_SEED):
    """
    Deterministically splits n_samples into train, validation, and test indices.
    Stage 1: Split into (1 - test_size) train_val and test_size test.
    Stage 2: Split train_val into (1 - val_size) train and val_size val.

    Returns:
        train_idx, val_idx, test_idx: 1D numpy integer index arrays.
    """
    all_indices = np.arange(n_samples)

    train_val_idx, test_idx = train_test_split(
        all_indices,
        test_size=test_size,
        random_state=random_state
    )

    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_size,
        random_state=random_state
    )

    return train_idx, val_idx, test_idx


def prepare_loc_features(UEloc: np.ndarray,
                         loc_std: float = config.DEFAULT_LOC_STD) -> np.ndarray:
    """
    Applies GPS positioning error and normalizes coordinates to [0, 1].
    Input:
        UEloc: (N, 2) raw user coordinates.
        loc_std: GPS noise standard deviation in meters.
    Returns:
        loc_norm: (N, 2) normalized noisy coordinates for RadioMapNet.
    """
    UEloc_error = add_error(UEloc, loc_std)
    return normalise_loc(UEloc_error).astype(np.float32)


def mask_pilots(CSI_pilot: np.ndarray, eta: int = config.DEFAULT_ETA) -> np.ndarray:
    """
    Applies spatial pilot masking across BS antennas according to reduction ratio eta.
    eta in {100, 75, 50, 25}.
    """
    masked = CSI_pilot.copy()
    if eta == 100:
        return masked
    elif eta == 75:
        masked[:, 3::4] = 0
    elif eta == 50:
        masked[:, 1::2] = 0
    elif eta == 25:
        masked[:, 1::2] = 0
        masked[:, 2::4] = 0
        masked[:, 3::4] = 0
    else:
        raise ValueError(f"Invalid pilot ratio eta={eta}. Must be in [100, 75, 50, 25].")
    return masked


def prepare_pilot_features(CSI_fading: np.ndarray,
                           snr: float = config.DEFAULT_SNR,
                           eta: int = config.DEFAULT_ETA,
                           Nc: int = config.NC,
                           Ns: int = config.NS,
                           Nt: int = config.NT,
                           return_raw_pilots: bool = False):
    """
    Simulates 3GPP pilot measurement across OFDM resource blocks, applies pilot masking,
    normalizes by Frobenius norm, and formats real/imaginary channels for ReducedPilotNet.

    Input:
        CSI_fading: (N, Nc, 1, Nt) fading channel matrix.
        snr: Signal-to-noise ratio in dB.
        eta: Pilot reduction percentage (e.g. 50).
        Nc, Ns, Nt: Subcarriers, symbols per slot, transmit antennas.
        return_raw_pilots: If True, also returns the (N, Nt) masked pilot array.

    Returns:
        pilot_comp: (N, Nc, Nt, 2) float32 tensor for ReducedPilotNet input.
        (Optional) CSI_pilot: (N, Nt) complex masked pilot measurements if return_raw_pilots=True.
    """
    # Resource block repeated across Ns OFDM symbols: (N, Nc, Ns, Nt)
    CSI_RB = np.repeat(CSI_fading, Ns, axis=2)
    CSI_noise_RB = add_noise(CSI_RB, snr)

    # Active pilot subcarrier and symbol indexing (3GPP standard layout)
    Nc_pos = np.array([2, 3, 4, 5, 8, 9, 10, 11])
    ant_idx = np.arange(Nt)
    j = ant_idx // 4
    k = ant_idx % 4

    sub_idx = Nc_pos[j]
    sym_idx = k + 3

    CSI_pilot = CSI_noise_RB[:, sub_idx, sym_idx, ant_idx]
    CSI_pilot = mask_pilots(CSI_pilot, eta)

    # Frobenius normalization per user
    norm = np.linalg.norm(CSI_pilot, axis=-1, keepdims=True)
    norm = np.where(norm == 0, 1e-12, norm)
    reduced_pilot = (CSI_pilot / norm)[:, None, :].repeat(Nc, axis=1)

    pilot_comp = np.concatenate([
        np.real(reduced_pilot[:, :, :, None]),
        np.imag(reduced_pilot[:, :, :, None]),
    ], axis=-1).astype(np.float32)

    if return_raw_pilots:
        return pilot_comp, CSI_pilot
    return pilot_comp


def prepare_integration_features(model_rm: torch.nn.Module,
                                 model_rp: torch.nn.Module,
                                 x_loc: np.ndarray,
                                 x_pilot: np.ndarray,
                                 device: str = config.DEVICE) -> np.ndarray:
    """
    Runs inference on pretrained Radio Map and Reduced Pilots models, normalizes
    their beamforming vectors, and concatenates real and imaginary channels
    to form the 4-channel input for IntegrationNet.

    Input:
        model_rm: Pretrained RadioMapNet.
        model_rp: Pretrained ReducedPilotNet.
        x_loc:    (N, 2) normalized GPS coordinates.
        x_pilot:  (N, Nc, Nt, 2) pilot measurement features.
        device:   Inference device ('cuda', 'mps', or 'cpu').

    Returns:
        x_int: (N, Nc, Nt, 4) float32 feature array for IntegrationNet.
    """
    model_rm.eval()
    model_rp.eval()

    with torch.no_grad():
        t_loc = torch.as_tensor(x_loc, dtype=torch.float32).to(device)
        t_pilot = torch.as_tensor(x_pilot, dtype=torch.float32).to(device)

        v_rm = normalise_V(model_rm(t_loc).cpu().numpy())
        v_rp = normalise_V(model_rp(t_pilot).cpu().numpy())

    # v_rp and v_rm have shape (N, Nc, Nt, 1). Concatenate along last dimension:
    x_int = np.concatenate([
        np.real(v_rp),
        np.imag(v_rp),
        np.real(v_rm),
        np.imag(v_rm),
    ], axis=-1).astype(np.float32)

    return x_int

def generate_svm_labels(rm_se, int_net_se):
    """
    Computes ground-truth binary labels for SVM:
        y = 1 if int_net_se > rm_se (worth sending pilots)
        y = 0 otherwise (RM wins; save pilot overhead)
    """
    labels = (int_net_se > rm_se).astype(int)
    best_se = np.maximum(rm_se, int_net_se)
    return labels, best_se


# =====================================================================
# Model Checkpoint & Weight Management Helpers
# =====================================================================
import json

def get_model_paths(model_name: str, dataset: str = "sydney", eta: int = None):
    """
    Resolves checkpoint .pth and history .json paths.
    Supports legacy filenames (e.g. RM.pth for RM_sydney.pth, RP_50.pth for RP_sydney_50.pth).
    """
    config.SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    tag = f"{model_name}_{dataset}"
    if eta is not None:
        tag += f"_{int(eta)}"

    save_path = config.SAVED_MODELS_DIR / f"{tag}.pth"
    if not save_path.exists() and dataset == "sydney":
        legacy_tag = f"{model_name}" if eta is None else f"{model_name}_{int(eta)}"
        legacy_path = config.SAVED_MODELS_DIR / f"{legacy_tag}.pth"
        if legacy_path.exists():
            save_path = legacy_path

    history_path = config.OUTPUTS_DIR / f"{tag}_history.json"
    if not history_path.exists() and dataset == "sydney":
        legacy_tag = f"{model_name}" if eta is None else f"{model_name}_{int(eta)}"
        legacy_hist = config.OUTPUTS_DIR / f"{legacy_tag}_history.json"
        if legacy_hist.exists():
            history_path = legacy_hist

    return save_path, history_path


def load_weights_if_available(model: torch.nn.Module,
                              save_path: Path,
                              history_path: Path = None,
                              force_train: bool = False,
                              device: str = config.DEVICE) -> bool:
    """
    Checks if pre-trained weights exist. If available and not force_train:
      - Loads weights into model with map_location=device.
      - Reads and prints best validation SE from history if available.
      - Returns True (weights loaded, skip training).
    Otherwise:
      - Returns False (proceed with training).
    """
    model.to(device)
    if save_path.exists() and not force_train:
        rel_save = save_path.relative_to(config.BASE_DIR) if save_path.is_relative_to(config.BASE_DIR) else save_path
        print(f"Found pre-trained weights at: {rel_save} (use --force-train to retrain)...")
        model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
        if history_path and history_path.exists() and history_path.stat().st_size > 0:
            try:
                with open(history_path, "r") as f:
                    hist = json.load(f)
                    if "val_loss" in hist and len(hist["val_loss"]) > 0:
                        best_val = min(hist["val_loss"])
                        print(f"Loaded model best validation SE: {-best_val:.3f} bps/Hz")
            except Exception:
                pass
        return True
    return False


def load_pretrained_model(model_class,
                          model_name: str,
                          dataset: str = "sydney",
                          eta: int = None,
                          device: str = config.DEVICE,
                          Nc: int = config.NC,
                          Nt: int = config.NT) -> torch.nn.Module:
    """
    Loads and returns an evaluation-ready pretrained model instance.
    Raises FileNotFoundError if weights do not exist.
    """
    save_path, _ = get_model_paths(model_name, dataset, eta)
    if not save_path.exists():
        raise FileNotFoundError(
            f"Pretrained weights for {model_name} ({dataset}) not found at {save_path}. "
            f"Please run scripts/train_{model_name.lower()}.py first!"
        )
    model = model_class(Nc, Nt).to(device)
    model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
    model.eval()
    return model
