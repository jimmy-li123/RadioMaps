"""
data/import_sim.py

Helper script to import, standardise, and validate external simulation datasets (e.g. DeepMIMO).
Takes an external directory of .npy files or DeepMIMO export and formats it into data/<name>/
with guaranteed MIMO parameter alignment (Nt=32, Nc=12, Nr=1).

Usage:
    # Import from a directory of arrays:
    python data/import_sim.py --name o1 --source ../reduced-pilots/data/FR1/

    # Verify existing simulation:
    python utils/verify_sim.py --dataset o1
"""

import sys
import argparse
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from utils.verify_sim import verify_dataset

def import_simulation(name, source_path):
    src = Path(source_path).resolve()
    dest = config.DATA_DIR / name
    dest.mkdir(parents=True, exist_ok=True)

    print(f"Importing simulation '{name}' from: {src}")
    print(f"Target directory: {dest}")

    if not src.exists():
        raise FileNotFoundError(f"Source path does not exist: {src}")

    # Load arrays from source
    ueloc_path = src / "UEloc.npy"
    if not ueloc_path.exists():
        ueloc_path = src / "loc.npy"

    csi_path = src / "CSI.npy"
    if not csi_path.exists():
        csi_path = src / "H.npy"

    bsloc_path = src / "BSloc.npy"
    los_path   = src / "LoS.npy"
    aod_path   = src / "AoD.npy"

    if not ueloc_path.exists() or not csi_path.exists():
        raise FileNotFoundError(f"Source directory must contain at least UEloc.npy (or loc.npy) and CSI.npy (or H.npy)")

    UEloc = np.load(ueloc_path)
    CSI   = np.load(csi_path)

    # Standardize CSI shape to (N, Nc, 1, Nt)
    # DeepMIMO standard format can be (N, 1, Nt, Nc) or (N, Nc, 1, Nt)
    if CSI.ndim == 4:
        if CSI.shape[1] == 1 and CSI.shape[2] == config.NT and CSI.shape[3] == config.NC:
            # (N, 1, Nt, Nc) -> (N, Nc, 1, Nt)
            CSI = np.transpose(CSI, (0, 3, 1, 2))
        elif CSI.shape[1] == config.NC and CSI.shape[2] == 1 and CSI.shape[3] == config.NT:
            pass
        elif CSI.shape[1] == config.NT and CSI.shape[3] == config.NC:
            # (N, Nt, 1, Nc) -> (N, Nc, 1, Nt)
            CSI = np.transpose(CSI, (0, 3, 2, 1))

    # Standardize UE coordinates to 2D (x, y)
    if UEloc.shape[-1] > 2:
        UEloc = UEloc[:, :2]

    # Handle BS location
    if bsloc_path.exists():
        BSloc = np.load(bsloc_path)
        if BSloc.ndim == 1:
            BSloc = BSloc[None, :]
        if BSloc.shape[-1] > 2:
            BSloc = BSloc[:, :2]
    else:
        # Default single BS at origin
        BSloc = np.array([[0.0, 0.0]], dtype=np.float32)

    # Handle LoS labels
    if los_path.exists():
        LoS = np.load(los_path).squeeze()
    else:
        # Default all users LoS
        LoS = np.ones(UEloc.shape[0], dtype=np.int64)

    # Handle AoD
    if aod_path.exists():
        AoD = np.load(aod_path).squeeze()
    else:
        # Compute geometric AoD from user and BS coordinates
        AoD = np.arctan2(UEloc[:, 1] - BSloc[0, 1], UEloc[:, 0] - BSloc[0, 0])

    # Filter out blocked users (LoS == -1 or zero CSI power)
    valid_mask = (LoS != -1)
    power = np.abs(np.sum(CSI.reshape(CSI.shape[0], -1), axis=1))
    valid_power_mask = power > 1e-15
    mask = valid_mask & valid_power_mask

    if not np.all(mask):
        n_filtered = UEloc.shape[0] - np.sum(mask)
        print(f"Filtering {n_filtered} blocked / zero-power users...")
        UEloc = UEloc[mask]
        CSI   = CSI[mask]
        LoS   = LoS[mask]
        AoD   = AoD[mask]

    # Save to destination directory
    np.save(dest / "UEloc.npy", UEloc.astype(np.float32))
    np.save(dest / "BSloc.npy", BSloc.astype(np.float32))
    np.save(dest / "CSI.npy", CSI.astype(np.complex64))
    np.save(dest / "LoS.npy", LoS.astype(np.int64))
    np.save(dest / "AoD.npy", AoD.astype(np.float32))

    print(f"\n[SUCCESS] Successfully imported '{name}' ({UEloc.shape[0]} users) into {dest}")
    verify_dataset(name)

def main():
    parser = argparse.ArgumentParser(description="Import and standardize simulation dataset")
    parser.add_argument("--name", type=str, required=True, help="Name of the simulation dataset (e.g. o1, o2, tokyo)")
    parser.add_argument("--source", type=str, required=True, help="Path to directory containing source .npy files")
    args = parser.parse_args()

    import_simulation(args.name, args.source)

if __name__ == "__main__":
    main()
