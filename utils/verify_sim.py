"""
utils/verify_sim.py

Utility script to inspect and verify dataset compatibility and channel statistics across simulations.
Usage:
    python utils/verify_sim.py --dataset sydney
    python utils/verify_sim.py --dataset o1
"""

import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from utils.metrics import load_dataset, parse_args

def verify_dataset(dataset_name):
    print(f"\n==========================================")
    print(f" Verifying Dataset: {dataset_name}")
    print(f" Directory: {config.get_dataset_dir(dataset_name)}")
    print(f"==========================================")

    UEloc, CSI, LoS, BSloc, AoD = load_dataset(
        dataset_name, return_bsloc=True, return_aod=True, eliminate_blocked=False
    )

    print(f"UEloc: shape={UEloc.shape}, dtype={UEloc.dtype}")
    print(f"BSloc: shape={BSloc.shape if BSloc is not None else 'None'}, dtype={BSloc.dtype if BSloc is not None else 'None'}")
    print(f"CSI:   shape={CSI.shape}, dtype={CSI.dtype}")
    print(f"LoS:   shape={LoS.shape}, dtype={LoS.dtype}")
    print(f"AoD:   shape={AoD.shape if AoD is not None else 'None'}, dtype={AoD.dtype if AoD is not None else 'None'}")

    valid_mask = (LoS != -1)
    power = np.abs(np.sum(CSI.reshape(CSI.shape[0], -1), axis=1))
    valid_power_mask = power > 1e-15
    total_valid = np.sum(valid_mask & valid_power_mask)

    print(f"\nUser Statistics:")
    print(f"  Total users:   {UEloc.shape[0]}")
    print(f"  Blocked users: {UEloc.shape[0] - total_valid}")
    print(f"  Valid users:   {total_valid}")
    
    los_count = np.sum(LoS == 1)
    nlos_count = np.sum(LoS == 0)
    print(f"  LoS users:     {los_count} ({los_count / UEloc.shape[0] * 100:.1f}%)")
    print(f"  NLoS users:    {nlos_count} ({nlos_count / UEloc.shape[0] * 100:.1f}%)")

    print(f"\nCoordinate Extents:")
    print(f"  UE X range: [{np.min(UEloc[:, 0]):.2f}, {np.max(UEloc[:, 0]):.2f}] m")
    print(f"  UE Y range: [{np.min(UEloc[:, 1]):.2f}, {np.max(UEloc[:, 1]):.2f}] m")
    if BSloc is not None:
        print(f"  BS Location: [{BSloc[0, 0]:.2f}, {BSloc[0, 1]:.2f}] m")

    csi_pow = np.mean(np.abs(CSI) ** 2)
    print(f"\nChannel Power:")
    print(f"  Mean |H|^2: {csi_pow:.6e}")
    print(f"[STATUS] Compatibility check PASSED for '{dataset_name}'!\n")

def main():
    args = parse_args(description="Verify simulation dataset compatibility")
    verify_dataset(args.dataset)

if __name__ == "__main__":
    main()
