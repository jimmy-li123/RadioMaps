# Dataset Directory Guide

This directory manages the MIMO channel state matrices, positioning vectors, and propagation labels used for training and evaluating beamforming models.

---

## Directory Structure

```
data/
├── sydney/                  # DeepMIMO v4 Sydney CBD scenario (city_32_sydney_3p5)
│   ├── CSI.npy              # Complex channel tensor (27805, 12, 1, 32)
│   ├── UEloc.npy            # User equipment 2D (x, y) coordinates (27805, 2)
│   ├── BSloc.npy            # Base station 2D (x, y) coordinates (1, 2)
│   ├── LoS.npy              # Line-of-sight binary labels: 1=LoS, 0=NLoS (27805,)
│   └── AoD.npy              # Dominant path Angle of Departure (27805,)
│
├── o1/                      # DeepMIMO v1 O1 urban scenario (FR1 outdoor)
│   ├── CSI.npy              # Complex channel tensor (1954, 12, 1, 32)
│   ├── UEloc.npy            # User equipment 2D (x, y) coordinates (1954, 2)
│   ├── BSloc.npy            # Base station 2D (x, y) coordinates (1, 2)
│   ├── LoS.npy              # Line-of-sight binary labels: 1=LoS, 0=NLoS (1954,)
│   └── AoD.npy              # Dominant path Angle of Departure (1954,)
│
├── raw/                     # Raw DeepMIMO ray-tracing outputs (.mat & .json)
│   ├── params.json          # Simulation metadata
│   ├── objects.json         # Building / terrain mesh objects
│   └── *.mat                # Ray path power, phase, delay, AoD, positions
│
├── dataset.py               # Preprocessing script: converts data/raw/ -> data/sydney/
├── import_sim.py            # Universal importer: formats any external DeepMIMO sim -> data/<name>/
└── README.md                # This documentation
```

---

## Standard Array Signatures

All simulation datasets adhere to a unified array signature across all models:

| Array | Shape | Type | Description |
| :--- | :--- | :--- | :--- |
| `UEloc.npy` | `(N, 2)` | `float32` | User coordinates $(x, y)$ in meters |
| `BSloc.npy` | `(1, 2)` | `float32` | Base Station coordinates $(x, y)$ in meters |
| `CSI.npy` | `(N, 12, 1, 32)` | `complex64` | Multi-carrier channel matrix $(N, N_c, N_r, N_t)$ |
| `LoS.npy` | `(N,)` | `int64` | Line-of-sight state ($1 = \text{LoS}, 0 = \text{NLoS}$) |
| `AoD.npy` | `(N,)` | `float32` | Dominant Angle of Departure (degrees or radians) |

---

## How to Set Up Datasets

### 1. Generating the Sydney Dataset
1. Download the raw ray-tracing scenario `city_32_sydney_3p5` from [DeepMIMO v4](https://deepmimo.net/scenarios/v4/city_32_sydney_3p5).
2. Place the extracted `.mat` and `.json` files into `data/raw/`.
3. Run the processing script:
   ```bash
   python data/dataset.py
   ```
   This synthesizes the OFDM channel matrices and populates `data/sydney/`.

### 2. Importing Any External Simulation (e.g. O1, O2, Tokyo)
Use the automated import script [`data/import_sim.py`](file:///Users/jamesli/Desktop/UNSW/Year%203/Taste%20of%20Research/repos/sydney_test/data/import_sim.py):
```bash
python data/import_sim.py --name <sim_name> --source /path/to/source/folder
```
This automatically:
- Transposes channels to standard `(N, 12, 1, 32)`.
- Slices positions to 2D `(x, y)`.
- Filters blocked users (`LoS == -1` or zero CSI power).
- Validates the resulting dataset.

### 3. Verifying Dataset Integrity
Run the verification tool at any time:
```bash
python utils/verify_sim.py --dataset sydney
python utils/verify_sim.py --dataset o1
```
