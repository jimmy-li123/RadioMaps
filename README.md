# Radio Map Construction for Telecommunications

A machine learning framework to reconstruct high-resolution radio environment maps from sparse spatial measurements in sub-6GHz massive MIMO networks.

## Overview
Current telecommunication protocols require large, bandwidth-expensive pilot signals to synthesize the Channel State Information (CSI) between the base station and user equipment. As we advance toward 6G infrastructure, environmental spatial awareness via **Radio Maps (RM)** offers a path to near-instantaneous beamforming using user coordinates. 

This project builds upon Yang et al. (2025), *"Radio Map-Based Beamforming Assisted With Reduced Pilots"*, reproducing and evaluating hybrid Radio Map and Reduced Pilot beamforming architectures using PyTorch across both the standard DeepMIMO **O1** scenario and a real-world 3D ray-traced scenario of **Sydney CBD** ([`city_32_sydney_3p5`](https://deepmimo.net/scenarios/v4/city_32_sydney_3p5)).

---

## System & MIMO Specifications

| Parameter | Symbol | Value | Notes |
| :--- | :--- | :--- | :--- |
| Carrier Frequency | $f_c$ | 3.5 GHz | Mid-band 5G / sub-6G |
| Bandwidth | $B$ | 10 MHz | 1 Resource Block (RB) allocation |
| Transmit Antennas (BS) | $N_t$ | 32 | Uniform Linear Array (ULA), $d = \lambda / 2$ |
| Receive Antennas (UE) | $N_r$ | 1 | Single antenna mobile equipment |
| OFDM Subcarriers | $N_c$ | 12 | Standard 12 subcarriers per RB |
| OFDM Symbols per Slot | $N_s$ | 14 | Standard 3GPP slot duration |
| Pilot Overhead Ratio | $\rho$ | 4.76% | At $\eta = 50\%$ pilot density (16 pilot REs / 168 total REs) |
| Default SNR | $\text{SNR}$ | 15.0 dB | Sweep: 0 to 30 dB |
| GPS Positioning Error | $\sigma_{\text{GPS}}$ | 1.0 m | Gaussian noise sweep: 0 to 5 m |
| Rayleigh Scattering Ratio | $K$ | 0.1 | Fast fading perturbation sweep: 0.0 to 0.2 |

---

## Project Structure
```
RadioMaps/
    ├── README.md                  # Project overview, workflow, and execution instructions
    ├── requirements.txt           # Python dependencies (PyTorch, NumPy, SciPy, scikit-learn)
    ├── config.py                  # Central configuration (MIMO parameters, paths, training hyperparameters)
    ├── train.py                   # Master multi-model training orchestrator
    ├── evaluate.py                # Master benchmarking & evaluation suite
    │
    ├── scripts/                   # Model training and experiment execution scripts
    │   ├── train_radio_map.py     # Radio Map MLP training & evaluation script
    │   ├── train_reduced_pilot.py # Reduced Pilot ResNet training & evaluation script
    │   └── train_integrate.py     # Integration fusion CNN training script
    │
    ├── data/                      # Multi-simulation dataset management
    │   ├── README.md              # Dataset format specifications and import instructions
    │   ├── dataset.py             # Generates Sydney arrays from raw DeepMIMO ray-tracing
    │   ├── import_sim.py          # Universal simulation importer (DeepMIMO, O1, etc.)
    │   ├── sydney/                # Sydney CBD dataset (27,805 users, untracked)
    │   ├── o1/                    # DeepMIMO O1 dataset (1,954 users, untracked)
    │   └── raw/                   # Raw ray-tracing files (.mat / .json, untracked)
    │
    ├── models/                    # Model architecture definitions (PyTorch)
    │   ├── radio_map.py           # Radio Map MLP ((x, y) -> beamforming vector V_RM)
    │   ├── reduced_pilot.py       # ResNet model (sparse CSI -> beamforming vector V_RP)
    │   ├── integrate.py           # Integration CNN (fuses V_RM and V_RP into V_INT)
    │   ├── svm_selector.py        # Dynamic SVM discriminator
    │   └── baseline/              # Algorithmic benchmarks
    │       ├── GBM.py             # Geometry-Based Method (trigonometric steering)
    │       └── CKM.py             # Channel Knowledge Map (k-NN + Inverse Distance Weighting)
    │
    ├── utils/                     # Mathematical & evaluation utilities
    │   ├── metrics.py             # SE calculations, normalisation, channel perturbations, loss, CLI parser
    │   └── verify_sim.py          # Dataset sanity and compatibility verification tool
    │
    ├── saved_models/              # Trained PyTorch model checkpoints (.pth)
    └── outputs/                   # Training curves, history logs, and evaluation metrics
```

---

## Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/sydney_test.git
   cd sydney_test
   ```

2. **Create a virtual environment and install dependencies:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Set up datasets:**
   - See [`data/README.md`](file:///Users/jamesli/Desktop/UNSW/Year%203/Taste%20of%20Research/repos/sydney_test/data/README.md) for full instructions.
   - For **Sydney**: place raw scenario files in `data/raw/` and run `python data/dataset.py`.
   - For **O1** (or any extra simulation): run `python data/import_sim.py --name <name> --source <path>`.
   - Verify compatibility:
     ```bash
     python utils/verify_sim.py --dataset sydney
     python utils/verify_sim.py --dataset o1
     ```

---

## Workflow & Execution

All models support standard command-line flags (`--dataset`, `--force-train`, `--epochs`, `--batch_size`, `--lr`, `--snr`, `--eta`).

### 1. Algorithmic Baselines
Evaluate the geometric baseline (GBM) or spatial database baseline (CKM):
```bash
# Geometry-Based Method
python models/baseline/GBM.py --dataset sydney
python models/baseline/GBM.py --dataset o1

# Channel Knowledge Map
python models/baseline/CKM.py --dataset sydney
python models/baseline/CKM.py --dataset o1
```

### 2. Radio Map Network
Trains a 3-layer MLP mapping user coordinates $(x, y) \to \mathbf{V}_{\text{RM}} \in \mathbb{C}^{N_c \times N_t \times 1}$ using unsupervised spectral efficiency loss:
```bash
# Train or evaluate on Sydney
python scripts/train_radio_map.py --dataset sydney

# Train on O1 with custom epochs
python scripts/train_radio_map.py --dataset o1 --epochs 1000 --force-train
```

### 3. Reduced Pilot ResNet
Reconstructs the full beamforming vector $\mathbf{V}_{\text{RP}}$ from sparse pilot measurements ($\eta \in \{25\%, 50\%, 75\%, 100\%\}$):
```bash
# Run with default 50% pilot density
python scripts/train_reduced_pilot.py --dataset sydney --eta 50

# Run on O1
python scripts/train_reduced_pilot.py --dataset o1 --eta 50
```

---

## Benchmark Results

Evaluated at $\text{SNR} = 15\text{ dB}$, $\sigma_{\text{GPS}} = 1.0\text{ m}$, $K = 0.1$:

| Model / Architecture | Metric | Sydney CBD (`city_32_sydney_3p5`) | DeepMIMO O1 (`FR1 outdoor`) |
| :--- | :--- | :--- | :--- |
| **GBM Baseline** | Ratio to Optimal ($\%$) | **91.61%** (LoS: 95.5%, NLoS: 31.7%) | **62.27%** (LoS: 90.5%, NLoS: 21.4%) |
| **CKM Baseline** | Ratio to Optimal ($\%$) | **94.03%** (LoS: 94.5%, NLoS: 86.6%) | **81.62%** (LoS: 86.4%, NLoS: 74.6%) |
| **Radio Map (RM)** | Ratio to Optimal ($\%$) | **92.06%** (LoS: 93.1%, NLoS: 76.2%) | **83.15%** (LoS: 86.8%, NLoS: 79.4%) |
| **Reduced Pilot (RP, $\eta=50\%$)** | Net Ratio to Optimal ($\%$) | **61.91%** (LoS: 64.5%, NLoS: 21.4%) | **78.17%** (LoS: 83.9%, NLoS: 69.9%) |

> [!NOTE]
> - In Sydney CBD, rich multipath scattering causes higher spatial aliasing for reduced antenna sampling ($\eta=50\%$), making Radio Maps and CKM superior for NLoS paths.
> - Reduced Pilots measures real-time fading directly, excelling when LoS is dominant and spatial correlation is high.

