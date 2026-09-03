# Radio Map Construction for Teleommunications

A machine learning framework to reconstruct high resolution radio environment maps from sparse spatial measurements in sub-6G networks

## Overview
Current telecomunication protocols require large, expensive pilot signals to generate the necessary beamforming vectors by synthesising the Channel State Information (CSI) between the base station and the user. As we approach future 6G infrastructure this technique can be replaced with Radio Maps, which take a 2D slice of spatial knowledge of radio propagation as a means for beamforming. This project builds on work of Yang's 2025 Paper `Radio Map-Based Beamforming Assisted With Reduced Pilots`, and replicates results with PyTorch into a Ray-Traced scenario of [city_32_sydney_3p5](https://deepmimo.net/scenarios/v4/city_32_sydney_3p5) in DeepMIMO v4

## Project Structure
```
RadioMaps/                                                                                                                            
    ├── README.md                  # Project overview, methodology, and execution instructions                                            
    ├── requirements.txt           # Python dependency specifications (PyTorch, NumPy, SciPy, etc.)                                       
    ├── config.py                  # Central configuration (MIMO parameters, paths, training hyperparameters)                             
    ├── train.py                   # End-to-end training pipeline orchestrator                                                            
    ├── train_radio_map.py         # Dedicated Radio Map training script                                                                  
    ├── evaluate.py                # Benchmarking, ablation studies, and spectral efficiency plotting suite                               
    │                                                                                                                                     
    ├── data/                      # Dataset management directory
    │   ├── dataset.py             # Parses raw ray-tracing data and synthesises CSI / AoD / LoS tensors
    │   ├── raw/                   # DeepMIMO simulation outputs (.mat & .json)
    │   │   ├── params.json        # Simulation metadata (3.5 GHz carrier, 10 MHz BW, isotropic antenna)
    │   │   ├── power_*.mat        # Received path powers (dBm)
    │   │   ├── phase_*.mat        # Carrier propagation and reflection phases (deg)
    │   │   ├── delay_*.mat        # Time-of-arrival propagation delays (s)
    │   │   ├── aod_az_*.mat       # Azimuth angle-of-departure (deg)
    │   │   ├── aod_el_*.mat       # Elevation angle-of-departure (deg)
    │   │   ├── rx_pos_*.mat       # 3D receiver coordinates (x, y, z)
    │   │   ├── tx_pos_*.mat       # 3D base station coordinates (x, y, z)
    │   │   └── inter_*.mat        # Path interaction counts (0 = direct LoS)
    │   └── processed/             # Preprocessed tensors ready for neural network training
    │       ├── CSI.npy            # Synthesised multi-carrier channel tensor (N, Nc, 1, Nt)
    │       ├── UEloc.npy          # 2D user coordinates (N, 2)
    │       ├── BSloc.npy          # 2D base station position (1, 2)
    │       ├── AoD.npy            # Dominant path Angle of Departure (N,)
    │       ├── LoS.npy            # Line-of-sight binary labels (N,)
    │       ├── SNR.npy            # SNR benchmark sweep values (0 to 30 dB)
    │       ├── loc_std.npy        # GPS error standard deviation sweep (0 to 5 m)
    │       └── fading_ratio.npy   # Rayleigh scattering power ratio sweep (0.0 to 0.2)
    │
    ├── models/                    # Neural network architecture definitions (PyTorch)
    │   ├── radio_map.py           # Radio Map MLP (maps GPS (x, y) -> beamforming vector V_RM)
    │   ├── reduced_pilot.py       # ResNet model (maps sparse pilot CSI -> beamforming vector V_RP)
    │   ├── integrate.py           # Integration CNN (fuses V_RM and V_RP into hybrid V_INT)
    │   └── svm_selector.py        # SVM classifier / discriminator (dynamic selection between RM & RP)
    │
    ├── utils/                     # Mathematical, channel impairment, and evaluation utilities
    │   ├── metrics.py             # SE calculations, normalisation, channel perturbations (Rayleigh/AWGN/GPS), loss
    │   └── debug.py               # Data inspection, tensor sanity checks, and debugging helpers
    │
    ├── saved_models/              # Serialised PyTorch model checkpoints (.pth) and SVM classifiers (.pkl)
    └── outputs/                   # Benchmark evaluation figures, CDFs, and performance logs
```

## Setup

## Workflow

## Metrics
