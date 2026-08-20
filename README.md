# Radio Map Construction for Teleommunications

A machine learning framework to reconstruct high resolution radio environment maps from sparse spatial measurements in sub-6G networks

## Overview
Current telecomunication protocols require large, expensive pilot signals to generate the necessary beamforming vectors by synthesising the Channel State Information (CSI) between the base station and the user. As we approach future 6G infrastructure this technique can be replaced with Radio Maps, which take a 2D slice of spatial knowledge of radio propagation as a means for beamforming. This project builds on work of Bin Yang's 2025 Paper `Radio Map-Based Beamforming Assisted With Reduced Pilots`, and replicates results into a DeepMIMO construction of sydney.

## Project Structure
```
├── data/               # Raw channel measurements, ray-tracing datasets, and maps
├── notebooks/          # Exploratory data analysis and prototyping
├── src/
│   ├── models/         # Neural network architectures (e.g., CNNs, U-Net, GNNs)
│   ├── preprocessing/  # Normalisation, spatial sampling, and grid generation
│   └── utils/          # Plotting, metrics (NMSE, SSIM), and evaluation helpers
├── configs/            # Experiment parameters and model hyperparameters
├── tests/              # Unit tests
├── requirements.txt    # Python dependencies
└── README.md
```

## Setup

## Workflow

## Metrics
