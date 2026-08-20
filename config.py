"""
config.py

Purpose:
Centralizes all system parameters (carrier frequency fc, bandwidth B, antenna counts Nt, Nr, subcarriers Nc), file paths, simulation defaults, and training hyperparameters.

Steps for Implementation:
- Define physical and array constants (fc = 3.5 GHz, B = 10 MHz, Nt = 32, Nr = 1, Nc = 12).
- Set default channel conditions (SNR = 15 dB, fading ratio K = 0.1, GPS error loc_std = 1.0 m, eta = 50%).
- Define project directory paths (RAW_DATA_DIR, PROCESSED_DATA_DIR, SAVED_MODELS_DIR, OUTPUTS_DIR).
- Define sweep parameter arrays for SNR, GPS error, and fading ratio benchmarks.
- Configure training hyperparameters (batch size, learning rate, epochs, device).
"""
