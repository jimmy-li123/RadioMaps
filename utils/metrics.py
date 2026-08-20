"""
utils/metrics.py

Purpose:
Provides core mathematical and channel impairment utilities, beamforming vector power normalization, spectral efficiency calculation, and the PyTorch unsupervised loss function.

Steps for Implementation:
- Implement channel perturbation functions: add_error (GPS noise), normalise_loc, add_fading (Rayleigh scattering), and add_noise (AWGN).
- Implement normalise_V to enforce the unit-power constraint ||V||^2 = 1 on beamforming vectors.
- Implement cal_SE for per-user and per-subcarrier Shannon capacity rate calculation: log2(1 + SNR).
- Implement compute to separate spectral efficiency results into LoS vs NLoS user groups.
- Implement optimal_svd_beamforming via SVD on perfect CSI as the theoretical upper-bound benchmark.
- Implement the PyTorch custom unsupervised loss su_loss that minimizes negative mean spectral efficiency: -E[log2(1 + |HV|^2 / sigma^2)].
"""