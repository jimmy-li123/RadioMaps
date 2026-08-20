"""
data/dataset.py

Purpose:
Parses raw Remcom Wireless InSite ray-tracing .mat files from data/raw/ and synthesizes the frequency-domain MIMO channel matrix CSI.npy alongside user coordinates, base station positions, AoD, and LoS labels in data/FR1/.

Steps for Implementation:
- Load raw .mat arrays (power, phase, delay, AoD, rx_pos, tx_pos, inter).
- Filter active receivers where power is non-NaN and convert power from dBm to linear Watts.
- Extract 2D coordinates (UEloc, BSloc) and dominant path Angle-of-Departure (AoD).
- Classify LoS vs NLoS using interaction counts from inter_t001_tx000_r000.mat (0 interactions = direct LoS).
- Synthesize multi-carrier MIMO channel tensors CSI of shape (N, Nc, 1, Nt) using steering vectors and subcarrier delays.
- Normalize CSI by its maximum absolute value and save CSI.npy, UEloc.npy, BSloc.npy, AoD.npy, and LoS.npy.
"""
