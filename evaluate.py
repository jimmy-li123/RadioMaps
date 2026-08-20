"""
evaluate.py

Purpose:
Evaluates and compares all methods (Optimal SVD, GBM, CKM, Radio Map, Reduced Pilots, Integrated, and SVM Hybrid) on test data across various SNR, GPS error, and fading conditions.

Steps for Implementation:
- Load test split and trained models from saved_models/.
- Compute theoretical upper bound via Singular Value Decomposition (SVD) on perfect CSI.
- Run inference across all 7 methods and calculate average spectral efficiency (bps/Hz) and percentage of optimal (% opt).
- Separate performance metrics into overall, LoS, and NLoS user groups.
- Generate comparison plots / sweep curves against SNR, GPS error, and fading ratio, saving outputs to outputs/.
"""
