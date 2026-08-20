"""
models/svm_selector.py

Purpose:
Implements an SVM classifier that predicts per user whether to use the zero-overhead Radio Map or the Reduced Pilot Integrated scheme based on net spectral efficiency after pilot penalties.

Steps for Implementation:
- Compute net SE for both schemes, applying the pilot overhead penalty factor (1 - N_pilot / N_total) to the integrated scheme.
- Generate binary classification labels (0 = Radio Map, 1 = Integrated).
- Train an RBF/Linear SVM on normalized user coordinates to act as a runtime discriminator.
- Export trained SVM using joblib / pickle to saved_models/.
"""
