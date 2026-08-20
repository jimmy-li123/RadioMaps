"""
models/integrate.py

Purpose:
Implements the fusion CNN that concatenates the predictions of the Radio Map model and Reduced Pilot model to output an integrated beamforming vector V_I.

Steps for Implementation:
- Stack real and imaginary channels of V_RP and V_RM into a 4-channel input tensor of shape (Nc, Nt, 4).
- Construct a 2D CNN (Conv2d(4, 16) -> Conv2d(16, 32) -> Conv2d(32, 2) -> Tanh) to extract complementary spatial features.
- Reconstruct the complex beamforming tensor (Nc, Nt, 1) and normalize to unit transmit power.
- Wrap model into a PyTorch nn.Module class.
"""
