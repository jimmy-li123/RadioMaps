"""
models/radio_map.py

Purpose:
Defines the PyTorch MLP neural network that maps a user's 2D GPS coordinates (x, y) directly to a complete multi-carrier beamforming vector V_RM with zero pilot transmission.

Steps for Implementation:
- Build an MLP architecture: Linear(2, 16) -> ReLU -> Linear(16, 128) -> ReLU -> Linear(128, 2 * Nc * Nt) -> Tanh.
- Reshape output into real and imaginary components (Nc, Nt, 2) and form a complex beamforming vector (Nc, Nt, 1).
- Apply power normalization across antennas before computing loss or inference output.
- Wrap the model into a standard PyTorch nn.Module class.
"""
