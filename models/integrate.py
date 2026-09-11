# Model definition for integration with Reduced Pilots and Radio Map
# 2D CNN to fuse both predictions into an integrated beamforming vector

import torch
import torch.nn as nn

class IntegrationNet(nn.Module):
    def __init__(self, Nc=12, Nt=32):
        super().__init__()
        
        self.Nc = Nc
        self.Nt = Nt
        
        self.conv1 = nn.Conv2d(4, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 2, 3, padding=1)
        
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        # If shape is (N, Nc, Nt, 4), permute to (N, 4, Nc, Nt)
        if x.shape[-1] == 4:
            x = x.permute(0, 3, 1, 2)

        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.tanh(self.conv3(x))
        
        x = x.permute(0, 2, 3, 1)
        
        real = x[..., 0]
        imag = x[..., 1]
        
        output = torch.complex(real, imag)
        
        return output.unsqueeze(-1)
