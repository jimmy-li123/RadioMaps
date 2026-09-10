# Model Definition for Reduced Pilots

import torch
import torch.nn as nn

# Residual skip connection
class ResNetBlock(nn.Module):
    def __init__(self): # Grid of 12x32 complex values
        super().__init__()

        # 2 input channels (real, im)
        self.conv1 = nn.Conv2d(2, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 2, 3, padding=1)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.tanh(self.conv3(x))

        return residual + x
    
class ReducedPilotNet(nn.Module):
    def __init__(self, Nc=12, Nt=32):
        super().__init__()
        
        self.Nc = Nc
        self.Nt = Nt
        
        self.res_net = ResNetBlock()
        self.conv_final = nn.Conv2d(2, 2, 3, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        # If not in (N, 2, Nc, Nt)
        if x.shape[-1] == 2:
            x = x.permute(0, 3, 1, 2)
        
        x = self.res_net(x)
        x = self.tanh(self.conv_final(x))
        
        # Convert back to (N, Nc, Nt, 2)
        x = x.permute(0, 2, 3, 1)

        real = x[...,0]
        imag = x[...,1]

        output = torch.complex(real, imag)

        return output.unsqueeze(-1)
        