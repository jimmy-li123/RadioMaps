# Model Definition
import torch
import torch.nn as nn

class RadioMapNet(nn.Module):
    def __init__(self, Nc=12, Nt=32):
        super().__init__()
        self.Nc = Nc
        self.Nt = Nt

        self.fc1 = nn.Linear(2, 16)
        self.fc2 = nn.Linear(16, 128)
        self.fc3 = nn.Linear(128, Nc * Nt * 2)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.tanh(self.fc3(x))

        # Reshape (N, Nc * Nt * 2) -> (N, Nc, Nt, 2)
        x = x.view(-1, self.Nc, self.Nt, 2)

        real = x[..., 0]
        imag = x[..., 1]

        output = torch.complex(real, imag)
        
        return output.unsqueeze(-1)     # Add extra dim - (N, Nc, Nt, 1)
    
        
        
