import torch
from torch import nn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Net(nn.Module):
    def __init__(self, middle_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, middle_dim),
            nn.ReLU(),
            nn.Linear(middle_dim, middle_dim),
            nn.ReLU(),
            nn.Linear(middle_dim, 1),
        ).to(DEVICE)

    def forward(self, a, b):
        return self.net(torch.cat([a.to(DEVICE), b.to(DEVICE)], dim=1))
