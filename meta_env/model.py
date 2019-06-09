import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ConvModel(nn.Module):
    """
    The optimizee model to be trained.
    """
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 32, 3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 32, 3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            torch.nn.AdaptiveAvgPool2d(1)
        )
        self.fc_out = nn.Linear(32, 10)

    def forward(self, x):
        x = self.layers(x)
        x = x.squeeze()
        x = self.fc_out(x)
        return x