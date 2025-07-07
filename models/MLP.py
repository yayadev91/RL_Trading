import torch
import torch.nn as nn
from config import Q_INPUT_DIM, Q_HIDDEN_DIMS, Q_OUTPUT_DIM, Q_DROPOUT

class MLPQNet(nn.Module):
    def __init__(self,
                 input_dim=Q_INPUT_DIM,
                 hidden_dims=Q_HIDDEN_DIMS,
                 output_dim=Q_OUTPUT_DIM,
                 dropout=Q_DROPOUT):
        super().__init__()
        layers = []
        last_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(last_dim, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            last_dim = h
        layers.append(nn.Linear(last_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
