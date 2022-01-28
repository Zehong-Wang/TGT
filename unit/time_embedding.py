import torch
from torch import nn
import numpy as np

class Time_embedding(nn.Module):
    def __init__(self, d_out):
        super(Time_embedding, self).__init__()
        self.project = nn.Linear(1, d_out)
        self.project.weight = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, d_out))).float().reshape(d_out, -1))
        self.project.bias = torch.nn.Parameter(torch.zeros(d_out).float())

    def forward(self, x):
        x = torch.sin(self.project(x))
        return x