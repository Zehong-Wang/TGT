import torch
from torch import nn

class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GRUCell, self).__init__()
        self.update_gate = nn.Sequential(
            nn.Linear((input_size + hidden_size), 1),
            nn.Sigmoid()
        )
        self.reset_gate = nn.Sequential(
            nn.Linear((input_size + hidden_size), 1),
            nn.Sigmoid()
        )
        self.memory_project = nn.Sequential(
            nn.Linear(hidden_size, hidden_size)
        )
        self.input_project = nn.Sequential(
            nn.Linear(input_size, hidden_size)
        )

    def forward(self, input, hidden):
        x = torch.cat([input, hidden], dim=1)
        update = self.update_gate(x)
        reset = self.reset_gate(x)
        hidden_project = self.memory_project(hidden)
        input = self.input_project(input)
        r = torch.tanh(reset * hidden_project + input)
        output = update * hidden + (1 - update) * r

        return output
