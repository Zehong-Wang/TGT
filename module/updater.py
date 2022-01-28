import torch
from torch import nn
from unit import rnn


class Updater(nn.Module):
    def __init__(self, d_mem, d_edge, d_time):
        super(Updater, self).__init__()
        d_msg = d_mem + d_mem + d_edge + d_time
        self.msg_project = nn.Sequential(
            nn.Linear(d_msg, d_msg),
            nn.Sigmoid(),
            nn.LayerNorm(d_msg)
        )
        self.updater = rnn.GRUCell(d_msg, d_mem)
        self.layernorm = nn.LayerNorm(d_mem)

    def forward(self, message, memory):
        message = self.msg_project(message)
        new_memory = self.updater(message, memory)
        new_memory = self.layernorm(new_memory)
        return new_memory
