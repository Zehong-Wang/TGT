import torch
from torch import nn
from unit import rnn

class Propagator(nn.Module):
    def __init__(self, d_emb, d_mem, d_edge, d_time):
        super(Propagator, self).__init__()
        d_msg = d_emb + d_mem + d_edge + d_time
        self.prop_src = rnn.GRUCell(d_msg, d_mem)
        self.prop_dest = rnn.GRUCell(d_msg, d_mem)
        self.layernorm = nn.LayerNorm(d_mem)

    def forward(self, message, memory, is_src=True):
        if is_src:
            new_memory = self.prop_src(message, memory)
        else:
            new_memory = self.prop_dest(message, memory)

        new_memory = self.layernorm(new_memory)

        return new_memory
