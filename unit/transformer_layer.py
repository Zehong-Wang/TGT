from torch import nn
from torch.nn import functional as F

class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)  # position-wise
        self.w_2 = nn.Linear(d_hid, d_in)  # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x

        x = self.w_1(x)
        x = F.relu(x)
        x = self.w_2(x)
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)

        return x

class EncoderLayer(nn.Module):

    def __init__(self, d_msg, d_inner, n_head, d_k, d_v, dropout=0.1, residual=True):
        super(EncoderLayer, self).__init__()
        self.slf_attn = nn.MultiheadAttention(embed_dim=d_msg, kdim=d_k, vdim=d_v, num_heads=n_head, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_v, d_inner, dropout=dropout)
        self.residual = residual
        self.layer_norm = nn.LayerNorm(d_v, eps=1e-6)

    def forward(self, q_input, k_input, v_input, slf_attn_mask=None):
        if self.residual:
            residual = v_input

        enc_output, enc_slf_attn = self.slf_attn(q_input, k_input, v_input, key_padding_mask=slf_attn_mask)
        if self.residual:
            enc_output += residual

        enc_output = self.layer_norm(enc_output)

        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn
