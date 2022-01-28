
import torch
from torch import nn
from unit.transformer_layer import EncoderLayer
from unit.time_embedding import Time_embedding

class Transformer(nn.Module):

    def __init__(self, n_layers, n_head, d_k, d_v, d_node_embedding, d_inner, d_node_memory, d_edge_features, aggregator='mean', dropout=0.1):
        super(Transformer, self).__init__()

        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([EncoderLayer(d_msg=d_node_embedding + d_edge_features,
                                                       d_inner=d_inner,
                                                       n_head=n_head,
                                                       d_k=d_node_embedding + d_edge_features,
                                                       d_v=d_node_embedding + d_edge_features, dropout=dropout) for _ in range(n_layers)])
        self.d_node_embedding = d_node_embedding
        self.time_encoder = Time_embedding(d_node_memory + d_edge_features)
        self.aggregator = aggregator
        if aggregator == 'mean':
            self.project = nn.Sequential(
                nn.Linear(2 * (d_node_embedding + d_edge_features), d_node_embedding),
                nn.Sigmoid(),
                nn.LayerNorm(d_node_embedding)
            )
        elif aggregator == 'sum':
            self.project = nn.Sequential(
                nn.Linear(d_node_embedding + d_edge_features, d_node_embedding),
                nn.Sigmoid(),
                nn.LayerNorm(d_node_embedding)
            )
        elif aggregator == 'pool':
            self.project = nn.Sequential(
                nn.Linear(2 * (d_node_embedding + d_edge_features), d_node_embedding),
                nn.Sigmoid(),
                nn.LayerNorm(d_node_embedding)
            )
            self.pool = nn.Linear(d_node_embedding + d_edge_features, d_node_embedding + d_edge_features)

    def forward(self, src_node_memory, src_node_time, neighbors_memory, edge_time, edge_features, neighbors_padding_mask):

        src_node_memory_unrolled = src_node_memory.unsqueeze(1)

        src_edge_features = edge_features.mean(1).unsqueeze(1)
        src_node_memory_unrolled = torch.cat([src_node_memory_unrolled, src_edge_features], dim=2)
        src_node_time_embedding = self.time_encoder(src_node_time.unsqueeze(-1).unsqueeze(-1))
        src_node_memory_unrolled = src_node_memory_unrolled + src_node_time_embedding

        neighbors_memory = torch.cat([neighbors_memory, edge_features], dim=2)
        edge_time_embedding = self.time_encoder(edge_time.unsqueeze(-1))
        neighbors_memory = neighbors_memory + edge_time_embedding

        node_memory = torch.cat([src_node_memory_unrolled, neighbors_memory], dim=1).permute([1, 0, 2])

        enc_output = node_memory

        neighbors_padding_mask = torch.tensor(neighbors_padding_mask, dtype=torch.bool)
        invalid_neighborhood_mask = neighbors_padding_mask.all(dim=1, keepdim=True)
        neighbors_padding_mask[invalid_neighborhood_mask.squeeze(), 0] = False
        neighbors_padding_mask_filling = torch.tensor(
            [False for _ in range(len(neighbors_padding_mask))],
            dtype=torch.bool,
            device=neighbors_padding_mask.device
        ).view(-1, 1)
        neighbors_padding_mask = torch.cat([neighbors_padding_mask_filling, neighbors_padding_mask], dim=1).type(torch.BoolTensor).to(enc_output.device)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, enc_output, enc_output, slf_attn_mask=neighbors_padding_mask)

        enc_output = enc_output.permute([1, 0, 2])
        enc_output = enc_output.masked_fill(neighbors_padding_mask.unsqueeze(-1), value=0)

        # SUM
        if self.aggregator == 'sum':
            src_output = enc_output[:, 0] + enc_output[:, 1:].mean(1)
        elif self.aggregator == 'mean':
            src_output = torch.cat([enc_output[:, 0], enc_output[:, 1:].mean(1)], dim=1).squeeze()
        # elif self.aggregator == 'pool':
        #     for idx, item in enc_output:
        #         if idx == 0:
        #             max_tmp = self.pool(item)
        #         else:
        #             max_tmp = torch.max(max_tmp, self.pool(item))
        #     src_output = torch.cat([enc_output[:, 0], max_tmp], dim=1).squeeze()
        # # src_output = enc_output[:, :].sum(1)

        src_output = self.project(src_output)

        return src_output
