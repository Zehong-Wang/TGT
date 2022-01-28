import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from module import updater, propagator, transformer
from unit import time_embedding

class TGT(nn.Module):
    def __init__(self, neighbor_finder, d_node_memory,
                 d_node_embedding, d_time_embedding,
                 n_layers, n_head, d_k, d_v, d_inner,
                 edge_features, device, aggregator,
                 src_time_shift_mean=0, src_time_shift_std=1,
                 dest_time_shift_mean=0, dest_time_shift_std=1):
        super(TGT, self).__init__()
        self.d_edge_features = edge_features.shape[1]
        self.updater = updater.Updater(d_mem=d_node_memory, d_edge=self.d_edge_features, d_time=d_time_embedding)
        self.propagator = propagator.Propagator(d_emb=d_node_embedding, d_mem=d_node_memory, d_edge=self.d_edge_features, d_time=d_time_embedding)
        self.embedder = transformer.Transformer(n_layers=n_layers,
                                                n_head=n_head,
                                                d_k=d_k,
                                                d_v=d_v,
                                                d_node_embedding=d_node_embedding,
                                                d_inner=d_inner,
                                                d_node_memory=d_node_memory,
                                                d_edge_features=self.d_edge_features,
                                                aggregator=aggregator)
        self.time_encoder = time_embedding.Time_embedding(d_time_embedding)
        self.memory = torch.zeros([30000, d_node_memory], dtype=torch.float, device=device)
        self.embedding = torch.zeros([30000, d_node_embedding], dtype=torch.float, device=device)
        self.last_update = torch.zeros([30000], dtype=torch.float, device=device)
        self.src_time_shift_mean = torch.tensor(src_time_shift_mean, dtype=torch.float, device=device)
        self.src_time_shift_std = torch.tensor(src_time_shift_std, dtype=torch.float, device=device)
        self.dest_time_shift_mean = torch.tensor(dest_time_shift_mean, dtype=torch.float, device=device)
        self.dest_time_shift_std = torch.tensor(dest_time_shift_std, dtype=torch.float, device=device)
        self.edge_features = torch.tensor(edge_features, dtype=torch.float, device=device)
        self.neighbor_finder = neighbor_finder
        self.d_node_memory = d_node_memory
        self.device = device
        self.mlp = nn.Sequential(
            nn.Linear(d_node_embedding * 2, 1),
            nn.Sigmoid()
        )

    def forward(self, src_node, dest_node, neg_node, edge_time, edge_src_dest_idx, n_neighbors=20, test=False):
        n_samples = len(src_node)
        edge_time = torch.tensor(edge_time, dtype=torch.float, device=self.device)

        src_node = torch.tensor(src_node, dtype=torch.long, device=self.device)
        dest_node = torch.tensor(dest_node, dtype=torch.long, device=self.device)
        neg_node = torch.tensor(neg_node, dtype=torch.long, device=self.device)

        nodes = torch.cat([src_node, dest_node, neg_node], dim=0)

        src_last_update= self.last_update[src_node]
        dest_last_update= self.last_update[dest_node]
        neg_last_update= self.last_update[neg_node]

        edge_features = self.edge_features[edge_src_dest_idx]

        # update memory
        src_node_memory = self.memory[src_node].clone()
        dest_node_memory = self.memory[dest_node].clone()
        neg_node_memory = self.memory[neg_node].clone()

        src_time_diff = (((edge_time - src_last_update) - self.src_time_shift_mean) / self.src_time_shift_std).unsqueeze(-1)
        dest_time_diff = (((edge_time - dest_last_update) - self.dest_time_shift_mean) / self.dest_time_shift_std).unsqueeze(-1)
        neg_time_diff = (((edge_time - neg_last_update) - self.dest_time_shift_mean) / self.dest_time_shift_std).unsqueeze(-1)

        src_time_diff = self.time_encoder(src_time_diff)
        dest_time_diff = self.time_encoder(dest_time_diff)
        neg_time_diff = self.time_encoder(neg_time_diff)

        msg_update_src_dest = torch.cat([src_node_memory, dest_node_memory, edge_features, src_time_diff], dim=1)
        msg_update_dest_src = torch.cat([dest_node_memory, src_node_memory, edge_features, dest_time_diff], dim=1)
        msg_update_src_neg = torch.cat([src_node_memory, neg_node_memory, edge_features, src_time_diff], dim=1)
        msg_update_neg_src = torch.cat([neg_node_memory, src_node_memory, edge_features, neg_time_diff], dim=1)

        updated_src_memory = self.updater(msg_update_src_dest, src_node_memory)
        updated_dest_memory = self.updater(msg_update_dest_src, dest_node_memory)
        updated_src_memory = self.updater(msg_update_src_neg, updated_src_memory)
        updated_neg_memory = self.updater(msg_update_neg_src, neg_node_memory)
        self.memory[src_node] = updated_src_memory
        self.memory[dest_node] = updated_dest_memory
        self.memory[neg_node] = updated_neg_memory

        # calculate embedding
        # neighbors_idx, neighbors_time = self.neighbor_finder()
        timestamps = torch.cat([edge_time, edge_time, edge_time], dim=0)
        neighbors_idx, edge_idx, neighbors_time = self.neighbor_finder.get_temporal_neighbor(
            source_nodes=nodes.cpu().numpy(),
            timestamps=timestamps.cpu().numpy(),
            n_neighbors=n_neighbors
        )
        src_neighbors_idx = neighbors_idx[:n_samples].flatten()
        dest_neighbors_idx = neighbors_idx[n_samples:2*n_samples].flatten()
        neg_neighbors_idx = neighbors_idx[2*n_samples:].flatten()

        src_memory = self.memory[nodes]

        n_d0, n_d1 = neighbors_idx.shape[0], neighbors_idx.shape[1]
        neighbors_memory = self.memory[neighbors_idx.flatten()].view(n_d0, n_d1, self.d_node_memory)
        neighbors_edge_features = self.edge_features[edge_idx.flatten()].view(n_d0, n_d1, self.d_edge_features)
        mask = neighbors_idx == 0
        neighbors_time = torch.tensor(neighbors_time, dtype=torch.float, device=self.device)
        embedding = self.embedder(src_node_memory=src_memory,
                                  src_node_time=timestamps,
                                  neighbors_memory=neighbors_memory,
                                  edge_time=neighbors_time,
                                  edge_features=neighbors_edge_features,
                                  neighbors_padding_mask=mask)

        src_embedding = embedding[:n_samples]
        dest_embedding = embedding[n_samples: 2 * n_samples]
        neg_embedding = embedding[2 * n_samples:]
        self.embedding[src_node] = src_embedding
        self.embedding[dest_node] = dest_embedding
        self.embedding[neg_node] = neg_embedding

        # propagate embedding to influenced nodes
        neighbors_time_diff = self.time_encoder(((timestamps.unsqueeze(-1) - neighbors_time - self.dest_time_shift_mean) / self.dest_time_shift_std).unsqueeze(-1))

        msg_prop = torch.cat([embedding.unsqueeze(1).repeat(1, 20, 1), neighbors_memory, neighbors_edge_features, neighbors_time_diff], dim=2)
        d_msg_prop = msg_prop.shape[2]

        src_msg_prop, dest_msg_prop, neg_msg_prop = msg_prop[:n_samples], msg_prop[n_samples:2*n_samples], msg_prop[2*n_samples:]
        src_neighbors_memory, dest_neighbors_memory, neg_neighbors_memory = neighbors_memory[:n_samples], neighbors_memory[n_samples:2*n_samples], neighbors_memory[2*n_samples:]

        src_prop_memory = self.propagator(src_msg_prop.view(n_samples * n_neighbors, d_msg_prop), src_neighbors_memory.view(n_samples * n_neighbors, self.d_node_memory), is_src=True)
        self.memory[src_neighbors_idx] = src_prop_memory

        dest_prop_memory = self.propagator(dest_msg_prop.view(n_samples * n_neighbors, d_msg_prop), dest_neighbors_memory.view(n_samples * n_neighbors, self.d_node_memory), is_src=False)
        self.memory[dest_neighbors_idx] = dest_prop_memory

        neg_prop_memory = self.propagator(neg_msg_prop.view(n_samples * n_neighbors, d_msg_prop), neg_neighbors_memory.view(n_samples * n_neighbors, self.d_node_memory), is_src=False)
        self.memory[neg_neighbors_idx] = neg_prop_memory

        pos_embedding = torch.cat([src_embedding, dest_embedding], dim=1)
        neg_embedding = torch.cat([src_embedding, neg_embedding], dim=1)
        pos_prob = self.mlp(pos_embedding)
        neg_prob = self.mlp(neg_embedding)

        self.memory.detach_()
        self.edge_features.detach_()
        self.embedding.detach_()

        return pos_prob, neg_prob

    def set_neighbor_finder(self, neighbor_finder):
        self.neighbor_finder = neighbor_finder