import torch
from simpatico.utils.model_utils import ResBlock, PositionalEdgeGenerator, KVEncoder
from simpatico.models import ProteinEncoderDefaults
from torch_geometric.nn import GATv2Conv, Sequential
from torch_geometric.nn.models import MLP
from torch_geometric.nn.aggr import AttentionalAggregation, MaxAggregation
from torch.nn import ReLU, LayerNorm, Dropout
from torch_geometric.utils import to_undirected
from torch_geometric.nn import knn_graph
from torch_geometric.nn.pool import knn
from copy import deepcopy
import math
import sys


class ProteinRefiner(torch.nn.Module):
    def __init__(
        self,
        feature_dim: int = 64,
        hidden_dim: int = ProteinEncoderDefaults["hidden_dim"],
        out_dim: int = ProteinEncoderDefaults["out_dim"],
        atom_k: int = 15,
    ):
        super().__init__()
        self.attention_layer = KVEncoder()
        self.edge_MLP = PositionalEdgeGenerator(8)
        self.atom_k = atom_k

    def forward(self, x, pos, batch):
        device = x.device

        edge_data = self.edge_MLP(pos, *[torch.arange(x.size(0))]*2, self.atom_k, batch)

        edge_index = edge_data[0].to(device)
        edge_attr = edge_data[1].to(device)

        kv = self.attention_layer(x, edge_index, edge_attr)
        return kv
        
