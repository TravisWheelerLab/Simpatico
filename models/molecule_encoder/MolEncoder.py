import torch
from torch_geometric.nn import (
    GATv2Conv,
    Sequential as PyG_Sequential,
)
from torch_geometric.nn.norm import BatchNorm

from torch.nn import Sequential, ReLU, Linear
from torch_geometric.nn.aggr import (
    AttentionalAggregation,
    MaxAggregation,
    MeanAggregation,
)
from torch_geometric.utils import subgraph
from torch_geometric.nn.models import MLP


class ResLayer(torch.nn.Module):
    def __init__(self, dims, act=ReLU, dr=0.25, heads=6):
        super().__init__()
        self.act = act()
        self.layer = GATv2Conv(dims * heads, dims, heads=heads, edge_dim=3)
        self.dropout = torch.nn.Dropout(dr)

    def forward(self, x, edge_index, edge_attr):
        x = x + self.layer(x, edge_index, edge_attr)
        x = self.act(x)
        return self.dropout(x)


class ResBlock(torch.nn.Module):
    def __init__(self, dims, heads, depth):
        super().__init__()
        self.res_block = PyG_Sequential(
            "x, edge_index, edge_attr",
            [
                (
                    ResLayer(dims, heads=heads),
                    "x, edge_index, edge_attr -> x",
                )
                for _ in range(depth)
            ],
        )

    def forward(self, x, ei, ea):
        return self.res_block(x, ei, ea)


class MolEmbedder(torch.nn.Module):
    """
    Generates molecular atom embeddings.
    Replaces 'substructure nodes' with single 'context node', if provided
    """

    def __init__(self, feature_dim, hidden_dim, out_dim, **kwargs):
        super().__init__()

        self.heads = 4
        self.block_depth = 2

        self.projection_layer = torch.nn.Linear(feature_dim, hidden_dim * self.heads)

        self.res_block1 = ResBlock(hidden_dim, self.heads, self.block_depth)
        self.res_block2 = ResBlock(hidden_dim, self.heads, self.block_depth)
        self.res_block3 = ResBlock(hidden_dim, self.heads, self.block_depth)

        self.out_layer = Sequential(
            torch.nn.Linear(4 * hidden_dim * self.heads, hidden_dim * self.heads),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim * self.heads, out_dim),
        )
        self.act = torch.nn.ReLU()

    def forward(self, data):
        x, edge_index, edge_attr, batch = (
            data.x,
            data.edge_index.long(),
            data.edge_attr.float(),
            data.batch,
        )

        h0 = self.projection_layer(x.float())

        h1 = self.res_block1(h0, edge_index, edge_attr)
        h2 = self.res_block2(h1, edge_index, edge_attr)
        h3 = self.res_block3(h2, edge_index, edge_attr)

        enc = self.out_layer(torch.hstack((h0, h1, h2, h3)))

        return enc
