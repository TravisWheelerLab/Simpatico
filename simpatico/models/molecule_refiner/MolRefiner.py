import torch
from simpatico.utils.model_utils import ResLayer, ResBlock
from torch_geometric.nn import (
    GATv2Conv,
    Sequential as PyG_Sequential,
)
from torch.nn import Sequential, Linear, ModuleList, SiLU
from typing import Optional
from simpatico.models import MolEncoderDefaults
from simpatico.utils.model_utils import CrossGraphUpdater


class MolRefiner(torch.nn.Module):

    def __init__(
        self,
        feature_dim: int = 64,
        hidden_dim: int = MolEncoderDefaults["hidden_dim"],
        out_dim: int = MolEncoderDefaults["out_dim"],
        heads: int = 4,
        blocks: int = 6,
        block_depth: int = 2,
        edge_dim: Optional[int] = MolEncoderDefaults["edge_dim"],
        **kwargs
    ):
        super().__init__()

        self.attention_layers = torch.nn.ModuleList([CrossGraphUpdater() for _ in range(4)])

    def forward(self, data, KV, kv_batch):
        """
        Forward method. Produces atom-level embeddings of PyG molecular graphs.
        Args:
            data (Batch): PyG batch of small-molecule graphs.
        Returns:
            (torch.Tensor): (N, self.out_dim) shaped tensor of embedding values generated for each atom.
        """
        x, edge_index, edge_attr, batch = (
            data.x,
            data.edge_index.long(),
            data.edge_attr.float(),
            data.batch
        )

        attention_outs = []

        for layer in self.attention_layers:
            x = x + layer(x, edge_index, edge_attr, *KV, batch, kv_batch)

        return x
        