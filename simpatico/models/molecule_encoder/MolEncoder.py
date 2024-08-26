import torch
from ..model_utils import ResLayer, ResBlock
from torch_geometric.nn import (
    GATv2Conv,
    Sequential as PyG_Sequential,
)
from torch.nn import Sequential
from typing import Optional


class MolEncoder(torch.nn.Module):
    """
    Generates molecular atom embeddings from molecular graph.

    Attributes:
        input_projection_layer (torch.nn.Module): linear layer to project node features into [num_nodes, dims * heads]
                                                  size tensor expected by residual block.
        residual_blocks (torch.nn.ModuleList): List of residual blocks. Outputs will be concatenated in final layer.
        ouptupt_projection_layer (torch.nn.Module): Non-linear layer which takes concatenation of residual_blocks outputs
                                                    and outputs final embeddings.

    Args:
        feature_dim (int): Dimensionality of the input features.
        hidden_dim (int): Dimensionality of hidden layers.
        out_dim (int): Dimensionality of output.
        heads (int, optional): Number of attention heads in graph attention layers (default 4).
        blocks (int, optional): Number of residual blocks in model (default 3).
        block_depth (int, optional): Number of residual layers in residual blocks (default 2).

    Returns:
        (torch.Tensor): 'out_dim' dimensional atom embeddings
    """

    def __init__(
        self,
        feature_dim,
        hidden_dim,
        out_dim,
        heads=4,
        blocks=3,
        block_depth=2,
        edge_dim: Optional[int] = None,
        **kwargs
    ):
        super().__init__()
        self.input_projection_layer = torch.nn.Linear(feature_dim, hidden_dim * heads)

        self.residual_blocks = torch.nn.ModuleList(
            [
                ResBlock(hidden_dim, heads, block_depth, edge_dim=edge_dim)
                for _ in range(blocks)
            ]
        )

        self.output_projection_layer = Sequential(
            torch.nn.Linear((blocks + 1) * hidden_dim * heads, hidden_dim * heads),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim * heads, out_dim),
        )

    def forward(self, data):
        x, edge_index, edge_attr = (
            data.x,
            data.edge_index.long(),
            data.edge_attr.float(),
        )
        # Outputs will be concatenated and input to final layer.
        rblock_outs = [self.input_projection_layer(x.float())]

        for rblock in self.residual_blocks:
            rblock_outs.append(rblock(rblock_outs[-1], edge_index, edge_attr))

        encoding = self.output_projection_layer(torch.hstack(rblock_outs))
        return encoding
