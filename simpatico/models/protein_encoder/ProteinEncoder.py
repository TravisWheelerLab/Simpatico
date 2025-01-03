import torch
from simpatico.utils.model_utils import ResBlock, PositionalEdgeGenerator
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


class ProteinEncoder(torch.nn.Module):
    """
    Generates protein atom embeddings from protein graph.

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
        blocks=6,
        block_depth=2,
        atom_k=10,
        atom_vox_k=15,
        vox_k=15,
    ):
        super().__init__()
        # GAT layers concatenate heads, so true hidden dim is (hidden_dim*heads) dimensional.
        adjusted_hidden_dim = hidden_dim * heads

        # Nonlinear functions to pass edges through.
        self.atom_edge_MLP = PositionalEdgeGenerator(8)
        self.atom_vox_edge_MLP = PositionalEdgeGenerator(8)
        self.vox_edge_MLP = PositionalEdgeGenerator(8)

        # Nearest-neighbor count for atom-atom network
        self.atom_k = atom_k
        # Nearest-neighbor count for atom-voxel network
        self.atom_vox_k = atom_vox_k
        # Nearest-neighbor count for voxel-voxel network
        self.vox_k = vox_k

        self.atom_input_projection = torch.nn.Linear(feature_dim, adjusted_hidden_dim)
        self.vox_input_projection = torch.nn.Linear(feature_dim, adjusted_hidden_dim)
        self.residual_blocks = torch.nn.ModuleList(
            [
                ResBlock(hidden_dim, heads, block_depth, edge_dim=1)
                for _ in range(blocks)
            ]
        )

        self.output_projection = torch.nn.Sequential(
            torch.nn.Linear((blocks + 1) * adjusted_hidden_dim, adjusted_hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(adjusted_hidden_dim, out_dim),
        )

    def forward(self, data):
        x, pos, pocket_mask, batch = (
            data.x.float(),
            data.pos,
            data.pocket_mask,
            data.batch,
        )

        device = x.device

        # Trim atoms that are excessively far from the voxel nodes.
        trimmed_atom_index = knn(
            pos, pos[pocket_mask], self.atom_vox_k, batch, batch[pocket_mask]
        )[1].unique()

        atom_x = self.atom_input_projection(x[trimmed_atom_index])
        atom_pos = pos[trimmed_atom_index]
        atom_batch = batch[trimmed_atom_index]

        vox_x = self.vox_input_projection(x[pocket_mask])
        vox_pos = pos[pocket_mask]
        vox_batch = batch[pocket_mask]

        full_x = torch.vstack((atom_x, vox_x))
        full_pos = torch.vstack((atom_pos, vox_pos))
        full_batch = torch.hstack((atom_batch, vox_batch))

        atom_node_index = torch.arange(atom_x.size(0)).to(device)
        vox_node_index = (torch.arange(vox_x.size(0)) + atom_node_index.size(0)).to(
            device
        )

        edge_combos = [
            (self.atom_edge_MLP, [atom_node_index, atom_node_index, self.atom_k]),
            (
                self.atom_vox_edge_MLP,
                [vox_node_index, atom_node_index, self.atom_vox_k],
            ),
            (self.vox_edge_MLP, [vox_node_index, vox_node_index, self.vox_k]),
        ]

        aa_edges, av_edges, vv_edges = [
            f(full_pos, *edge_params, full_batch) for f, edge_params in edge_combos
        ]

        full_edge_index = torch.hstack((aa_edges[0], av_edges[0], vv_edges[0])).to(
            device
        )
        full_edge_attr = torch.vstack((aa_edges[1], av_edges[1], vv_edges[1])).to(
            device
        )

        rblock_outs = [full_x]

        for rblock in self.residual_blocks:
            rblock_outs.append(rblock(rblock_outs[-1], full_edge_index, full_edge_attr))

        encoding = self.output_projection(torch.hstack(rblock_outs)[vox_node_index])
        return (encoding, vox_pos, vox_batch)
