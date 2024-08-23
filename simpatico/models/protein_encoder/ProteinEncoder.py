import torch
from ..model_utils import ResLayer, ResBlock
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
        atom_voxel_k=10,
        vox_k=15,
    ):
        super().__init__()
        # GAT layers concatenate heads, so true hidden dim is (hidden_dim*heads) dimensional.
        adjusted_hidden_dim = hidden_dim * heads

        # dimensionality from concatenating two hidden_dim length embeds plus their distace
        edge_mlp_in_dim = adjusted_hidden_dim * 2 + 1

        # Nonlinear functions to pass edges through.
        self.atom_atom_edge_MLP = MLP([edge_mlp_in_dim, 8, 1])
        self.atom_vox_edge_MLP = MLP([edge_mlp_in_dim, 8, 1])
        self.vox_vox_edge_MLP = MLP([edge_mlp_in_dim, 8, 1])

        # Nearest-neighbor count for atom-atom network
        self.atom_k = atom_k
        # Nearest-neighbor count for atom-voxel network
        self.atom_voxel_k = atom_voxel_k
        # Nearest-neighbor count for voxel-voxel network
        self.vox_k = vox_k

        self.input_projection_layer = torch.nn.Linear(feature_dim, adjusted_hidden_dim)
        self.vox_input_projection_layer = torch.nn.Linear(
            feature_dim, adjusted_hidden_dim
        )
        self.residual_blocks = torch.nn.ModuleList(
            [ResBlock(hidden_dim, heads, block_depth) for _ in range(blocks)]
        )

        self.out_layer = torch.nn.Sequential(
            torch.nn.Linear((blocks + 1) * adjusted_hidden_dim, adjusted_hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(adjusted_hidden_dim, out_dim),
        )

    def forward(self, data):
        x, pos, patch_mask, batch = (data.x, data.pos, data.patch_mask, data.batch)

        vox_x = x[patch_mask]
        vox_h0 = self.vox_in(vox_x.float())
        vox_pos = pos[patch_mask]
        vox_batch = batch[patch_mask]

        prox_atom_index = knn(pos, vox_pos, self.vox_k, batch, vox_batch)[1].unique()

        atom_x = x[prox_atom_index]
        atom_h0 = self.atom_in(atom_x.float())
        atom_pos = pos[prox_atom_index]
        atom_batch = batch[prox_atom_index]

        # -- STEP 1 - ATOM-VOX MODEL ----------|
        aa_ei = knn_graph(atom_pos, self.atom_k, atom_batch)
        aa_weights = self.get_edge_weights(aa_ei, atom_pos, atom_pos, self.aa_edge_MLP)

        av_ei = knn(atom_pos, vox_pos, self.vox_k, atom_batch, vox_batch)
        av_weights = self.get_edge_weights(av_ei, vox_pos, atom_pos, self.av_edge_MLP)

        vv_ei = knn_graph(vox_pos, self.vox_k, vox_batch)
        vv_weights = self.get_edge_weights(vv_ei, vox_pos, vox_pos, self.vv_edge_MLP)

        atom_count = atom_x.size(0)

        # Swap edge index order so messages are passed FROM atoms TO voxels
        av_ei = torch.vstack((av_ei[1], av_ei[0] + atom_count))

        av_graph_x = torch.vstack((atom_h0, vox_h0))
        av_graph_ei = torch.hstack((aa_ei, av_ei, vv_ei))
        av_graph_ea = torch.hstack((aa_weights, av_weights, vv_weights))

        av_h1 = self.res_block1(av_graph_x, av_graph_ei, av_graph_ea)
        av_h2 = self.res_block2(av_h1, av_graph_ei, av_graph_ea)
        av_h3 = self.res_block3(av_h2, av_graph_ei, av_graph_ea)
        av_h4 = self.res_block3(av_h2, av_graph_ei, av_graph_ea)

        v_out = self.out_layer(
            torch.hstack(
                (
                    vox_h0,
                    av_h1[atom_count:],
                    av_h2[atom_count:],
                    av_h3[atom_count:],
                    av_h4[atom_count:],
                )
            )
        )

        return v_out
