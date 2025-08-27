import torch
from simpatico.utils.pdb_utils import include_residues
from simpatico.utils.model_utils import ResBlock, PositionalEdgeGenerator
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
from simpatico import config


class ProteinEncoder(torch.nn.Module):
    """
    Generates protein atom embeddings from protein graph.

    Args:
        feature_dim (int, optional): Dimensionality of the input features.
        hidden_dim (int, optional): Dimensionality of hidden layers.
        out_dim (int, optional): Dimensionality of output.
        heads (int, optional): Number of attention heads in graph attention layers (default 4).
        blocks (int, optional): Number of residual blocks in model (default 3).
        block_depth (int, optional): Number of residual layers in residual blocks (default 2).

    Attributes:
        input_projection_layer (torch.nn.Module): linear layer to project node features into [num_nodes, dims * heads]
                                                  size tensor expected by residual block.
        residual_blocks (torch.nn.ModuleList): List of residual blocks. Outputs will be concatenated in final layer.
        ouptupt_projection_layer (torch.nn.Module): Non-linear layer which takes concatenation of residual_blocks outputs
                                                    and outputs final embeddings.
    """

    def __init__(
        self,
        feature_dim: int = ProteinEncoderDefaults["feature_dim"],
        hidden_dim: int = ProteinEncoderDefaults["hidden_dim"],
        out_dim: int = ProteinEncoderDefaults["out_dim"],
        heads: int = 4,
        blocks: int = 6,
        block_depth: int = 2,
        backbone_k: int = 20,
        alpha_c_k: int = 20,
    ):
        # NAMING CONVENTION NOTE:
        # references to `vox` or `voxels` is a legacy convention from earlier versions of the model.
        # these refer to `surface_atoms`
        # this will change in next version.

        super().__init__()
        # GAT layers concatenate heads, so true hidden dim is (hidden_dim*heads) dimensional.
        adjusted_hidden_dim = hidden_dim * heads

        # Nonlinear functions to pass edges through.
        self.backbone_edge_MLP = PositionalEdgeGenerator(8)
        self.residue_backbone_edge_MLP = PositionalEdgeGenerator(8)
        self.alpha_carbon_edge_MLP = PositionalEdgeGenerator(8)

        # Nearest-neighbor count for atom-atom network
        self.backbone_k = backbone_k 
        # Nearest-neighbor count for atom-voxel network
        self.alpha_c_k = alpha_c_k 

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
        self.backbone_index = torch.tensor([config['atom_vocab'].index(w) for w in ['C','CA','N','O']])
        self.alpha_c_index = config['atom_vocab'].index('CA')

    def forward(self, data):
        """
        Forward method. Produces atom-level embeddings of PyG protein-pocket graphs.
        Args:
            data (Batch): PyG batch of protein-pocket graphs.
        Returns:
            (torch.Tensor): (N, self.out_dim) shaped tensor of embedding values generated for pocket-surface atoms.
            (torch.Tensor): (N, 3) sized tensor corresponding to pocket-surface postions.
            (torch.Tensor): (N) sized tensor corresponding to pocket-surface postions.
        """
        x, pos, pocket_mask, residue = (
            data.x.float(),
            data.pos,
            data.pocket_mask,
            data.residue
        )

        if data.batch is None:
            batch = torch.zeros(len(x))
        else:
            batch = data.batch

        residue_keys = torch.vstack((residue, batch)).T
        _, residue_batch = torch.unique(residue_keys, dim=0, return_inverse=True)

        device = x.device

        # Trim atoms that are excessively far from the voxel nodes.
        trimmed_atom_index = knn(
            pos, pos[pocket_mask], 25, batch, batch[pocket_mask]
        )[1].unique()

        trimmed_atom_index = include_residues(data, trimmed_atom_index)

        x_trimmed = x[trimmed_atom_index] 
        pocket_mask = pocket_mask[trimmed_atom_index]
        atom_pos = pos[trimmed_atom_index]
        atom_batch = batch[trimmed_atom_index]
        residue_batch = residue_batch[trimmed_atom_index]

        atom_x = self.atom_input_projection(x_trimmed)

        backbone_atoms = torch.where(x_trimmed[:, self.backbone_index].any(1))[0]
        sidechain_atoms = torch.where(~x_trimmed[:, self.backbone_index].any(1))[0]
        alpha_carbon_atoms = torch.where(x_trimmed[:, self.alpha_c_index])[0]

        edge_combos = [
            (self.backbone_edge_MLP, [backbone_atoms, backbone_atoms, self.backbone_k]),
            (self.alpha_carbon_edge_MLP, [alpha_carbon_atoms, alpha_carbon_atoms, self.alpha_c_k])
        ]

        sc_ca_edges = sidechain_to_alpha_carbon_edges(sidechain_atoms, alpha_carbon_atoms, residue_batch)
        sc_ca_features = torch.ones(sc_ca_edges.size(1)).unsqueeze(1).to(device)

        bb_edges, ac_edges  = [
            f(atom_pos, *edge_params, atom_batch) for f, edge_params in edge_combos
        ]

        full_edge_index = torch.hstack((bb_edges[0], ac_edges[0], sc_ca_edges)).to(
            device
        )
        full_edge_attr = torch.vstack((bb_edges[1], ac_edges[1], sc_ca_features)).to(
            device
        )

        rblock_outs = [atom_x]

        for rblock in self.residual_blocks:
            rblock_outs.append(rblock(rblock_outs[-1], full_edge_index, full_edge_attr))

        encoding = self.output_projection(torch.hstack(rblock_outs)[pocket_mask])
        return (encoding, atom_pos[pocket_mask], atom_batch[pocket_mask])

def sidechain_to_alpha_carbon_edges(sidechain_atoms, alpha_carbons, residue_batch):
    batch_a = residue_batch[sidechain_atoms]   # shape [len(index_a)]
    batch_b = residue_batch[alpha_carbons]   # shape [len(index_b)]
    
    # broadcast compare
    mask = batch_a[:, None] == batch_b[None, :]  # [len_a, len_b]
    
    # get all matching (a_idx, b_idx) pairs
    a_idx, b_idx_rel = mask.nonzero(as_tuple=True)
    edge_index = torch.stack([alpha_carbons[b_idx_rel], sidechain_atoms[a_idx]])

    return edge_index