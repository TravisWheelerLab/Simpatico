from typing import List
from torch_geometric.nn import radius
from torch_geometric.data import Data
import torch


def get_proximal_atom_masks(
    protein_pos: torch.Tensor, ligand_pos: torch.Tensor, r: int = 4
) -> List[torch.Tensor]:
    """Get per-node boolean masks of protein and ligand graphs in complex, indicating which atoms are interacting."""
    protein_mask = torch.zeros(protein_pos.size(0)).bool()
    ligand_mask = torch.zeros(protein_pos.size(0)).bool()

    interaction_index = radius(protein_pos, ligand_pos, r)

    protein_mask[interaction_index[0].unique()] = True
    ligand_mask[interaction_index[1].unique()] = True

    return protein_mask, ligand_mask


def get_pocket_mask(
    protein_graph: Data, center: torch.Tensor, pocket_dim: int = 15
) -> torch.Tensor:
    zero_centered_range = torch.arange(-int(pocket_dim / 2), int(pocket_dim / 2))
    voxel_box = torch.cartesian_prod(*[zero_centered_range] * 3)
    pocket_coords = center + voxel_box

    voxel_filter = torch.zeros(pocket_coords.size(0)).bool()
    # Only include voxels within 4 angstroms of protein atom
    voxel_filter[radius(protein_graph.pos, voxel_box, 4)[0].unique()] = True
    # Exclude voxels that are closer than 2 angstroms to a protein atom
    voxel_filter[radius(protein_graph.pos, voxel_box, 2)[0].unique()] = False
    pocket_surface_atoms = radius(protein_graph.pos, pocket_coords[voxel_filter], 4)[
        1
    ].unique()
    pocket_mask = torch.zeros(protein_graph.pos.size(0)).bool()
    pocket_mask[pocket_surface_atoms] = True

    return pocket_mask
