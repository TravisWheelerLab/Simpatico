from typing import List
from torch_geometric.nn import radius
from torch_geometric.data import Data
import torch


def get_proximal_atom_masks(
    protein_pos: torch.Tensor, ligand_pos: torch.Tensor, r: float = 4
) -> List[torch.Tensor]:
    """
    Generate per-atom masks for protein and ligand graphs indicating which are involved in interactions.
    Args:
        protein_pos (torch.Tensor): position values of protein graph.
        ligand_pos (torch.Tensor): position values of ligand graph.
        r (float, optional): threshold distance for interactivity (default = 4).
    Returns:
        (torch.Tensor, torch.Tensor): boolean masks for protein and ligand graphs.
    """
    protein_mask = torch.zeros(protein_pos.size(0)).bool()
    ligand_mask = torch.zeros(ligand_pos.size(0)).bool()

    interaction_index = radius(protein_pos, ligand_pos, r)

    protein_mask[interaction_index[1].unique()] = True
    ligand_mask[interaction_index[0].unique()] = True

    return protein_mask, ligand_mask


def get_pocket_mask(
    protein_graph: Data, center: torch.Tensor, pocket_dim: int = 20
) -> torch.Tensor:
    """
    Specify a protein pocket  (i.e. a pocket-surface atom mask) based on voxel box of width, length, and height `pocket_dim`.
    Args:
        protein_graph (Data): protein PyG graph
        center (torch.Tensor): center of voxel sphere
        pocket_dim (int, optional): value used as width, length, and height of pocket box.
    Returns:
        (torch.Tensor): boolean mask specifying pocket surface atoms.
    """
    zero_centered_range = torch.arange(-int(pocket_dim / 2), int(pocket_dim / 2) + 1)
    voxel_box = torch.cartesian_prod(*[zero_centered_range] * 3).float()
    pocket_coords = center + voxel_box

    voxel_filter = torch.zeros(pocket_coords.size(0)).bool()
    # Only include voxels within 4 angstroms of protein atom
    voxel_filter[
        radius(protein_graph.pos, pocket_coords, 4, max_num_neighbors=999)[0].unique()
    ] = True
    # Exclude voxels that are closer than 2 angstroms to a protein atom
    voxel_filter[
        radius(protein_graph.pos, pocket_coords, 2, max_num_neighbors=999)[0].unique()
    ] = False
    pocket_surface_atoms = radius(
        protein_graph.pos, pocket_coords[voxel_filter], 4, max_num_neighbors=999
    )[1].unique()
    pocket_mask = torch.zeros(protein_graph.pos.size(0)).bool()
    pocket_mask[pocket_surface_atoms] = True

    return pocket_mask
