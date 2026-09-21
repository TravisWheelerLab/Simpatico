from typing import List
from torch_geometric.nn import radius
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
