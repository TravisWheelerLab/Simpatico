from typing import List, Optional, Callable, Tuple
from torch_geometric.data import Batch, Data
from torch_geometric.nn import radius
import torch
from simpatico.utils import graph_utils


class ProteinLigandDataLoader:
    def __init__(
        self,
        protein_data_list: List[Data],
        ligand_data_list: List[Data],
        parity_check: Optional[Callable[[str, str], bool]] = None,
    ):
        if parity_check is not None:
            for a, b in zip(protein_data_list, ligand_data_list):
                if parity_check(a.name, b.name) == False:
                    sys.exit("Noncompatible lists provided.")

        self.proteins = protein_data_list
        self.ligands = ligand_data_list
        self.size = len(self.proteins)

        self.set_proximal_atom_masks()

    def set_proximal_atom_masks(self, r: Optional[int] = 4):
        """Adds per-node boolean masks to graphs in self.proteins and self.ligands indicating whether or not atom is involved in interaction."""
        for g_i in range(self.size):
            protein_mask, ligand_mask = graph_utils.get_proximal_atom_masks(
                self.proteins[g_i].pos, self.ligands[g_i].pos, r
            )
            self.proteins[g_i].proximal = protein_mask
            self.ligands[g_i].proximal = ligand_mask

    def get_random_pocket(self, protein_graph: Data) -> torch.Tensor:
        """Selects random coordinate based on graph's 'proximal' mask as pocket center"""
        pocket_atoms = torch.where(protein_graph.prox_mask)[0]
        pocket_atoms_pos = protein_graph.pos[pocket_atoms]
        random_pocket_atom = torch.randperm(pocket_atoms.size(0))[0]

        random_pocket_neighbors = radius(
            pocket_atoms_pos, pocket_atoms_pos[random_pocket_atom], 5
        )[1].unique()

        pocket_center = (
            random_pocket_neighbors
            * torch.rand(random_pocket_neighbors.size(0)).unsqueeze(1)
        ).sum(0)

        return graph_utils.get_pocket_mask(protein_graph, pocket_center)

    def get_batch(
        self,
        graph_index: torch.Tensor,
        pocket_mask: bool = True,
        proteins_only: bool = False,
        mols_only: bool = False,
    ) -> Tuple[Batch, Batch] | Batch:
        protein_list = []
        mol_list = []

        for g_idx in graph_index:
            if mols_only is False:
                protein_graph = self.proteins[g_idx.item()].clone()
                if pocket_mask is True:
                    protein_graph.pocket_mask = self.get_random_pocket(protein_graph)
                protein_list.append(protein_graph)

            if proteins_only is False:
                mol_list.append(self.ligands[g_idx.item()])

        if proteins_only is True:
            return Batch.from_data_list(protein_list)

        if mols_only is True:
            return Batch.from_data_list(mol_list)

        return Batch.from_data_list(protein_list), Batch.from_data_list(mol_list)

    def get_random_batch(
        self,
        batch_size: int,
        pocket_mask: bool = True,
        proteins_only: bool = False,
        mols_only: bool = False,
    ) -> Batch:
        random_index = torch.randperm(len(self.size))[:batch_size]
        return self.get_batch(random_index, pocket_mask, proteins_only, mols_only)
