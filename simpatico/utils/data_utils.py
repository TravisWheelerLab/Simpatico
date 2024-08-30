from typing import List, Optional, Callable, Tuple
import sys
from torch_geometric.data import Batch, Data
from torch_geometric.nn import radius, knn
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

    def get_random_negatives(self, mol_batch):
        return torch.randperm(mol_batch.x.size(0))

    def get_self_negatives(self, mol_batch):
        negative_index = []

        for mol_i in torch.unique(mol_batch.batch):
            self_index = torch.where(mol_batch.batch == mol_i)[0]
            distant_indices = torch.cdist(
                mol_batch.pos[self_index], mol_batch.pos[self_index]
            ).argsort(dim=1, descending=True)
            distant_indices = distant_indices[:, : int(distant_indices.size(1) * 0.75)]
            r_indices = torch.randint(
                0, distant_indices.size(1), (distant_indices.size(0),)
            )
            shuffled_indices = distant_indices[
                (torch.arange(distant_indices.size(0)), r_indices)
            ]
            negative_index.append(self_index[shuffled_indices])

        return torch.hstack(negative_index)

    def get_random_pocket(self, protein_graph: Data) -> torch.Tensor:
        """Selects random coordinate based on graph's 'proximal' mask as pocket center"""
        pocket_atoms = torch.where(protein_graph.proximal)[0]
        pocket_atoms_pos = protein_graph.pos[pocket_atoms]
        random_pocket_atom = torch.randperm(pocket_atoms.size(0))[0]

        random_pocket_neighbors = radius(
            pocket_atoms_pos, pocket_atoms_pos[random_pocket_atom], 5
        )[1].unique()

        random_pos_coefficients = torch.rand(random_pocket_neighbors.size(0))
        random_pos_coefficients /= random_pos_coefficients.sum()
        random_pos_coefficients = random_pos_coefficients.unsqueeze(1)

        pocket_center = (
            pocket_atoms_pos[random_pocket_neighbors] * random_pos_coefficients
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

        protein_batch = Batch.from_data_list(protein_list)
        if proteins_only is True:
            return protein_batch

        mol_batch = Batch.from_data_list(mol_list)

        mol_batch.random_negatives = self.get_random_negatives(mol_batch)
        mol_batch.self_negatives = self.get_self_negatives(mol_batch)

        if mols_only is True:
            return mol_batch

        return protein_batch, mol_batch, positive_pair_index

    def get_random_batch(
        self,
        batch_size: int,
        pocket_mask: bool = True,
        proteins_only: bool = False,
        mols_only: bool = False,
    ) -> Batch:
        random_index = torch.randperm(self.size)[:batch_size]
        return self.get_batch(random_index, pocket_mask, proteins_only, mols_only)


class TrainingOutputHandler:
    def __init__(
        self, protein_embeds, protein_pos, protein_batch, mol_embeds, mol_pos, mol_batch
    ):
        self.protein_embeds = protein_embeds
        self.protein_pos = protein_pos
        self.protein_batch = protein_batch

        self.mol_embeds = mol_embeds
        self.mol_pos = mol_pos
        self.mol_batch = mol_batch

        self.prot_positives, self.mol_positives = radius(
            mol_pos,
            protein_pos,
            4,
        )

    def get_self_mask(self):
        positive_batch = self.protein_batch[self.prot_positives]
        batch_index = self.mol_batch.unsqueeze(0).repeat(positive_batch, 1)
        self_mask = torch.zeros_like(batch_index).bool()
        self_mask[batch_index == positive_batch.unsqueeze(1)] = True
        return self_mask

    def get_hard_negatives(self, k=25):
        positive_embeds = self.protein_embeds[self.prot_positives]
        embed_distances = torch.cdist(positive_embeds, self.mol_embeds)

        self_mask = self.get_self_mask()

        embed_distances[self_mask] = 999
        hard_negative_index = embed_distances.argsort(1)[:, :k][
            (
                torch.arange(embed_distances.size(0)),
                torch.randint(0, k, (embed_distances.size(0),)),
            )
        ]
        return hard_negative_index

    def get_random_negatives(self):
        self_mask = self.get_self_mask()
