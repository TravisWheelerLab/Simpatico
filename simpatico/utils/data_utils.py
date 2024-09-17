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
                    sys.exit("Non-compatible lists provided.")

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

    def get_random_pocket(self, protein_graph: Data) -> torch.Tensor:
        """Selects random coordinate based on graph's 'proximal' mask as pocket center"""
        pocket_atoms = torch.where(protein_graph.proximal)[0]

        if pocket_atoms.size(0) == 0:
            return None

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
                    mask = self.get_random_pocket(protein_graph)

                    if mask is None:
                        continue

                    protein_graph.pocket_mask = mask
                protein_list.append(protein_graph)

            if proteins_only is False:
                mol_list.append(self.ligands[g_idx.item()])

        protein_batch = Batch.from_data_list(protein_list)
        if proteins_only is True:
            return protein_batch

        mol_batch = Batch.from_data_list(mol_list)

        if mols_only is True:
            return mol_batch

        return protein_batch, mol_batch

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

        self.mol_positives, self.prot_positives = radius(
            protein_pos, mol_pos, 4, protein_batch, mol_batch
        )

        self.self_mask = self.get_self_mask()

    def get_self_mask(self):
        positive_batch = self.protein_batch[self.prot_positives]
        batch_index = self.mol_batch.unsqueeze(0).repeat(positive_batch.size(0), 1)
        self_mask = torch.zeros_like(batch_index).bool()
        self_mask[batch_index == positive_batch.unsqueeze(1)] = True
        return self_mask

    def get_hard_negatives(self, difficulty=0.25):
        positive_embeds = self.protein_embeds[self.prot_positives]
        embed_distances = torch.cdist(positive_embeds, self.mol_embeds)
        k = int(embed_distances.size(1) * difficulty)

        embed_distances[self.self_mask] = 999
        hard_negative_index = embed_distances.argsort(1)[:, :k][
            (
                torch.arange(embed_distances.size(0)),
                torch.randint(0, k, (embed_distances.size(0),)),
            )
        ]
        return hard_negative_index

    def get_self_negatives(self):
        negative_index = []

        for mol_i in self.mol_positives:
            self_index = torch.where(self.mol_batch == self.mol_batch[mol_i])[0]
            # Sample only from the 3/4 farthest atoms
            distant_indices = torch.cdist(
                self.mol_pos[mol_i].unsqueeze(0), self.mol_pos[self_index]
            )[0].argsort(descending=True)[: int(self_index.size(0) * 0.75)]

            negative_index.append(
                distant_indices[torch.randint(0, distant_indices.size(0), (1,))]
            )

        return torch.hstack(negative_index)

    def get_random_negatives(self):
        random_index = []

        for row in self.self_mask:
            non_self_indices = torch.where(~row)[0]
            random_idx = torch.randint(0, len(non_self_indices), (1,))
            random_index.append(non_self_indices[random_idx])

        return torch.hstack(random_index)

    def get_all_train_pairs(self, difficulty=0.25):
        random_negatives = self.get_random_negatives()
        self_negatives = self.get_self_negatives()
        hard_negatives = self.get_hard_negatives(difficulty)

        return (
            self.prot_positives,
            self.mol_positives,
            random_negatives,
            self_negatives,
            hard_negatives,
        )
