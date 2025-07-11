from typing import List, Optional, Callable, Tuple
import sys
import os
from torch_geometric.data import Batch, Data
from torch_geometric.nn import radius, knn
import torch
from simpatico.utils import graph_utils
from simpatico.utils.utils import get_k_hop_edges
from pathlib import Path


def handle_no_overwrite(outfile):
    """
    check if output exists or is being generated.
    returns False if we should skip the file, otherwise True
    input: output file path
    output: bool
    """
    if os.path.exists(outfile) and os.path.getsize(outfile) > 0:
        return False
    else:
        Path(outfile).touch()
        return True


def report_results(queries, vector_db, results):
    target_list = ">query sources:\n"
    db_list = ">db sources:\n"
    results_content = ">results:\n"

    for t_i, file_string in enumerate(queries["sources"]):
        target_list += f"{t_i+1} {file_string}\n"

    for d_i, file_string in enumerate(vector_db["sources"]):
        db_list += f"{d_i+1} {file_string}\n"

    for qi, result_vals in enumerate(results):
        rank = 1
        q_item_index = queries["item_batch"] == qi
        q_source_idx = queries["file_batch"][q_item_index][0].item()
        q_source_item_idx = queries["source_index"][q_item_index][0].item()
        query_source = queries["sources"][q_source_idx]

        for result_idx in result_vals[1]:
            r_idx = result_idx.item()
            r_item_index = vector_db["item_batch"] == r_idx
            r_source_idx = vector_db["file_batch"][r_item_index][0].item()
            r_source_item_idx = vector_db["source_index"][r_item_index][0].item()

            r_source = vector_db["sources"][r_source_idx]
            results_content += f"{q_source_idx+1} {q_source_item_idx+1} {r_source_idx+1} {r_source_item_idx+1} {rank}\n"
            rank += 1

    output = target_list + db_list + results_content
    print(output)


class ProteinLigandDataLoader:
    def __init__(self, pl_graph_pairs: List[Data], batch_size: int):

        self.proteins = []
        self.ligands = []
        self.batch_size = batch_size

        for p_graph, l_graph in pl_graph_pairs:
            self.proteins.append(p_graph)
            self.ligands.append(l_graph)

        self.size = len(self.proteins)
        self.batch_iterator = self.new_batch_iterator()

        self.set_proximal_atom_masks()

    def new_batch_iterator(self):
        random_index = torch.randperm(self.size)
        for i in range(0, len(random_index), self.batch_size):
            yield random_index[i : i + self.batch_size]

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
    ):
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
                mol_graph = self.ligands[g_idx.item()]

                if hasattr(mol_graph, "scaffold") == False:
                    mol_graph.scaffold = torch.zeros(mol_graph.x.size(0)).bool()

                mol_list.append(mol_graph)

        if mols_only is False:
            protein_batch = Batch.from_data_list(protein_list)

        if proteins_only is True:
            return protein_batch

        mol_batch = Batch.from_data_list(mol_list)

        if mols_only is True:
            return mol_batch

        return protein_batch, mol_batch

    def get_random_batch(
        self,
        pocket_mask: bool = True,
        proteins_only: bool = False,
        mols_only: bool = False,
    ) -> Batch:

        random_index = next(self.batch_iterator, None)

        if random_index is None:
            self.batch_iterator = self.new_batch_iterator()
            random_index = next(self.batch_iterator)

        return self.get_batch(random_index, pocket_mask, proteins_only, mols_only)


class TrainingOutputHandler:
    def __init__(
        self,
        protein_embeds,
        protein_pos,
        protein_batch,
        mol_embeds,
        mol_pos,
        mol_batch,
    ):
        self.protein_embeds = protein_embeds
        self.protein_pos = protein_pos
        self.protein_batch = protein_batch

        self.mol_embeds = mol_embeds
        self.mol_pos = mol_pos
        self.mol_batch = mol_batch

        self.mol_actives, self.prot_actives = radius(
            protein_pos, mol_pos, 4, protein_batch, mol_batch
        )

    def get_mol_data(self, active_only=False):
        if active_only:
            mol_index = self.mol_actives
        else:
            mol_index = torch.arange(self.mol_embeds.size(0))

        return (
            self.mol_embeds[mol_index],
            self.mol_pos[mol_index],
            self.mol_batch[mol_index],
        )

    def get_protein_data(self, active_only=False):
        if active_only:
            protein_index = self.prot_actives
        else:
            protein_index = torch.arange(self.protein_embeds.size(0))

        return (
            self.protein_embeds[protein_index],
            self.protein_pos[protein_index],
            self.protein_batch[protein_index],
        )

    def get_hard_negatives(self, prot_anchor=True, difficulty=0.25):
        anchor_embeds, _, anchor_batch = (
            self.get_protein_data(True) if prot_anchor else self.get_mol_data(True)
        )
        negative_embeds, _, negative_batch = (
            self.get_mol_data() if prot_anchor else self.get_protein_data()
        )

        embed_distances = torch.cdist(anchor_embeds, negative_embeds)
        k = int(embed_distances.size(1) * difficulty)

        self_mask = anchor_batch.unsqueeze(1) == negative_batch.unsqueeze(0)
        # Set the distance between any protein atom and true ligand embed to maximum observed distance
        embed_distances[self_mask] = embed_distances.max()

        hard_negative_index = embed_distances.argsort(1)[:, :k][
            (
                torch.arange(embed_distances.size(0)),
                torch.randint(0, k, (embed_distances.size(0),)),
            )
        ]
        return hard_negative_index

    def get_self_negatives(self, prot_anchor=True):

        _, anchor_pos, anchor_batch = (
            self.get_protein_data(True) if prot_anchor else self.get_mol_data(True)
        )
        _, negative_pos, negative_batch = (
            self.get_mol_data() if prot_anchor else self.get_protein_data()
        )

        negative_index = []
        self_mask = anchor_batch.unsqueeze(1) == negative_batch.unsqueeze(0)

        for a_pos, row in zip(anchor_pos, self_mask):
            # indices of negative embeds from same complex
            self_samples = torch.where(row)[0]

            spatial_distances = torch.cdist(
                a_pos.unsqueeze(0), negative_pos[self_samples]
            )[0]
            k = int(self_samples.size(0) * 0.75)

            # sort indices of negative embeds by decreasing spatial distance from anchor embed.
            # Cut the quarter nearest embeds.
            farthest_self_negatives = self_samples[
                spatial_distances.argsort(descending=True)
            ][:k]

            # Append one of these random 3/4 farthest self-negatives to 'negative_index'
            negative_index.append(
                farthest_self_negatives[
                    torch.randperm(farthest_self_negatives.size(0))[0]
                ]
            )
        return torch.hstack(negative_index)

    def get_random_negatives(self, prot_anchor=True):

        _, _, positive_batch = (
            self.get_protein_data(True) if prot_anchor else self.get_mol_data(True)
        )
        _, _, negative_batch = (
            self.get_mol_data() if prot_anchor else self.get_protein_data()
        )

        random_index = []
        self_mask = positive_batch.unsqueeze(1) != negative_batch.unsqueeze(0)

        for row in self_mask:
            non_self_indices = torch.where(row)[0]
            random_idx = torch.randint(0, len(non_self_indices), (1,))
            random_index.append(non_self_indices[random_idx])

        return torch.hstack(random_index)

    def get_protein_anchor_pairs(self, difficulty=0.25):
        with torch.no_grad():
            random_negatives = self.get_random_negatives(prot_anchor=True)
            self_negatives = self.get_self_negatives(prot_anchor=True)
            hard_negatives = self.get_hard_negatives(
                prot_anchor=True, difficulty=difficulty
            )

        all_negatives = torch.hstack((random_negatives, self_negatives, hard_negatives))

        return (
            self.protein_embeds[self.prot_actives],
            self.mol_embeds[self.mol_actives],
            self.mol_embeds[all_negatives],
        )

    def get_mol_anchor_pairs(self, difficulty=0.25):
        with torch.no_grad():
            random_negatives = self.get_random_negatives(prot_anchor=False)
            self_negatives = self.get_self_negatives(prot_anchor=False)
            hard_negatives = self.get_hard_negatives(
                prot_anchor=False, difficulty=difficulty
            )

        all_negatives = torch.hstack((random_negatives, self_negatives, hard_negatives))

        return (
            self.mol_embeds[self.mol_actives],
            self.protein_embeds[self.prot_actives],
            self.protein_embeds[all_negatives],
        )
