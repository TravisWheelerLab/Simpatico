from typing import List, Optional, Callable, Tuple
import sys
import os
from torch_geometric.data import Batch, Data
from torch_geometric.nn import radius, knn
import torch
from simpatico.utils import graph_utils
from simpatico.utils.utils import get_k_hop_edges
from pathlib import Path


def handle_no_overwrite(outfile: str) -> bool:
    """
    If file does not exist, generate a sentinel file and return True.
    Otherwise return False
    Args:
        outfile (str): file path
    Returns:
        (bool): Truth indicates that output file did not exists.
    """
    if os.path.exists(outfile) and os.path.getsize(outfile) > 0:
        return False
    else:
        # Ensure that parallel jobs know this file is being worked on elsewhere.
        Path(outfile).touch()
        return True


def report_results(query_output: List) -> None:
    """
    Print human-readable results from vector database query/screen.
    Args:
        query_output (List): list containing data from vector database query produced by `simpatico query`
    Returns:
        (None): prints output
    """
    queries, vector_db, results = query_output
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

        for result_idx in result_vals[1]:
            r_idx = result_idx.item()
            r_item_index = vector_db["item_batch"] == r_idx
            r_source_idx = vector_db["file_batch"][r_item_index][0].item()
            r_source_item_idx = vector_db["source_index"][r_item_index][0].item()

            results_content += f"{q_source_idx+1} {q_source_item_idx+1} {r_source_idx+1} {r_source_item_idx+1} {rank}\n"
            rank += 1

    output = target_list + db_list + results_content
    print(output)


class ProteinLigandDataLoader:
    """
    Data loader for storing and producing protein-ligand pairs during training.

    Args:
        pl_graph_pairs (List[(Data, Data)]): List of protein-ligand graph pairs.
        batch_size (int): batch size

    Attributes:
        proteins (List[Data]): N-length list of proteins
        ligands (List[Data]): N-length list of ligands
        batch_size (int): batch size
    """

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
        """
        Generator for random indices in upcoming batches.

        Yields:
            (torch.Tensor): A 1D tensor containing indices for the current batch.
        """
        random_index = torch.randperm(self.size)
        for i in range(0, len(random_index), self.batch_size):
            yield random_index[i : i + self.batch_size]

    def set_proximal_atom_masks(self, r: Optional[int] = 4):
        """
        For each protein-ligand pair, add per-node boolean masks to graphs in self.proteins and self.ligands to indicate atoms involved in interactions.
        Args:
            r (int, optional): radius value that determines interaction-distance
        """
        for g_i in range(self.size):
            protein_mask, ligand_mask = graph_utils.get_proximal_atom_masks(
                self.proteins[g_i].pos, self.ligands[g_i].pos, r
            )
            self.proteins[g_i].proximal = protein_mask
            self.ligands[g_i].proximal = ligand_mask

    def get_random_negatives(self, mol_batch):
        """
        returns random index of all atoms in molecule batch for random negatives.
        Args:
            mol_batch (Batch): molecular graph batch
        Returns:
            (torch.Tensor): random index
        """
        return torch.randperm(mol_batch.x.size(0))

    def get_random_pocket(self, protein_graph: Data) -> torch.Tensor:
        """
        Selects random pocket coordinates based on graph's 'proximal' mask as pocket center
        Args:
            protein_graph (Data): protein graph
        Returns:
            (torch.Tensor): boolean mask to indicate pocket surface atoms
        """
        pocket_atoms = torch.where(protein_graph.proximal)[0]

        if pocket_atoms.size(0) == 0:
            return None

        pocket_atoms_pos = protein_graph.pos[pocket_atoms]
        random_pocket_atom = torch.randperm(pocket_atoms.size(0))[0]

        # get a local cluster of surface atoms which will be used to weight the location of the pocket
        random_pocket_neighbors = radius(
            pocket_atoms_pos, pocket_atoms_pos[random_pocket_atom], 5
        )[1].unique()

        # get weights to produce a randomized average of pocket atom locations
        # weight values should sum to 1 to produce a proper average
        random_pos_coefficients = torch.rand(random_pocket_neighbors.size(0))
        random_pos_coefficients /= random_pos_coefficients.sum()
        random_pos_coefficients = random_pos_coefficients.unsqueeze(1)

        # set pocket center to randomized average of pocket atom locations
        pocket_center = (
            pocket_atoms_pos[random_pocket_neighbors] * random_pos_coefficients
        ).sum(0)

        return graph_utils.get_pocket_mask(protein_graph, pocket_center)

    def get_batch(
        self,
        graph_index: torch.Tensor,
        pocket_mask: bool = True,
    ):
        """
        Get batch of protein graphs and corresponding batch of ligand graphs based on index.
        Args:
            graph_index (torch.Tensor): index of protein-ligand pairs.
            pocket_mask (bool, optional): if True, generates random pocket mask for protein graphs.

        Returns:
            (Batch, Batch): PyG batches of protein graphs and corresponding ligand graphs
        """

        protein_list = []
        mol_list = []

        for g_idx in graph_index:
            protein_graph = self.proteins[g_idx.item()].clone()
            if pocket_mask is True:
                mask = self.get_random_pocket(protein_graph)

                if mask is None:
                    continue

                protein_graph.pocket_mask = mask
            protein_list.append(protein_graph)

            mol_graph = self.ligands[g_idx.item()].clone()
            mol_list.append(mol_graph)

        protein_batch = Batch.from_data_list(protein_list)
        mol_batch = Batch.from_data_list(mol_list)

        return protein_batch, mol_batch

    def get_random_batch(self, pocket_mask: bool = True) -> Batch:
        """
        Get a random batch of protein graphs and corresponding batch of ligand graphs.
        Args:
            pocket_mask (bool, optional): if True, generates random pocket mask for protein graphs.

        Returns:
            (Batch, Batch): PyG batches of protein graphs and corresponding ligand graphs
        """
        random_index = next(self.batch_iterator, None)

        if random_index is None:
            self.batch_iterator = self.new_batch_iterator()
            random_index = next(self.batch_iterator)

        return self.get_batch(random_index, pocket_mask, proteins_only, mols_only)


class TrainingOutputHandler:
    """
    Organizes the output of protein and molecule encoders into inputs for the contrastive loss module during training.
    This entails selecting anchor-positive pairs and anchor-(random/self/hard) negative pairs.

    Args:
        protein_embeds (torch.Tensor)
        protein_pos (torch.Tensor)
        protein_batch (torch.Tensor)
        mol_embeds (torch.Tensor)
        mol_pos (torch.Tensor)
        mol_batch (torch.Tensor)
        interaction_radius (int, optional): threshold distance to define atomic interaction (default 4)

    Attributes:
        protein_embeds (torch.Tensor)
        protein_pos (torch.Tensor)
        protein_batch (torch.Tensor)
        mol_embeds (torch.Tensor)
        mol_pos (torch.Tensor)
        mol_batch (torch.Tensor)
        mol_actives (torch.Tensor): index of molecular atoms involved in interaction with corresponding item in self.prot_actives
        prot_actives (torch.Tensor): index of protein atoms involved in interaction with corresponding item in self.mol_actives
    """

    def __init__(
        self,
        protein_embeds: torch.Tensor,
        protein_pos: torch.Tensor,
        protein_batch: torch.Tensor,
        mol_embeds: torch.Tensor,
        mol_pos: torch.Tensor,
        mol_batch: torch.Tensor,
        interaction_radius: int = 4,
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
        """
        Fetches encoding, positional and batch data of molecular embeddings.
        Args:
            active_only (bool): If true, only fetch values corresponding to active atom embeds (used as contrastive learning anchors).
        Returns:
            (torch.Tensor, torch.Tensor, torch.Tensor): encoding, positional and batch data.
        """
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
        """
        Fetches encoding, positional and batch data of protein embeddings.
        Args:
            active_only (bool): If true, only fetch values corresponding to active atom embeds (used as contrastive learning anchors).
        Returns:
            (torch.Tensor, torch.Tensor, torch.Tensor): encoding, positional and batch data.
        """
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
        """
        Fetches embeddings from non-corresponding structures with upper-quartile embedding similarity.
        Args:
            prot_anchor (bool, optional): If True, fetch hard negatives from molecule data, otherwise from protein data (default True).
            difficulty (float, optional): value between 0 and 1 specifying the quartile threshold
                                          (e.g. diffulty=0.1, negatives will be selected from 10% nearest non-positive embeddings)
        Returns:
            (torch.Tensor): index of hard negative embeds.
        """
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
        # This prevents positive embeds from being selected
        embed_distances[self_mask] = embed_distances.max()

        # randomly select a negative from among the top-k closest embeds per anchor atom
        hard_negative_index = embed_distances.argsort(1)[:, :k][
            (
                torch.arange(embed_distances.size(0)),
                torch.randint(0, k, (embed_distances.size(0),)),
            )
        ]
        return hard_negative_index

    def get_self_negatives(self, prot_anchor=True):
        """
        Fetches non-interacting embeddings from corresponding structures.
        Args:
            prot_anchor (bool, optional): If True, fetch hard negatives from molecule data, otherwise from protein data (default True).
        Returns:
            (torch.Tensor): self negatives index
        """

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
        """
        Fetches random embeddings from non-corresponding structures.
        Args:
            prot_anchor (bool, optional): If True, fetch hard negatives from molecule data, otherwise from protein data (default True).
        Returns:
            (torch.Tensor): random negatives index
        """

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

    def get_anchors_positives_negatives(self, prot_anchor=True, difficulty=0.25):
        """
        Assembles protein-anchor-negative embed collections to pass to contrastive loss module.
        Args:
            difficulty (float, optional): value between 0 and 1 specifying the quartile threshold
                                          (e.g. diffulty=0.1, negatives will be selected from 10% nearest non-positive embeddings)

        Returns:
            (torch.Tensor, torch.Tensor, torch.Tensor): Two (N, embed_dim) tensors of anchors and positives.
                                                        One (3*N, embed_dim) tensor of random, self, and hard negatives.
        """
        anchor_embeds = self.protein_embeds if prot_anchor else self.mol_embeds
        anchor_actives = self.prot_actives if prot_anchor else self.mol_actives
        pos_neg_embeds = self.mol_embeds if prot_anchor else self.protein_embeds
        pos_neg_actives = self.mol_actives if prot_anchor else self.prot_actives

        with torch.no_grad():
            random_negatives = self.get_random_negatives(prot_anchor=prot_anchor)
            self_negatives = self.get_self_negatives(prot_anchor=prot_anchor)
            hard_negatives = self.get_hard_negatives(
                prot_anchor=prot_anchor, difficulty=difficulty
            )

        all_negatives = torch.hstack((random_negatives, self_negatives, hard_negatives))

        return (
            anchor_embeds[anchor_actives],
            pos_neg_embeds[pos_neg_actives],
            pos_neg_embeds[all_negatives],
        )
