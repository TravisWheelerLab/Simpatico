from typing import List, Optional, Callable, Tuple
import sys
from torch_geometric.data import Batch, Data
from torch_geometric.nn import radius, knn
import torch
from simpatico.utils import graph_utils
from simpatico.utils.utils import get_k_hop_edges


def update_h_count(h_data, increment=True):
    """
    Increment or decrement one-hot hydrogen data.
    """
    updated_h_data = torch.zeros_like(h_data)
    h_counts = torch.where(h_data)[1]

    if increment:
        h_counts += 1
    else:
        h_counts -= 1

    updated_h_data[torch.arange(h_counts.size(0)), h_counts] = 1
    return updated_h_data


def get_affected_atoms(mol_batch, altered_mask, k=3):
    affected_mask = torch.zeros(altered_mask.size(0)).bool()
    # Edges representing atoms within k covalent bonds will have a 1
    # before the kth index of the edge feature one hot
    edge_mask = mol_batch.edge_attr[:, :k].sum(1).bool()
    ei = mol_batch.edge_index[:, edge_mask]

    affected_edge_mask = altered_mask[ei].T.int().sum(1) > 0
    affected_nodes = ei[:, affected_edge_mask].flatten().unique()
    affected_mask[affected_nodes] = True

    return affected_mask


def shuffle_sidechains(mol_batch, k_affected=2):
    """
    Shuffles sidechain-scaffold connections on PyG batch of molecular graphs.
    """
    shuffled_batch = mol_batch.clone()

    # Set mask to True if corresponding scaffold atom
    # is adjacent to new sidechain or loses sidechain
    altered_mask = torch.zeros(shuffled_batch.x.size(0)).bool()

    # remove all 2+ hop edges
    one_edges = torch.where(shuffled_batch.edge_attr[:, 0] == 1)[0]
    one_ei = shuffled_batch.edge_index[:, one_edges]

    # for simplicity, make scaffold-to-sidechain edges directed
    scaffold_to_sc_mask = shuffled_batch.scaffold[one_ei]
    # negate second row so True-True pairs represent scaffold-to-sidechain edges
    scaffold_to_sc_mask[1] = ~scaffold_to_sc_mask[1]
    scaffold_to_sc_mask = scaffold_to_sc_mask.T.all(1)

    # Now there are no scaffold-to-sidechain edges
    updated_ei = one_ei[:, ~scaffold_to_sc_mask]

    # Get mask for sidechain-to-scaffold edges
    link_mask = shuffled_batch.scaffold[updated_ei]
    link_mask[0] = ~link_mask[0]
    link_mask = torch.all(link_mask.T, dim=1)

    old_link_edges = updated_ei[:, link_mask].clone()

    # reduce old linking scaffold atoms
    scaffold_link_atoms = updated_ei[1][link_mask]
    link_atom_h_data = shuffled_batch.x[scaffold_link_atoms, -4:].clone()

    shuffled_batch.x[scaffold_link_atoms, -4:] = update_h_count(
        link_atom_h_data, increment=True
    )

    altered_mask[scaffold_link_atoms] = True

    # set directional linking edges to random candidate linker carbon atoms
    candidate_mask = torch.all(
        torch.vstack(
            (
                # Conditions:
                #   - is a scaffold atom
                shuffled_batch.scaffold,
                #   - is a carbon
                shuffled_batch.x[:, 4] == 1,
                #   - is not fully oxidized
                shuffled_batch.x[:, -4] == 0,
            )
        ).T,
        dim=1,
    )

    candidate_index = torch.where(candidate_mask)[0]
    candidate_batch = shuffled_batch.batch[candidate_index]
    link_batch = shuffled_batch.batch[updated_ei[1, link_mask]]

    new_link_atoms = []

    for bi in link_batch.unique():
        candidate_link_atoms = candidate_index[candidate_batch == bi]
        shuffled_candidates = candidate_link_atoms[
            torch.randperm(candidate_link_atoms.size(0))
        ]

        # if there are fewer candidate link atoms than sidechains,
        # shuffling molecule requires specialized process.
        # Just skip for now.
        if shuffled_candidates.size(0) < (link_batch == bi).sum():
            link_mask[shuffled_batch.batch[updated_ei[1]] == bi] = False
            continue

        updated_link_atoms = shuffled_candidates[: (link_batch == bi).sum()]
        updated_mask = (old_link_edges[1, link_batch == bi] - updated_link_atoms).bool()

        # If sidechain is assigned to the same scaffold link atom,
        # negate altered mask at link atom index.
        altered_mask[updated_link_atoms[~updated_mask]] = False

        # likewise, set altered mask to True for any new scaffold link atoms
        altered_mask[updated_link_atoms[updated_mask]] = True

        new_link_atoms.append(shuffled_candidates[: (link_batch == bi).sum()])

    if len(new_link_atoms) == 0:
        return None

    new_link_atoms = torch.hstack(new_link_atoms)

    if new_link_atoms.size(0) == 0:
        return None

    updated_ei[1, link_mask] = new_link_atoms

    # oxidate new linking scaffold atoms
    new_link_h_data = shuffled_batch.x[new_link_atoms, -4:].clone()
    shuffled_batch.x[new_link_atoms, -4:] = update_h_count(
        new_link_h_data, increment=False
    )

    final_edge_index, final_edge_attr = get_k_hop_edges(updated_ei)
    shuffled_batch.edge_index = final_edge_index
    shuffled_batch.edge_attr = final_edge_attr
    shuffled_batch.affected_mask = get_affected_atoms(
        shuffled_batch, altered_mask, k_affected
    )

    return shuffled_batch


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
        batch_size: int,
        pocket_mask: bool = True,
        proteins_only: bool = False,
        mols_only: bool = False,
    ) -> Batch:
        random_index = torch.randperm(self.size)[:batch_size]
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
