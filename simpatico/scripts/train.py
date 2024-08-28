# scripts/train.py
import sys
import argparse
import torch
from typing import List, Tuple, Optional
from torch_geometric.data import Data, Batch
from torch_geometric.nn import radius
from typing import Callable


# API: simpatico train -d /path/to/data
def train(model, train_loader, optimizer, criterion, device):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print(
                f"Train Epoch: [{batch_idx * len(data)}/{len(train_loader.dataset)}]\tLoss: {loss.item():.6f}"
            )


def check_pdb_ids(a: str, b: str) -> bool:
    # Based on pdbbind data convention of [PDB_ID]_[SUFFIX]
    # check that strings a and b have the same PDB_ID value.
    def get_id(x: str) -> str:
        return x.split("/")[-1].split("_")[0]

    return get_id(a) == get_id(b)


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

    def get_proximal_atom_masks(
        self, protein_pos: torch.Tensor, ligand_pos: torch.Tensor, r: Optional[int] = 4
    ) -> List[torch.Tensor]:
        """Get per-node boolean masks of protein and ligand graphs in complex, indicating which atoms are interacting."""
        protein_mask = torch.zeros(protein_pos.size(0)).bool()
        ligand_mask = torch.zeros(protein_pos.size(0)).bool()

        interaction_index = radius(protein_pos, ligand_pos, r)

        protein_mask[interaction_index[0].unique()] = True
        ligand_mask[interaction_index[1].unique()] = True

        return protein_mask, ligand_mask

    def set_proximal_atom_masks(self, r: Optional[int] = 4):
        """Adds per-node boolean masks to graphs in self.proteins and self.ligands indicating whether or not atom is involved in interaction."""
        for g_i in range(self.size):
            protein_pos = self.proteins[g_i].pos
            ligand_pos = self.ligands[g_i].pos
            protein_mask, ligand_mask = self.get_proximal_atom_masks(
                protein_pos, ligand_pos, r
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

        return self.get_pocket_mask(protein_graph, pocket_center)

    def get_pocket_mask(
        self, protein_graph: Data, center: torch.Tensor, pocket_dim: int = 15
    ):
        zero_centered_range = torch.arange(-int(pocket_dim / 2), int(pocket_dim / 2))
        voxel_box = torch.cartesian_prod(*[zero_centered_range] * 3)
        pocket_coords = center + voxel_box

        voxel_filter = torch.zeros(pocket_coords.size(0)).bool()
        # Only include voxels within 4 angstroms of protein atom
        voxel_filter[radius(protein_graph.pos, voxel_box, 4)[0].unique()] = True
        # Exclude voxels that are closer than 2 angstroms to a protein atom
        voxel_filter[radius(protein_graph.pos, voxel_box, 2)[0].unique()] = False
        pocket_surface_atoms = radius(
            protein_graph.pos, pocket_coords[voxel_filter], 4
        )[1].unique()
        pocket_mask = torch.zeros(protein_graph.pos.size(0)).bool()
        pocket_mask[pocket_surface_atoms] = True

        return pocket_mask

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


def main(args):
    # Load data
    train_data, validation_data = torch.load(args.data_path)

    train_loader = ProteinLigandDataLoader(*train_data, check_pdb_ids)
    validation_loader = ProteinLigandDataLoader(*validation_data, check_pdb_ids)

    # train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

    # # Model, criterion, optimizer
    # model = model1.Model().to(args.device)
    # criterion = torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # # Train the model
    # for epoch in range(1, args.epochs + 1):
    #     train(model, train_loader, optimizer, criterion, args.device)
    #     # Save checkpoint
    #     if args.save_model:
    #         torch.save(
    #             model.state_dict(), f"../experiments/checkpoints/model_epoch_{epoch}.pt"
    #         )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training Script")
    parser.add_argument(
        "-d",
        "--data_path",
        type=str,
        default="../data/processed/",
        help="Path to the dataset",
    )
    parser.add_argument(
        "-b", "--batch_size", type=int, default=16, help="Input batch size for training"
    )
    parser.add_argument(
        "-e", "--epochs", type=int, default=100, help="Number of epochs to train"
    )
    parser.add_argument(
        "-lr", "--learning_rate", type=float, default=0.0001, help="Learning rate"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for training",
    )
    parser.add_argument(
        "--save_model", action="store_true", help="For Saving the current Model"
    )

    args = parser.parse_args()
    main(args)
