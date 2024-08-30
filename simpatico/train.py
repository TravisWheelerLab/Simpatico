# scripts/train.py
import sys
import argparse
import torch
from typing import List, Tuple, Optional
from torch_geometric.data import Data, Batch
from torch_geometric.nn import radius
from simpatico.utils.data_utils import (
    ProteinLigandDataLoader,
    TrainingOutputHandler,
)
from simpatico.models.molecule_encoder.MolEncoder import MolEncoder
from simpatico.models.protein_encoder.ProteinEncoder import ProteinEncoder
from simpatico.models import MolEncoderDefaults, ProteinEncoderDefaults
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


def positive_margin_loss(anchors, positives, negatives, m=2.0, d=3):
    positive_distances = torch.norm(anchors - positives, dim=1)
    anchors = anchors.repeat(negatives.size(0) // anchors.size(0), 1)

    negative_distances = torch.norm(anchors - negatives, dim=1)
    positive_loss = torch.clamp(positive_distances - m, min=0)
    negative_loss = torch.clamp(m * d - negative_distances, min=0)

    return positive_loss.mean() + negative_loss.mean()


def main(args):
    device = args.device
    # Load data
    train_data, validation_data = torch.load(args.data_path)

    train_loader = ProteinLigandDataLoader(*train_data, check_pdb_ids)
    validation_loader = ProteinLigandDataLoader(*validation_data, check_pdb_ids)

    protein_encoder = ProteinEncoder(**ProteinEncoderDefaults).to(device)
    mol_encoder = MolEncoder(**MolEncoderDefaults).to(device)

    optimizer = torch.optim.AdamW(
        list(protein_encoder.parameters()) + list(mol_encoder.parameters()),
        lr=args.learning_rate,
    )

    for epoch in range(1, args.epochs + 1):
        for batch_idx in range(train_loader.size // args.batch_size):
            protein_batch, molecule_batch = train_loader.get_random_batch(
                batch_size=args.batch_size
            )
            protein_batch = protein_batch.to(device)
            molecule_batch = molecule_batch.to(device)

            protein_out = protein_encoder(protein_batch)
            mol_out = mol_encoder(molecule_batch)

            output_handler = TrainingOutputHandler(
                *protein_out, mol_out, molecule_batch.pos, molecule_batch.batch
            )

            (
                positive_protein_index,
                positive_mol_index,
                random_negatives,
                self_negatives,
                hard_negatives,
            ) = output_handler.get_all_train_pairs()

            negative_mol_index = torch.hstack(
                # (random_negatives, self_negatives, hard_negatives)
                (random_negatives, self_negatives)
            )
            loss = positive_margin_loss(
                protein_out[0][positive_protein_index],
                mol_out[positive_mol_index],
                mol_out[negative_mol_index],
            )
            print(loss)
            loss.backward()
            optimizer.step()

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
