import sys
from os import path
import pickle
import argparse
import torch
from datetime import datetime
from typing import List, Tuple, Optional
from simpatico.utils.data_utils import (
    ProteinLigandDataLoader,
    TrainingOutputHandler,
)
from simpatico.models.molecule_encoder.MolEncoder import MolEncoder
from simpatico.models.protein_encoder.ProteinEncoder import ProteinEncoder
from simpatico.get_train_set import construct_tv_set
from simpatico.models import MolEncoderDefaults, ProteinEncoderDefaults
from typing import Callable
from torch.nn import TripletMarginLoss
import torch.nn.functional as F



def add_arguments(parser):
    parser.add_argument(
        "input",
        type=str,
        help="Path to train-eval dataset",
    )
    parser.add_argument("weight_path"),
    parser.add_argument("-o", "--output", type=str, help="Model performance output")
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
        "-l",
        "--load_model",
        help="Path to previously trained weights",
    )
    parser.add_argument("--epoch_start", type=int, default=1)

    parser.set_defaults(main=main)
    return parser

triplet_loss = TripletMarginLoss(margin=1.0, p=2, eps=1e-7)
bce_loss = torch.nn.BCEWithLogitsLoss()

def positive_margin_loss(anchors, positives, negatives, m=1.0, d=3):
    positive_distances = torch.norm(anchors - positives, dim=1)
    anchors = anchors.repeat(negatives.size(0) // anchors.size(0), 1)

    negative_distances = torch.norm(anchors - negatives, dim=1)
    positive_loss = torch.clamp(positive_distances - m, min=0)
    negative_loss = torch.clamp(m * d - negative_distances, min=0)

    return positive_loss.mean() + negative_loss.mean()


def hard_negative_scheduler(target_epoch, target_difficulty):
    def scheduler(epoch):
        d_modifier = min(epoch / target_epoch, 1)
        return 1 - (1 - target_difficulty) * d_modifier

    return scheduler


def logger(output_path):
    def log_text(message):
        if output_path is not None:
            with open(output_path, "a") as f_out:
                f_out.write(f"{message}\n")
        else:
            print(message)

    return log_text


def training_step(
    data_loader, protein_encoder, mol_encoder, difficulty_value, prot_loss=True
):
    device = next(protein_encoder.parameters()).device

    protein_batch, molecule_batch = data_loader.get_random_batch()
    protein_batch = protein_batch.clone()
    protein_batch.pos += torch.randn_like(protein_batch.pos) * 0.25

    protein_batch = protein_batch.to(device)
    molecule_batch = molecule_batch.to(device)

    protein_out = protein_encoder(protein_batch)
    mol_out = mol_encoder(molecule_batch)

    output_handler = TrainingOutputHandler(
        protein_out.x,
        protein_out.pos,
        protein_out.batch,
        mol_out,
        molecule_batch.pos,
        molecule_batch.batch,
    )

    anchor_samples, positive_samples, negative_samples = (
        output_handler.get_anchors_positives_negatives(
            prot_anchor=prot_loss, difficulty=difficulty_value
        )
    )

    anchor_samples = anchor_samples.repeat(negative_samples.size(0) // anchor_samples.size(0), 1)
    positive_samples = positive_samples.repeat(negative_samples.size(0) // positive_samples.size(0), 1)

    loss = positive_margin_loss(anchor_samples, positive_samples, negative_samples)
    return loss, (protein_out.x, protein_out.batch, mol_out, molecule_batch.batch)


def diag_ranks(D):
    """
    D: [V, V] tensor
    Returns: [V] tensor, where out[v] is the rank position of D[v, v]
             in row v when sorted ascending.
    """
    V = D.shape[0]

    # argsort each row
    sorted_idx = D.argsort(dim=1)  # [V, V]

    # create a mask where sorted_idx[row] == row
    row_idx = torch.arange(V, device=D.device)
    mask = (sorted_idx == row_idx[:, None])  # [V, V], True at the diagonal element’s position

    # get the column index where True
    ranks = mask.nonzero(as_tuple=False)[:, 1]  # [V]

    return ranks


def batch_avg_cdist(A, B, A_batch, B_batch, V=None):
    """
    Compute [V, V] matrix where entry (v_a, v_b) is the average
    pairwise distance between all items of A in batch v_a and
    all items of B in batch v_b.
    """
    V = max(A_batch.max(), B_batch.max()).item() + 1

    # pairwise distances
    D = torch.cdist(A, B)  # [M, N]

    # one-hot encodings of batch membership
    A_onehot = F.one_hot(A_batch, V).to(D.dtype)  # [M, V]
    B_onehot = F.one_hot(B_batch, V).to(D.dtype)  # [N, V]

    # total distances per (v_a, v_b)
    # (V,V) = (A^T @ D @ B)
    sums = A_onehot.T @ D @ B_onehot

    # counts per (v_a, v_b)
    counts = (A_onehot.sum(0)[:, None] * B_onehot.sum(0)[None, :])

    return sums / counts



class ScreenTest:
    def __init__(self):
        self.prot_embeds = []
        self.prot_batch = []
        self.mol_embeds = []
        self.mol_batch = []

    def add(self, prot_embeds, prot_batch, mol_embeds, mol_batch):
        self.prot_embeds.append(prot_embeds)
        self.mol_embeds.append(mol_embeds)

        if len(self.prot_batch) == 0:
            prot_batch_mod = 0
            mol_batch_mod = 0
        else:
            prot_batch_mod = self.prot_batch[-1][-1] + 1
            mol_batch_mod = self.mol_batch[-1][-1] + 1
        
        prot_batch += prot_batch_mod
        mol_batch += mol_batch_mod

        self.prot_batch.append(prot_batch)
        self.mol_batch.append(mol_batch)
    
    def run(self):
        prot_embeds = torch.vstack(self.prot_embeds)
        prot_batch = torch.hstack(self.prot_batch)
        mol_embeds = torch.vstack(self.mol_embeds)
        mol_batch = torch.hstack(self.mol_batch)

        avg_distances = batch_avg_cdist(prot_embeds, mol_embeds, prot_batch, mol_batch)
        rankings = diag_ranks(avg_distances)
        acc = 1 - (rankings.float().mean() / len(rankings))
        return acc.item()

def validate(
    data_loader, protein_encoder, mol_encoder, difficulty_value=1, batch_size=16 
):
    validation_loss_vals = []
    screen_test = ScreenTest()

    batch_count = data_loader.size // batch_size

    for prot_loss in [True, False]:
        for batch_idx in range(batch_count):
            with torch.no_grad():
                loss, embed_data = training_step(
                    data_loader,
                    protein_encoder,
                    mol_encoder,
                    difficulty_value,
                    prot_loss,
                )
                screen_test.add(*embed_data)
                validation_loss_vals.append(loss)

    epoch_acc = screen_test.run()
    return sum(validation_loss_vals) / len(validation_loss_vals), epoch_acc


def main(args):
    device = args.device
    log_text = logger(args.output)
    # Load data

    _, input_filetype = path.splitext(args.input)

    if input_filetype in [".pkl", '.tv']:
        with open(args.input, "rb") as train_validate_data:
            train_data, validation_samples = pickle.load(train_validate_data)

    elif input_filetype == ".csv":
        train_data, validation_samples = construct_tv_set(args.input)

    validation_data = []

    g = torch.Generator()
    g.manual_seed(1234)

    # reproducible randperm, only for this call
    for random_idx in torch.randperm(len(validation_samples), generator=g)[:200]:
        validation_data.append(validation_samples[random_idx])


    if args.output is not None:
        with open(args.output, "w"):
            True

    train_loader = ProteinLigandDataLoader(train_data, batch_size=args.batch_size)
    validation_loader = ProteinLigandDataLoader(
        validation_data, batch_size=args.batch_size
    )

    protein_encoder = ProteinEncoder(**ProteinEncoderDefaults).to(device)
    mol_encoder = MolEncoder(**MolEncoderDefaults).to(device)
    
    # Difficulty ratio value arrived at by observing that 0.05 works well for a batch size of 16.
    difficulty_ratio = (0.05 * 16) / args.batch_size
    get_hard_negative_difficulty = hard_negative_scheduler(50, difficulty_ratio)

    if args.load_model:
        protein_model_weights, mol_model_weights = torch.load(args.load_model)

        protein_encoder.load_state_dict(protein_model_weights)
        mol_encoder.load_state_dict(mol_model_weights)

        difficulty_value = get_hard_negative_difficulty(args.epoch_start)

        initial_validation_loss, initial_acc = validate(
            validation_loader, protein_encoder, mol_encoder
        )
        log_text(f"Best validation loss: {initial_validation_loss}, accuracy: {initial_acc}")

    optimizer = torch.optim.AdamW(
        list(protein_encoder.parameters()) + list(mol_encoder.parameters()),
        lr=args.learning_rate,
    )

    prot_loss = True
    best_validation_loss = None
    best_accuracy = None

    for epoch in range(args.epoch_start, args.epochs + 1):
        log_text(f'Epoch {epoch} - {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        difficulty_value = get_hard_negative_difficulty(epoch)

        if best_validation_loss is None:
            best_validation_loss, best_accuracy = validate(
                validation_loader, protein_encoder, mol_encoder
            )

        log_message = f"Epoch {epoch} difficulty: {difficulty_value}"
        log_text(log_message)
        loss_vals = []

        for batch_idx in range(train_loader.size // args.batch_size):
            prot_loss = not prot_loss
            loss, _ = training_step(
                train_loader, protein_encoder, mol_encoder, difficulty_value, prot_loss
            )

            loss_vals.append(loss)

            if batch_idx % 10 == 0:
                loss_avg = torch.hstack(loss_vals).mean().item()
                log_text(f"Epoch {epoch}, batch {batch_idx} loss: {loss_avg}")
                loss_vals = []

            loss.backward()

            if prot_loss:
                optimizer.step()
                optimizer.zero_grad()
                torch.cuda.empty_cache()

        epoch_validation_loss, epoch_acc = validate(
            validation_loader, protein_encoder, mol_encoder
        )

        log_text(f"Epoch {epoch} validation loss: {epoch_validation_loss}, accuracy: {epoch_acc}")

        if epoch % 50 == 0:
            split_weight_path = args.weight_path.split('.')
            split_weight_path[-2] += f'_{epoch}'
            current_weight_path = '.'.join(split_weight_path)

            torch.save(
                [protein_encoder.state_dict(), mol_encoder.state_dict()],
                current_weight_path
            )


        if epoch_acc > best_accuracy:
            best_accuracy = epoch_acc

            torch.save(
                [protein_encoder.state_dict(), mol_encoder.state_dict()],
                args.weight_path,
            )

            log_text(f"Weights updated")
        else:
            split_weight_path = args.weight_path.split('.')
            split_weight_path[-2] += '_CURRENT'
            current_weight_path = '.'.join(split_weight_path)

            torch.save(
                [protein_encoder.state_dict(), mol_encoder.state_dict()],
                current_weight_path
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train")
    add_arguments(parser)
    args = parser.parse_args()
    args.func(args)
