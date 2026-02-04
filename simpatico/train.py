import sys
from glob import glob
from pathlib import Path
from os import path
import pickle
import argparse
import torch
from datetime import datetime
from typing import List, Tuple, Optional
from simpatico.utils.utils import get_logger
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
import json

def add_arguments(parser):
    parser.add_argument(
        "input",
        type=str,
        help="Path to train-eval dataset",
    )
    parser.set_defaults(main=main)
    return parser

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
                validation_loss_vals.append(loss.item())

    epoch_acc = screen_test.run()
    return sum(validation_loss_vals) / len(validation_loss_vals), epoch_acc

def get_tv_sets(data, holdout_file):
    with open(holdout_file) as f_in:
        holdout_substrings = [x.strip() for x in f_in.readlines()]

    holdout_index = []
    for hs in holdout_substrings:
        for i in range(len(data)):
            if hs in data[i][0].name:
                holdout_index.append(i)

    train_data = []
    validation_data = []

    for idx in range(len(data)):
        if idx in holdout_index:
            validation_data.append(data[idx])
        else:
            train_data.append(data[idx])

    return train_data, validation_data


def main(args):
    with open(args.input) as json_f:
        train_params = json.load(json_f)

    train_handle = train_params['train_handle']
    weights_dir = Path(f"{train_params['output_dir']}/weights")
    weights_dir.mkdir(exist_ok=True)

    BATCH_SIZE = train_params['batch_size']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_file = f"{train_params['output_dir']}/{train_handle}.o"
    stats_file = f"{train_params['output_dir']}/{train_handle}_stats.pkl"
    log = get_logger(output_file)

    with open(train_params['data_file'], "rb") as train_validate_data:
        data_corpus = pickle.load(train_validate_data)

    train_data, validation_samples = get_tv_sets(data_corpus, train_params['holdout_file'])
    validation_data = []

    g = torch.Generator()
    g.manual_seed(1234)

    for random_idx in torch.randperm(len(validation_samples), generator=g)[:200]:
        validation_data.append(validation_samples[random_idx])

    train_loader = ProteinLigandDataLoader(train_data, batch_size=BATCH_SIZE)
    validation_loader = ProteinLigandDataLoader(
        validation_data, batch_size=BATCH_SIZE
    )

    protein_encoder = ProteinEncoder().to(device)
    mol_encoder = MolEncoder().to(device)

    # Difficulty ratio value arrived at by observing that 0.05 works well for a batch size of 16.
    difficulty_ratio = (0.05 * 16) / BATCH_SIZE
    get_hard_negative_difficulty = hard_negative_scheduler(50, difficulty_ratio)
    weights_file_template = str(weights_dir / f"{train_handle}_%s.w")
    train_stats = {
        'train_loss': [],
        'validation_loss': [],
        'validation_accuracy': []
    }
    epoch_start = 1

    if Path(stats_file).exists():
        with open(stats_file, 'rb') as stats_in:
            train_stats = pickle.load(stats_in)

        epoch_start = len(train_stats['train_loss'])+1
        protein_model_weights, mol_model_weights = torch.load(weights_file_template % 'CURRENT')
        protein_encoder.load_state_dict(protein_model_weights)
        mol_encoder.load_state_dict(mol_model_weights)
    else:
        with open(output_file, 'w') as log_out:
            True

    difficulty_value = get_hard_negative_difficulty(epoch_start)

    optimizer = torch.optim.AdamW(
        list(protein_encoder.parameters()) + list(mol_encoder.parameters()),
        lr=train_params['learning_rate'],
    )

    prot_loss = True

    for epoch in range(epoch_start, train_params['epochs'] + 1):
        difficulty_value = get_hard_negative_difficulty(epoch)
        log.info(f"Epoch {epoch} |--| difficulty: {difficulty_value}")

        epoch_loss_vals = []
        batch_loss_vals = []

        for batch_idx in range(train_loader.size // BATCH_SIZE):
            prot_loss = not prot_loss
            loss, _ = training_step(
                train_loader, protein_encoder, mol_encoder, difficulty_value, prot_loss
            )

            batch_loss_vals.append(loss)

            if batch_idx % 100 == 0:
                batch_loss_avg = torch.hstack(batch_loss_vals).mean().item()
                log.info(f"Epoch {epoch}, batch {batch_idx} loss: {batch_loss_avg}")
                batch_loss_vals = []
                epoch_loss_vals.append(batch_loss_avg)

            loss.backward()

            if prot_loss:
                optimizer.step()
                optimizer.zero_grad()
                torch.cuda.empty_cache()

        epoch_train_loss = torch.tensor(epoch_loss_vals).mean().item()
        epoch_validation_loss, epoch_acc = validate(
            validation_loader, protein_encoder, mol_encoder
        )

        log.info(f"Epoch {epoch} validation loss: {epoch_validation_loss}, accuracy: {epoch_acc}")
        for k,v in zip(['train_loss', 'validation_loss', 'validation_accuracy'],
                       [epoch_train_loss, epoch_validation_loss, epoch_acc]):
            train_stats[k].append(v)

        with open(stats_file, 'wb') as stats_out:
            pickle.dump(train_stats, stats_out)

        torch.save(
            [protein_encoder.state_dict(), mol_encoder.state_dict()],
            weights_file_template % "CURRENT"
        )

        if epoch % 50 == 0:
            torch.save(
                [protein_encoder.state_dict(), mol_encoder.state_dict()],
                weights_file_template % f'e{epoch}'
            )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train")
    add_arguments(parser)
    args = parser.parse_args()
    args.func(args)
