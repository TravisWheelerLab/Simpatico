import argparse
import json
import pickle
import sys
from datetime import datetime
from glob import glob
from os import path
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.nn import TripletMarginLoss
from torch_geometric.nn import radius

from simpatico.get_train_set import construct_tv_set
from simpatico.models import MolEncoderDefaults, ProteinEncoderDefaults
from simpatico.models.molecule_encoder.MolEncoder import MolEncoder
from simpatico.models.protein_encoder.ProteinEncoder import ProteinEncoder
from simpatico.utils.data_utils import (
    ProteinLigandDataLoader,
    TrainingOutputHandler,
)
from simpatico.utils.utils import get_logger


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

def contrastive_loss(
    p_embeddings, l_embeddings,
    p_coords, l_coords,
    p_batch, l_batch,
    temperature=0.07,
    phys_dist_threshold=6.0,
    structural_weight=1.0,
    hard_negative_ratio=0.1
):
    device = p_embeddings.device
    N = p_embeddings.shape[0]

    # --- 1. Compute Logits & Distances ---
    # Memory: O(N^2) - This is the bottleneck
    logits = (p_embeddings @ l_embeddings.t()) / temperature
    phys_dists = torch.cdist(p_coords, l_coords)

    # --- 2. Create Masks (Boolean, Low Memory) ---
    batch_mask = p_batch.unsqueeze(1) == l_batch.unsqueeze(0)
    proximal_mask = phys_dists < phys_dist_threshold
    eye_mask = torch.eye(N, device=device, dtype=torch.bool)

    # Label for Cross Entropy (0, 1, 2, ... N)
    labels = torch.arange(N, device=device)

    # --- 3. Component 1: Structural Loss ---
    # We apply the mask *directly* to the logits during the function call
    # or via temporary modification to save memory, but cloning is safer for autograd.
    # To save memory, we calculate the mask first.

    # IGNORE: (Different Batch) OR (Proximal neighbors that aren't the diagonal)
    # Using -1e9 instead of -inf for stability
    struct_mask = (~batch_mask) | (proximal_mask & ~eye_mask)

    # We use masked_fill on a clone (necessary for autograd)
    # But we can reuse this clone if we are careful, or just pay the cost.
    logits_struct = logits.masked_fill(struct_mask, -1e9)
    loss_struct = F.cross_entropy(logits_struct, labels)

    # --- 4. Component 2: Hard Batch Loss ---
    # GOAL: Keep Diagonal + Top K Hardest Negatives from OTHER batches.

    # Step A: Identify Hard Negatives
    # We want to mine from 'logits', but we must ignore Same-Batch pairs.
    # We use a temporary view or mask for topk.
    # We DO NOT clone the whole matrix just for mining if we can help it.

    # We want to mask OUT the batch_mask for mining.
    # Instead of cloning, we can use the value -1e9 in a new tensor,
    # or just accept one clone here.
    mining_view = logits.masked_fill(batch_mask, -1e9)
    hard_neg_k = int(logits.size(0) * hard_negative_ratio)

    # Safety check: Ensure k is not larger than available samples
    # If N is small, this prevents index errors
    valid_k = min(hard_neg_k, N - 1)

    if valid_k > 0:
        # Get indices of the hardest negatives
        # largest=True because these are logits (similarity)
        _, hard_indices = torch.topk(mining_view, valid_k, dim=1)

        # Create a boolean mask for these hard negatives
        hard_mask = torch.zeros_like(batch_mask) # Bool tensor
        hard_mask.scatter_(1, hard_indices, True)
    else:
        hard_mask = torch.zeros_like(batch_mask)

    # Step B: Final Batch Logits
    # We want to KEEP: Diagonal OR Hard Negatives
    keep_mask = eye_mask | hard_mask
    # Apply Mask: Everything NOT kept becomes -1e9
    logits_batch = logits.masked_fill(~keep_mask, -1e9)

    loss_batch = F.cross_entropy(logits_batch, labels)

    random_mask = ((~batch_mask) & (~hard_mask)) | eye_mask
    random_logits = logits.masked_fill(~random_mask, -1e9)

    loss_random = F.cross_entropy(random_logits, labels)
    return loss_batch + (structural_weight * loss_struct) + loss_random


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

    p_index, m_index = radius(molecule_batch.pos, protein_out.pos, 4.0, molecule_batch.batch, protein_out.batch)

    loss = contrastive_loss(protein_out.x[p_index],
                            mol_out[m_index],
                            protein_out.pos[p_index],
                            molecule_batch.pos[m_index],
                            protein_out.batch[p_index],
                            molecule_batch.batch[m_index],
                            hard_negative_ratio=difficulty_value
                        )
    return loss, (protein_out.x, protein_out.batch, mol_out, molecule_batch.batch)

# def contrastive_loss(p_embeddings, l_embeddings, temperature=0.07):
#     # p_embeddings and l_embeddings are already normalized from ProjectionHead
#     logits = (p_embeddings @ l_embeddings.t()) / temperature

#     labels = torch.arange(logits.shape[0], device=logits.device)
#     loss_p = torch.nn.functional.cross_entropy(logits, labels)
#     loss_l = torch.nn.functional.cross_entropy(logits.t(), labels)

#     return (loss_p + loss_l) / 2

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

    protein_encoder.eval()
    mol_encoder.eval()

    batch_count = data_loader.size // batch_size

    for batch_idx in range(batch_count):
        with torch.no_grad():
            loss, embed_data = training_step(
                data_loader,
                protein_encoder,
                mol_encoder,
                difficulty_value,
                True,
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
    get_hard_negative_difficulty = hard_negative_scheduler(25, 0.1)
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

        protein_encoder.train()
        mol_encoder.train()

        for batch_idx in range(train_loader.size // BATCH_SIZE):
            # prot_loss = not prot_loss
            loss, _ = training_step(
                train_loader, protein_encoder, mol_encoder, difficulty_value, True
            )

            batch_loss_vals.append(loss)

            if batch_idx % 100 == 0:
                batch_loss_avg = torch.hstack(batch_loss_vals).mean().item()
                log.info(f"Epoch {epoch}, batch {batch_idx} loss: {batch_loss_avg}")
                batch_loss_vals = []
                epoch_loss_vals.append(batch_loss_avg)

            loss.backward()
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

        if epoch % 10 == 0:
            torch.save(
                [protein_encoder.state_dict(), mol_encoder.state_dict()],
                weights_file_template % f'e{epoch}'
            )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train")
    add_arguments(parser)
    args = parser.parse_args()
    args.func(args)
