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


def hard_negative_scheduler(start_difficulty, target_difficulty, target_epoch):
    def scheduler(epoch):
        difficulty_schedule = torch.linspace(start_difficulty, target_difficulty, target_epoch)
        difficulty_idx = min(epoch-1, target_epoch-1)
        return difficulty_schedule[difficulty_idx].item()

    return scheduler

class DynamicRankScheduler:
    def __init__(self, start_ratio=1.0, max_ratio=1.0, min_ratio=0.005, decay_rate=0.99):
        """
        Adaptive scheduler that scales k_hard based on model performance.

        Args:
            start_ratio: Initial hard_negative_ratio (e.g., 0.5 = 50% of external batch)
            min_ratio:   The floor for the ratio (e.g., 0.005 = 0.5%)
            decay_rate:  How fast we allow the ratio to drop (0.9 to 0.999)
        """
        self.current_ratio = start_ratio
        self.max_ratio = max_ratio
        self.min_ratio = min_ratio
        self.decay_rate = decay_rate

    def step(self, relative_rank):
        """
        Adjusts the hard_negative_ratio for the NEXT step based on
        the current batch's mean rank and size M.

        Args:
            mean_rank: The average rank of the positive (0 = top 1, M = last).
            current_M: The size of the external batch for this step.
        """

        # 2. Determine the Ideal Ratio
        # We want the ratio (k/M) to be roughly 2x the relative rank.
        # Why 2x? To ensure the batch is large enough to include the rank
        # plus a buffer of "easier" negatives for stability.
        target_ratio = relative_rank * 2.0

        # 3. Update Logic (Smooth Decay)
        # We only want to decrease the ratio if the model is ready (target < current).
        # We generally don't want to INCREASE ratio unless performance catastrophic collapsed.

        if target_ratio < self.current_ratio:
            # Smoothly decay towards the target
            # New = (0.99 * Old) + (0.01 * Target)
            self.current_ratio = (self.decay_rate * self.current_ratio) + \
                                 ((1 - self.decay_rate) * target_ratio)
        else:
            # If model is struggling (rank spiked), boost ratio immediately to stabilize
            # We move faster upwards (0.9 factor) to rescue training
            self.current_ratio = (0.9 * self.current_ratio) + (0.1 * target_ratio)

        # 4. Clamp
        self.current_ratio = max(self.min_ratio, min(self.max_ratio, self.current_ratio))

        return self.current_ratio

def contrastive_loss(
    p_embeddings, l_embeddings,
    p_coords, l_coords,
    p_batch, l_batch,
    hard_l_embeddings,       # External batch (guaranteed disjoint from p_batch)
    temperature=0.07,
    phys_dist_threshold=5.5,
    hard_negative_ratio=0.1
):
    """
    Computes a hybrid contrastive loss:
    1. Structural: Local Batch (Intra-Molecule)
    2. Random:     Local Batch (Inter-Molecule)
    3. Hard:       External Batch (Inter-Molecule, Top-K mined)
    """
    device = p_embeddings.device
    N = p_embeddings.shape[0]
    M = hard_l_embeddings.shape[0]

    # --- 1. Compute Local Logits & Masks (N x N) ---
    # Used for Structural and Random loss (Local Batch)
    logits_local = (p_embeddings @ l_embeddings.t()) / temperature
    phys_dists = torch.cdist(p_coords, l_coords)

    # Masks
    batch_mask = p_batch.unsqueeze(1) == l_batch.unsqueeze(0)   # Same Molecule
    proximal_mask = phys_dists < phys_dist_threshold            # Too Close
    eye_mask = torch.eye(N, device=device, dtype=torch.bool)    # The Positive

    # Labels for local cross entropy (0, 1, ..., N)
    labels_local = torch.arange(N, device=device)

    # ==============================================================================
    # COMPONENT 1: Structural Loss (Local Batch)
    # Goal: Contrast Atom vs Distant Atoms in SAME molecule
    # ==============================================================================

    # IGNORE: Inter-molecular (Different Batch) OR Proximal Neighbors (Too close)
    # KEEP:   The Diagonal (True Positive) + Distant Intra-molecular atoms
    struct_mask = (~batch_mask) | (proximal_mask & ~eye_mask)

    # Apply Mask (Set ignored to -1e9)
    logits_struct = logits_local.masked_fill(struct_mask, -1e9)
    loss_struct = F.cross_entropy(logits_struct, labels_local)

    # ==============================================================================
    # COMPONENT 2: Random Loss (Local Batch)
    # Goal: Contrast Atom vs Random Atoms in OTHER molecules
    # ==============================================================================

    # IGNORE: Same-Batch atoms (Intra-molecular).
    # KEEP:   The Diagonal (True Positive) + All Inter-molecular atoms (Background noise)
    # (Note: We mask out batch_mask but explicitly keep eye_mask)
    random_mask = batch_mask & (~eye_mask)

    # Apply Mask
    logits_random = logits_local.masked_fill(random_mask, -1e9)
    loss_random = F.cross_entropy(logits_random, labels_local)

    # ==============================================================================
    # COMPONENT 3: Hard Loss (External Batch)
    # Goal: Contrast Atom vs Hardest Atoms in EXTERNAL batch
    # ==============================================================================

    # A. Get the True Positives (from Local Batch)
    # The "Answer Key" is the diagonal of the local logits.
    # Shape: (N, 1)
    pos_logits = torch.diag(logits_local).unsqueeze(1)

    # B. Compute External Logits (N x M)
    # Since hard_l_embeddings has NO overlapping molecules, ALL are valid negatives.
    # We do not need masks here.
    logits_external = (p_embeddings @ hard_l_embeddings.t()) / temperature

    # --- MEMORY SAFE RANK COMPUTATION ---
    # Instead of: rank_counts = (logits_external > pos_logits).sum(dim=1).float()
    # We process columns in chunks to avoid allocating the huge boolean matrix.

    N, M = logits_external.shape
    rank_counts = torch.zeros(N, device=device)

    chunk_size = 1000  # Adjust based on memory (5000 columns at a time)

    with torch.no_grad():
        for start_col in range(0, M, chunk_size):
            end_col = min(start_col + chunk_size, M)

            # Slice the existing logits (No new allocation)
            sub_logits = logits_external[:, start_col:end_col]

            # Compare and sum just this slice
            # This creates a small temporary boolean mask (N x chunk_size)
            chunk_matches = (sub_logits > pos_logits)

            # Accumulate
            rank_counts += chunk_matches.sum(dim=1).float()

            # Delete temp vars explicitly to free graph memory immediately
            del chunk_matches

        mean_rank = rank_counts.mean().item()

    # C. Mine Hard Negatives
    # Calculate K based on the EXTERNAL batch size (M)
    k_hard = int(M * hard_negative_ratio)
    valid_k = max(1, min(k_hard, M))

    # Get Top-K largest logits (hardest negatives)
    # Shape: (N, K)
    hard_neg_logits, _ = torch.topk(logits_external, valid_k, dim=1)

    # D. Concatenate: [Positive, Hard Negatives]
    # Shape: (N, 1 + K)
    logits_hard_mining = torch.cat([pos_logits, hard_neg_logits], dim=1)

    # E. Compute Loss
    # The "correct" class is always index 0 (the first column)
    labels_hard = torch.zeros(N, device=device, dtype=torch.long)
    loss_hard = F.cross_entropy(logits_hard_mining, labels_hard)
    return loss_struct + loss_random + loss_hard, (mean_rank, M, loss_struct.item(), loss_random.item(), loss_hard.item())


def training_step(
    data_loader, protein_encoder, mol_encoder, difficulty_value, prot_loss=True
):
    device = next(protein_encoder.parameters()).device

    protein_batch, molecule_batch = data_loader.get_random_batch()
    protein_batch = protein_batch.clone()
    protein_batch.pos += torch.randn_like(protein_batch.pos) * 0.05

    protein_batch = protein_batch.to(device)
    molecule_batch = molecule_batch.to(device)
    random_ligand_batch = data_loader.get_random_ligand_batch(512, molecule_batch.ligand_id).to(device)

    protein_out = protein_encoder(protein_batch)
    mol_out = mol_encoder(molecule_batch)
    hard_out = mol_encoder(random_ligand_batch)

    p_index, m_index = radius(molecule_batch.pos, protein_out.pos, 4.5, molecule_batch.batch, protein_out.batch)

    loss, loss_metrics = contrastive_loss(protein_out.x[p_index],
                            mol_out[m_index],
                            protein_out.pos[p_index],
                            molecule_batch.pos[m_index],
                            protein_out.batch[p_index],
                            molecule_batch.batch[m_index],
                            hard_out,
                            hard_negative_ratio=difficulty_value
                        )
    return loss, (protein_out.x, protein_out.batch, mol_out, molecule_batch.batch), loss_metrics

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
            loss, embed_data, _ = training_step(
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

    weights_file_template = str(weights_dir / f"{train_handle}_%s.w")
    train_stats = {
        'train_loss': [],
        'validation_loss': [],
        'validation_accuracy': [],
        'mean_rank': []
    }
    epoch_start = 1
    current_ratio = 1
    scheduler = DynamicRankScheduler(start_ratio=current_ratio, min_ratio=0.01)

    if Path(stats_file).exists():
        with open(stats_file, 'rb') as stats_in:
            train_stats = pickle.load(stats_in)

        epoch_start = len(train_stats['train_loss'])+1
        protein_model_weights, mol_model_weights = torch.load(weights_file_template % 'CURRENT')
        protein_encoder.load_state_dict(protein_model_weights)
        mol_encoder.load_state_dict(mol_model_weights)
        current_ratio = train_stats['mean_rank'][-1]*2
        scheduler = DynamicRankScheduler(start_ratio=train_stats['mean_rank'][-1]*2)
    else:
        with open(output_file, 'w') as log_out:
            True

    optimizer = torch.optim.AdamW(
        list(protein_encoder.parameters()) + list(mol_encoder.parameters()),
        lr=train_params['learning_rate'],
    )

    prot_loss = True

    # get_hard_negative_difficulty = hard_negative_scheduler(0.5, 0.01, 20)


    for epoch in range(epoch_start, train_params['epochs'] + 1):
        log.info(f"Epoch {epoch}")

        epoch_loss_vals = []
        epoch_rank_vals = []

        batch_loss_vals = []
        batch_rank_vals = []

        protein_encoder.train()
        mol_encoder.train()

        for batch_idx in range(train_loader.size // BATCH_SIZE):
            # prot_loss = not prot_loss
            loss, _, loss_metrics = training_step(
                train_loader, protein_encoder, mol_encoder, current_ratio, True
            )

            mean_rank, hard_neg_count = loss_metrics[:2]
            current_ratio = scheduler.step(mean_rank / hard_neg_count)

            batch_loss_vals.append(loss)
            batch_rank_vals.append(mean_rank / hard_neg_count)

            if batch_idx % 100 == 0:
                batch_loss_avg = torch.hstack(batch_loss_vals).mean().item()
                batch_rank_avg = torch.tensor(batch_rank_vals).mean().item()
                log.info(f"Epoch {epoch}, batch {batch_idx} loss: {batch_loss_avg}")
                log.info(f"Mean rank: {batch_rank_avg}, Ratio: {current_ratio}")
                batch_loss_vals = []
                batch_rank_vals = []
                epoch_loss_vals.append(batch_loss_avg)
                epoch_rank_vals.append(batch_rank_avg)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            torch.cuda.empty_cache()

        epoch_train_loss = torch.tensor(epoch_loss_vals).mean().item()
        epoch_rank = torch.tensor(epoch_rank_vals).mean().item()
        epoch_validation_loss, epoch_acc = validate(
            validation_loader, protein_encoder, mol_encoder
        )

        log.info(f"Epoch {epoch} validation loss: {epoch_validation_loss}, accuracy: {epoch_acc}")
        for k,v in zip(['train_loss', 'validation_loss', 'validation_accuracy', 'mean_rank'],
                       [epoch_train_loss, epoch_validation_loss, epoch_acc, epoch_rank]):
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
