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
import torch.multiprocessing as mp
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


class HardBatchScheduler:
    def __init__(
        self,
        start_size=32,
        max_size=512,
        growth_factor=2,
        patience=100,
        target_metric=0.15,
    ):
        """
        Args:
            start_size: Initial HARD_BATCH_SIZE (e.g., 32 molecules).
            max_size: Maximum HARD_BATCH_SIZE (e.g., 512 molecules).
            growth_factor: Multiplier for size increase (e.g., 2 -> doubles size).
            patience: Number of batches with good metrics required to level up.
            target_metric: The mean_rank value below which we consider the task 'solved'
                           (0.0 = perfect top-1 rank, 1.0 = worst rank).
        """
        self.current_size = min(start_size, max_size)
        self.max_size = max_size
        self.growth_factor = growth_factor
        self.patience = patience
        self.target_metric = target_metric
        self.win_streak = 0

    def step(self, mean_rank_metric):
        """
        Updates the schedule based on the latest loss metric.
        Returns the new HARD_BATCH_SIZE.
        """
        # Check if model is performing well (Rank is low/good)
        if mean_rank_metric < self.target_metric:
            self.win_streak += 1
        else:
            self.win_streak = 0  # Reset if performance drops

        # Level Up Mechanism
        if self.win_streak >= self.patience:
            old_size = self.current_size
            self.current_size = int(self.current_size * self.growth_factor)

            # Cap at max
            self.current_size = min(self.current_size, self.max_size)

            if self.current_size > old_size:
                print(
                    f"\n[Curriculum] Level Up! Increased Hard Batch Size: {old_size} -> {self.current_size}"
                )
                self.win_streak = 0  # Reset streak for the new difficulty level

        # return self.current_size


def contrastive_loss(
    prot_x,
    prot_pos,
    prot_batch,
    lig_x,
    lig_pos,
    lig_batch,
    hard_l_embeddings,
    prot_anchor=True,
    temperature=0.2,
    interaction_radius=4.0,
    phys_dist_threshold=6.0,
    window_size=100,
    max_considered_rank=2048,
):
    p_index, m_index = radius(
        lig_pos,
        prot_pos,
        interaction_radius,
        lig_batch,
        prot_batch,
    )
    p_embeddings, p_coords, p_batch = (
        prot_x[p_index],
        prot_pos[p_index],
        prot_batch[p_index],
    )
    l_embeddings, l_coords, l_batch = (
        lig_x[m_index],
        lig_pos[m_index],
        lig_batch[m_index],
    )

    if prot_anchor:
        a_embeddings, a_coords, a_batch = p_embeddings, p_coords, p_batch
        s_embeddings, s_coords, s_batch = l_embeddings, l_coords, l_batch
    else:
        a_embeddings, a_coords, a_batch = l_embeddings, l_coords, l_batch
        s_embeddings, s_coords, s_batch = p_embeddings, p_coords, p_batch

    device = prot_x.device
    N = a_embeddings.shape[0]
    M = hard_l_embeddings.shape[0]  # Total Atoms in External Batch

    # --- DEFENSE 1: NaN Checks ---
    if torch.isnan(p_embeddings).any():
        raise ValueError("Critical: NaNs detected in p_embeddings!")

    # --- DEFENSE 2: Force Float32 ---
    with torch.cuda.amp.autocast(enabled=False):
        a_f32 = a_embeddings.float()
        s_f32 = s_embeddings.float()
        hard_l_f32 = hard_l_embeddings.float()

        # Normalize
        # p_f32 = F.normalize(p_f32, p=2, dim=1)
        # l_f32 = F.normalize(l_f32, p=2, dim=1)
        # hard_l_f32 = F.normalize(hard_l_f32, p=2, dim=1)

        # 1. Compute Local Logits (FP32)
        logits_local = (a_f32 @ s_f32.t()) / temperature
        logits_external = (a_f32 @ hard_l_f32.t()) / temperature

    # --- Masks ---
    phys_dists = torch.cdist(a_coords, s_coords)
    batch_mask = a_batch.unsqueeze(1) == s_batch.unsqueeze(0)
    proximal_mask = phys_dists < phys_dist_threshold
    eye_mask = torch.eye(N, device=device, dtype=torch.bool)
    labels_local = torch.arange(N, device=device)

    # ==============================================================================
    # COMPONENT 1: Structural Loss (Intra-Molecule)
    # ==============================================================================
    struct_mask = (~batch_mask) | (proximal_mask & ~eye_mask)
    logits_struct = logits_local.masked_fill(struct_mask, -1e9)
    loss_struct = F.cross_entropy(logits_struct, labels_local)

    # ==============================================================================
    # COMPONENT 2: Balanced Inter-Molecular Loss (Hard Window + Random Buffer)
    # ==============================================================================

    # A. Positive Scores (Diagonal of Local)
    pos_logits = torch.diag(logits_local).unsqueeze(1)

    # B. Hard Mining (Rank Window)
    # Search Horizon: limited to 2048 atoms or total size M
    limit_k = min(max_considered_rank, M)
    sorted_logits, _ = torch.topk(logits_external, k=limit_k, dim=1)

    # Calculate Rank relative to Search Horizon
    pos_expanded = pos_logits.expand(-1, limit_k)
    rank_in_topk = (sorted_logits > pos_expanded).sum(dim=1)
    # Calculate Relative Rank as specified
    relative_rank = rank_in_topk.float() / limit_k

    # Calculate Linear Window Start Index
    # Maps [0, 1] to [0, limit_k - window_size]
    start_indices = (relative_rank * (limit_k - window_size)).long()

    # Generate the window indices for each row
    offsets = torch.arange(window_size, device=device)
    gather_indices = (start_indices.unsqueeze(1) + offsets.unsqueeze(0)).clamp(
        min=0, max=limit_k - 1
    )

    hard_window_logits = torch.gather(sorted_logits, 1, gather_indices)

    # D. Random Buffer (The Stabilizer)
    # We sample exactly as many randoms as we have hard window samples to maintain 1:1 balance
    # rand_indices = torch.randint(0, M, (N, hard_window_logits.size(1)), device=device)
    # random_ext_logits = torch.gather(logits_external, 1, rand_indices)

    # Concatenate: [Positive, Hard_Window, Random_Buffer]
    logits_final = torch.cat([pos_logits, hard_window_logits], dim=1)

    # E. Compute Loss
    labels_hard = torch.zeros(N, device=device, dtype=torch.long)
    loss_external = F.cross_entropy(logits_final, labels_hard)

    # Metric: Normalized by limit_k (Search Horizon) to prevent "Success Illusion"
    # 0.0 = Best possible rank
    # 1.0 = Positive fell out of the search window (Curriculum Failure)
    mean_rank = rank_in_topk.float().mean().item() / limit_k

    return loss_struct + loss_external, (
        mean_rank,
        loss_struct.item(),
        loss_external.item(),
    )


def training_step(data_loader, protein_encoder, mol_encoder, hard_batch_scheduler):
    device = next(protein_encoder.parameters()).device

    protein_batch, molecule_batch = data_loader.get_random_batch()
    protein_batch = protein_batch.clone()
    protein_batch.pos += torch.randn_like(protein_batch.pos) * 0.25

    protein_batch = protein_batch.to(device)
    molecule_batch = molecule_batch.to(device)
    random_ligand_batch = data_loader.get_random_ligand_batch(
        hard_batch_scheduler.current_size, molecule_batch.ligand_id
    ).to(device)

    protein_out = protein_encoder(protein_batch, molecule_batch)
    mol_out = mol_encoder(molecule_batch)
    hard_out = mol_encoder(random_ligand_batch)

    total_loss = None
    pm_metrics = []

    for prot_anchor in [True, False]:
        loss, loss_metrics = contrastive_loss(
            protein_out.x,
            protein_out.pos,
            protein_out.batch,
            mol_out,
            molecule_batch.pos,
            molecule_batch.batch,
            hard_out,
            prot_anchor,
        )

        if total_loss is None:
            total_loss = loss
        else:
            total_loss = total_loss + loss

        pm_metrics.append(loss_metrics)

    return (
        total_loss,
        (protein_out.x, protein_out.batch, mol_out, molecule_batch.batch),
        pm_metrics,
    )


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
    mask = (
        sorted_idx == row_idx[:, None]
    )  # [V, V], True at the diagonal element’s position

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
    counts = A_onehot.sum(0)[:, None] * B_onehot.sum(0)[None, :]

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
    data_loader,
    protein_encoder,
    mol_encoder,
    hard_batch_scheduler,
    difficulty_value=1,
    batch_size=16,
):
    validation_loss_vals = []
    screen_test = ScreenTest()

    protein_encoder.eval()
    mol_encoder.eval()

    batch_count = max(data_loader.size // batch_size, 1)

    for batch_idx in range(batch_count):
        with torch.no_grad():
            loss, embed_data, _ = training_step(
                data_loader, protein_encoder, mol_encoder, hard_batch_scheduler
            )
            screen_test.add(*embed_data)
            validation_loss_vals.append(loss.item())

    epoch_acc = screen_test.run()
    return sum(validation_loss_vals) / len(validation_loss_vals), epoch_acc


def get_tv_sets(data, validation_file, holdout_file=None):
    with open(validation_file) as f_in:
        validation_substrings = [x.strip() for x in f_in.readlines()]

    holdout_substrings = []
    if holdout_file:
        with open(holdout_file) as f_in:
            holdout_substrings = [x.strip() for x in f_in.readlines()]

    validation_index = []
    holdout_index = []

    for hs in validation_substrings:
        for i in range(len(data)):
            if hs in data[i][0].name:
                validation_index.append(i)

    for hs in holdout_substrings:
        for i in range(len(data)):
            if hs in data[i][0].name:
                holdout_index.append(i)

    train_data = []
    validation_data = []

    for idx in range(len(data)):
        if idx in holdout_index:
            continue
        elif idx in validation_index:
            validation_data.append(data[idx])
        else:
            train_data.append(data[idx])

    return train_data, validation_data


def main(args):
    with open(args.input) as json_f:
        train_params = json.load(json_f)

    train_handle = train_params["train_handle"]
    weights_dir = Path(f"{train_params['output_dir']}/weights")
    weights_dir.mkdir(exist_ok=True)
    weight_checkpoint_interval = int(train_params['weight_checkpoint_interval'])

    BATCH_SIZE = train_params["batch_size"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_file = f"{train_params['output_dir']}/{train_handle}.o"
    stats_file = f"{train_params['output_dir']}/{train_handle}_stats.pkl"
    log = get_logger(output_file)

    with open(train_params["data_file"], "rb") as train_validate_data:
        data_corpus = pickle.load(train_validate_data)

    train_data, validation_data = get_tv_sets(
        data_corpus, train_params['validation_file'], train_params["holdout_file"]
    )

    train_loader = ProteinLigandDataLoader(train_data, batch_size=BATCH_SIZE)
    validation_loader = ProteinLigandDataLoader(validation_data, batch_size=BATCH_SIZE)

    protein_encoder = ProteinEncoder().to(device)
    mol_encoder = MolEncoder().to(device)

    weights_file_template = str(weights_dir / f"{train_handle}_%s.w")
    train_stats = {
        "train_loss": [],
        "validation_loss": [],
        "validation_accuracy": [],
        "mean_rank": [],
        "hn_batch_size": [],
    }
    epoch_start = 1
    hn_batch_size = 32
    if Path(stats_file).exists():
        with open(stats_file, "rb") as stats_in:
            train_stats = pickle.load(stats_in)

        epoch_start = len(train_stats["train_loss"]) + 1
        protein_model_weights, mol_model_weights = torch.load(
            weights_file_template % "CURRENT"
        )
        protein_encoder.load_state_dict(protein_model_weights)
        mol_encoder.load_state_dict(mol_model_weights)
        hn_batch_size = train_stats["hn_batch_size"][-1]
    else:
        with open(output_file, "w") as log_out:
            True

    # --- INITIALIZATION ---
    # Start small (32) to let the model learn basic atomic identity.
    batch_scheduler = HardBatchScheduler(
        start_size=hn_batch_size,
        max_size=162,
        growth_factor=1.5,  # Increase by 50% each time
        patience=10,  # Require 50 stable batches before increasing
        target_metric=0.10,  # Target: Positive is in the top 10% of candidates
    )

    optimizer = torch.optim.AdamW(
        list(protein_encoder.parameters()) + list(mol_encoder.parameters()),
        lr=train_params["learning_rate"],
    )

    for epoch in range(epoch_start, train_params["epochs"] + 1):
        log.info(f"Epoch {epoch}")

        epoch_loss_vals = []
        epoch_rank_vals = []

        batch_loss_vals = []
        batch_rank_vals = []

        protein_encoder.train()
        mol_encoder.train()

        for batch_idx in range(train_loader.size // BATCH_SIZE):
            loss, _, loss_metrics = training_step(
                train_loader, protein_encoder, mol_encoder, batch_scheduler
            )

            mean_rank = loss_metrics[0][0]
            batch_scheduler.step(mean_rank)

            batch_loss_vals.append(loss)
            batch_rank_vals.append(mean_rank)

            if batch_idx % 100 == 0:
                batch_loss_avg = torch.hstack(batch_loss_vals).mean().item()
                batch_rank_avg = torch.tensor(batch_rank_vals).mean().item()
                log_string = (
                    "Epoch %s, batch %s loss: %s, Mean rank: %s, HN batch size: %s"
                )
                log.info(
                    log_string
                    % (
                        epoch,
                        batch_idx,
                        round(batch_loss_avg, 3),
                        round(batch_rank_avg, 3),
                        batch_scheduler.current_size,
                    )
                )

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
        epoch_loss_vals = []
        epoch_rank_vals = []

        epoch_validation_loss, epoch_acc = validate(
            validation_loader, protein_encoder, mol_encoder, batch_scheduler
        )

        log.info(
            f"Epoch {epoch} validation loss: {epoch_validation_loss}, accuracy: {epoch_acc}"
        )
        for k, v in zip(
            [
                "train_loss",
                "validation_loss",
                "validation_accuracy",
                "mean_rank",
                "hn_batch_size",
            ],
            [
                epoch_train_loss,
                epoch_validation_loss,
                epoch_acc,
                epoch_rank,
                batch_scheduler.current_size,
            ],
        ):
            train_stats[k].append(v)

        with open(stats_file, "wb") as stats_out:
            pickle.dump(train_stats, stats_out)

        torch.save(
            [protein_encoder.state_dict(), mol_encoder.state_dict()],
            weights_file_template % "CURRENT",
        )

        if epoch % weight_checkpoint_interval == 0:
            torch.save(
                [protein_encoder.state_dict(), mol_encoder.state_dict()],
                weights_file_template % f"e{epoch}",
            )


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass  # Context already set, which is fine
    parser = argparse.ArgumentParser(description="Train")
    add_arguments(parser)
    args = parser.parse_args()
    args.func(args)
