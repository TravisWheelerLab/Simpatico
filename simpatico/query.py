import os
import faiss
import numpy as np
import pickle
from os import path
from pathlib import Path
import sys
import argparse
import torch
from typing import List, Tuple, Optional
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import radius
from simpatico.utils.mol_utils import molfile2pyg, get_xyz_from_file

from simpatico.utils.data_utils import (
    ProteinLigandDataLoader,
    TrainingOutputHandler,
    report_results,
)
from simpatico.models.molecule_encoder.MolEncoder import MolEncoder
from simpatico.models.protein_encoder.ProteinEncoder import ProteinEncoder
from simpatico.models import MolEncoderDefaults, ProteinEncoderDefaults
from simpatico.utils.pdb_utils import pdb2pyg
from simpatico.utils.utils import SmartFormatter

from typing import Callable
from glob import glob


def add_arguments(parser):
    parser.add_argument("input_file", type=str, help="R|Path to input file.\n")
    parser.add_argument("output_file", type=str, help="File for results")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for training",
    )
    parser.add_argument("-o", "--print-output", action="store_true")
    parser.set_defaults(main=main)
    return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query tool")
    add_arguments(parser)
    args = parser.parse_args()
    args.func(args)


class VectorDatabase:
    def __init__(self):
        self.sources = []
        self.vectors = None
        self.item_batch = None
        self.file_batch = None
        self.score_thresholds = None
        self.source_index = None

    def initialize(self, embed_files, score_thresholds=None):
        vectors = []
        item_batch = []
        file_batch = []
        source_index = []

        for ef_i, embed_file in enumerate(embed_files):
            g = torch.load(embed_file, weights_only=False)

            self.sources.append(g.source)
            vectors.append(g.x)

            batch_modifier = item_batch[-1][-1] + 1 if len(item_batch) else 0

            item_batch.append(g.batch + batch_modifier)
            file_batch.append(torch.zeros(len(g.x), dtype=torch.long).fill_(ef_i))

            if hasattr(g, "source_idx") and g.source_idx is not None:
                source_index.append(g.source_idx)
            else:
                source_index.append(torch.zeros(len(g.x), dtype=torch.long))

        self.vectors = torch.vstack(vectors)
        self.item_batch = torch.hstack(item_batch)
        self.file_batch = torch.hstack(file_batch)

        if len(source_index):
            self.source_index = torch.hstack(source_index)

        if score_thresholds:
            self.score_thresholds = score_thresholds

    def get_score_thresholds(
        self, vector_db, n_random: int = 4000, q: float = 0.99, n_trials=5
    ):
        # pick sample size that works for both arrays
        n1, _ = self.vectors.shape
        n2, _ = vector_db.vectors.shape

        k1 = min(n_random, n1)
        k2 = min(n_random, n2)

        item_max_D = torch.zeros(self.item_batch[-1] + 1)
        item_thresholds = torch.zeros_like(item_max_D)

        for trial_n in range(n_trials):
            idx2 = torch.randperm(len(vector_db.vectors))[:n_random]
            rv_2 = vector_db.vectors[idx2]

            for item_index in torch.arange(len(item_max_D)):
                item_vectors = self.vectors[self.item_batch == item_index]

                D = torch.cdist(item_vectors, rv_2).flatten()
                max_D = D.max()
                S = max_D - D
                threshold = torch.quantile(S, q)

                item_max_D[item_index] += max_D / n_trials
                item_thresholds[item_index] += threshold / n_trials

        self.score_thresholds = (item_max_D, item_thresholds)

    def save_metadata(self, path):
        metadata = {"sources": self.sources, "score_thresholds": self.score_thresholds}
        with open(path, "wb") as metadata_out:
            pickle.dump(metadata, metadata_out)

    def load(self, path):
        with open(path, "rb") as metadata_in:
            metadata = pickle.load(metadata_in)

        self.initialize(metadata["sources"], metadata["score_thresholds"])

    def faiss_index(self, gpu=True):
        np_vectors = self.vectors.detach().cpu().numpy().astype("float32")

        dim = np_vectors.shape[1]
        index = faiss.IndexFlatL2(dim)

        if gpu:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)

        index.add(np_vectors)
        return index

    def np_vectors(self):
        return self.vectors.detach().cpu().numpy().astype("float32")

    def get_scores(self, D, I, query_db):
        D = torch.as_tensor(D)
        I = torch.as_tensor(I)

        score_thresholds = query_db.score_thresholds
        target_batch = query_db.item_batch

        max_D = score_thresholds[0][target_batch].unsqueeze(1)
        threshold_S = score_thresholds[1][target_batch].unsqueeze(1)

        S = torch.clamp(max_D.unsqueeze(1) - D - threshold_S.unsqueeze(1), min=0)
        mol_I = self.item_batch[I].long()

        target_mol_scores = []

        for t_i in target_batch.unique():
            target_mask = target_batch == t_i
            S_t = S[target_mask].flatten()
            I_t = I[target_mask].flatten()
            mol_I = self.item_batch[I_t].long()

            mol_scores = torch.zeros_like(self.item_batch.unique()).float()
            mol_scores.scatter_add_(0, mol_I, S_t)
            target_mol_scores.append(mol_scores)

        target_mol_scores = torch.vstack((target_mol_scores))

        return target_mol_scores

    def search(self, query_db):
        queries = query_db.np_vectors()
        faiss_index = self.faiss_index()

        D, I = faiss_index.search(queries, 2048)

        scores = self.get_scores(D, I, query_db)
        sorted_scores, sorted_index = scores.sort(descending=True, dim=1)

        final_scores = []

        for sc, si in zip(sorted_scores, sorted_index):
            final_scores.append((sc[sc > 0], si[sc > 0]))

        return final_scores


def main(args):
    query_files = []
    db_files = []

    queries = VectorDatabase()
    vector_db = VectorDatabase()

    with open(args.input_file) as spec_in:
        for line in spec_in:
            data_type, embed_file = [x.strip() for x in line.split(",")]
            data_type = data_type.lower()

            if data_type == "q":
                query_files.append(embed_file)

            if data_type == "d":
                db_files.append(embed_file)

    queries.initialize(query_files)
    vector_db.initialize(db_files)

    queries.get_score_thresholds(vector_db)

    search_results = vector_db.search(queries)

    output_data = [vars(queries), vars(vector_db), search_results]
    pickle.dump(output_data, open(args.output_file, "wb"))

    if args.print_output:
        report_results(output_data)
