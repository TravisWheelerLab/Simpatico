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

from typing import Callable
from glob import glob

import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


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
    """
    Interface for embedding collections to be used for building and querying FAISS vector databases.

    Args:
        embed_files (list[str]): list of filepaths to PyG graph files containing embedding values to store in VectorDatabase.
        score_threshold (tuple): max-distance and n-quantile distantance statistics used in aggregation/scoring function during query.

    Attributes:
        sources (list): ordered list of filepaths of ultimate vector sources.
        vectors (torch.Tensor): list of vectors.
        item_batch (torch.Tensor): pyg-style batch tensor corresponding vectors to a specific item/graph.
        file_batch (torch.Tensor): pyg-style batch tensor corresponding vectors to a source file.
        source_index (torch.Tensor): pyg-style batch tensor specifying which item of the original source file vector belongs to.
        score-thresholds (tuple): max-distance and n-quantile distantance statistics used in aggregation/scoring function during query.
    """

    def __init__(self, embed_files, score_thresholds=None):
        self.sources = []
        self.score_thresholds = score_thresholds

        vectors = []
        item_batch = []
        file_batch = []
        source_index = []

        # Iterate through inputs and consolidate batch values across files.
        for ef_i, embed_file in enumerate(embed_files):
            if isinstance(embed_file, str):
                g = torch.load(embed_file, weights_only=False)
            else:
                g = embed_file

            self.sources.append(g.source)
            vectors.append(g.x)

            # update batch values so we can stack (first value is previous batch's last value + 1)
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

    def get_index_data(self):
        """
        Return all indexing data.
        """
        return {
            "sources": self.sources,
            "item_batch": self.item_batch,
            "file_batch": self.file_batch,
            "source_index": self.source_index,
        }

    def get_score_thresholds(
        self, vector_db, n_random: int = 4000, q: float = 0.99, n_trials=5
    ):
        """
        Retrieve and store in `self.score_thresholds` per-item maximum observed distance and nth-quantile score values
        from a random sampling of distances between `self.vectors` and those of a different VectorDatabase
        (used for score aggregation during query).

        Args:
            vector_db (VectorDatabase): VectorDatabase object containing vectors to randomly sample distances from.
            n_random (int, optional): number of random vector distances to sample.
            q (float, optional): score quantile (default = 0.99)
            n_trials (int, optional): number of sampling trials to perform (workaround for maxing out `torch.cdist`.)
        """

        # We need a unique maximum distance and quantile score value for each item in the VectorDatabase.
        device = vector_db.vectors.device
        item_max_D = torch.zeros(self.item_batch[-1] + 1).to(device)
        item_thresholds = torch.zeros_like(item_max_D).to(device)

        for _ in range(n_trials):
            r_idx = torch.randperm(len(vector_db.vectors))[:n_random].to(device)
            random_vecs = vector_db.vectors[r_idx].to(device)

            for item_index in torch.arange(len(item_max_D)):
                item_vectors = self.vectors[self.item_batch == item_index]

                D = torch.cdist(item_vectors, random_vecs).flatten()
                max_D = D.max()
                S = max_D - D
                threshold = torch.quantile(S, q)

                item_max_D[item_index] += max_D / n_trials
                item_thresholds[item_index] += threshold / n_trials

        self.score_thresholds = (item_max_D, item_thresholds)

    def faiss_index(self, gpu=True):
        """
        Convert self.vectors object into a proper FAISS index for querying.
        Args:
            gpu (bool, optional): indicates use of GPU (default = True).
        Returns:
            faiss index object
        """
        np_vectors = self.np_vectors()

        dim = np_vectors.shape[1]
        index = faiss.IndexFlatL2(dim)

        if gpu:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)

        index.add(np_vectors)
        return index

    def np_vectors(self):
        """
        return numpy version of self.vectors.
        """
        return self.vectors.detach().cpu().numpy().astype("float32")

    def get_scores(self, D, I, query_db):
        """
        Provided the results of a FAISS-based nearest neighbors operation, get per-item scores of the queried VectorDatabase.
        Nearest-neighbors operation returns N = 2048 nearest neighbors.

        Args:
            D (np.array): numpy array of distances of 2048 neighbor-distance values per-vector.
            I (np.array): neighbor-vector indices corresponding to values in D.
            query_db (VectorDatabase): VectorDB used as query.
        """
        device = query_db.vectors.device

        D = torch.as_tensor(D).to(device)
        I = torch.as_tensor(I).to(device)
        

        score_thresholds = query_db.score_thresholds
        target_batch = query_db.item_batch
        

        max_D = score_thresholds[0][target_batch].unsqueeze(1)
        threshold_S = score_thresholds[1][target_batch].unsqueeze(1)
        
        # Score value calculated so that the smaller the vector distance, the greater the score.
        S = torch.clamp(max_D.unsqueeze(1) - D - threshold_S.unsqueeze(1), min=0)
        
        # get index of VectorDatabase item, rather than index of individual vectors.
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

    def query(self, query_db):
        """
        Query `self.vectors` with vectors from another VectorDatabase.

        Args:
            query_db (VectorDatabase): VectorDatabase used to query `self.vectors`.

        Returns:
            (list): List containing for each item in query, a 2-tuple list containing
            tensor of sorted non-zero scores and corresponding item index tensor from queried VectorDatabase.
        """
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

    with open(args.input_file) as spec_in:
        for line in spec_in:
            data_type, embed_file = [x.strip() for x in line.split(",")]
            data_type = data_type.lower()

            if data_type == "q":
                query_files.append(embed_file)

            if data_type == "d":
                db_files.append(embed_file)

    queries = VectorDatabase(query_files)
    vector_db = VectorDatabase(db_files)

    queries.get_score_thresholds(vector_db)

    search_results = vector_db.query(queries)

    # To produce human readable results, we need to save data from queries, database, and search results.
    output_data = [queries.get_index_data(), vector_db.get_index_data(), search_results]

    log.info("Successfully completed query, results stored in: %s", args.output_file)
    pickle.dump(output_data, open(args.output_file, "wb"))

    if args.print_output:
        report_results(output_data)
