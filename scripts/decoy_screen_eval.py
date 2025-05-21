import torch
import sys
import pickle
import os
import numpy as np
import faiss
import argparse
from glob import glob
import time


def get_args():
    parser = argparse.ArgumentParser(
        description="Convert PyG outputs of eval script into faiss index files."
    )

    parser.add_argument(
        "-i",
        "--input_path",
        type=str,
        help='Path to input file. May use unix style pathname patterns if enclosed in quotes. e.g. "/path/to/directory/*.pyg".',
    )
    parser.add_argument("-o", "--output_path")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for training",
    )
    return parser.parse_args()


def load_faiss_index(path, gpu=True):
    index = faiss.read_index(path)
    batch = pickle.load(open(path.replace(".faiss", ".batch"), "rb"))

    if gpu:
        gpu_res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(gpu_res, 0, index)

    return index, batch


def index_to_vectors(faiss_index):
    n = faiss_index.ntotal
    d = faiss_index.d
    vectors = np.zeros((n, d), dtype=np.float32)
    faiss_index.reconstruct_batch(np.arange(n), vectors)
    return vectors


def concat_actives_decoys(decoy_index, decoy_batch, active_index, active_batch):
    active_vectors = active_index.reconstruct_n(0, active_index.ntotal)
    decoy_index.add(active_vectors)
    all_batch = torch.hstack((decoy_batch, (active_batch + decoy_batch[-1] + 1)))

    decoy_mask = torch.zeros_like(decoy_batch.unique())
    active_mask = torch.ones_like(active_batch.unique())
    active_mol_mask = torch.hstack((decoy_mask, active_mask)).bool()

    return decoy_index, all_batch, active_mol_mask


def get_score_thresholds(target_index, faiss_index, n_random=4000, q=0.99):
    # r_ids_1 = np.random.choice(faiss_index.ntotal, n_random, replace=False).astype(
    #     np.int64
    # )

    r_ids_2 = np.random.choice(faiss_index.ntotal, n_random, replace=False).astype(
        np.int64
    )

    rv_1 = np.zeros((target_index.ntotal, faiss_index.d), dtype=np.float32)
    rv_2 = np.zeros((n_random, faiss_index.d), dtype=np.float32)

    faiss_index.reconstruct_batch(torch.arange(target_index.ntotal).numpy(), rv_1)
    faiss_index.reconstruct_batch(r_ids_2, rv_2)

    rv_1 = torch.from_numpy(rv_1)
    rv_2 = torch.from_numpy(rv_2)

    random_D = torch.cdist(rv_1, rv_2).flatten()
    max_D = random_D.max()
    random_S = max_D - random_D
    threshold_S = torch.quantile(random_S, q)
    return max_D, threshold_S


def get_score(D, I, max_D, threshold_S, batch):
    S = torch.clamp(max_D - D - threshold_S, min=0).flatten()
    I = I.flatten()
    mol_I = batch[I].long()
    atom_scores = torch.zeros_like(batch.unique()).float()
    atom_scores.scatter_add_(0, mol_I, S)
    return atom_scores


def get_EF(sorted_active_mask, sample_ratio):
    population_size = sorted_active_mask.size(0)
    sample_size = int(population_size * sample_ratio)

    sample = sorted_active_mask[:sample_size]

    true_ratio = sorted_active_mask.sum().float() / population_size
    sample_ratio = sample.sum().float() / sample_size
    EF = sample_ratio / true_ratio
    return EF


def main(args):
    sample_ratios = []
    start = time.time()
    with open(args.input_path) as csv_in:
        for line in csv_in:
            line = line.rstrip()
            target_index_path, active_index_path, decoy_index_paths = [
                x.strip() for x in line.split(",")
            ]
            target_index, _ = load_faiss_index(target_index_path, False)
            query_vectors = index_to_vectors(target_index)
            active_index, active_batch = load_faiss_index(active_index_path, False)
            decoy_index_paths = glob(decoy_index_paths)
            print(target_index_path)
            target_id = target_index_path.split("/")[-1].split("_")[0]

            for d_i, decoy_index_file in enumerate(decoy_index_paths):
                # if d_i == 0:
                decoy_index, decoy_batch = load_faiss_index(decoy_index_file, False)
                all_index, all_batch, active_mask = concat_actives_decoys(
                    decoy_index, decoy_batch, active_index, active_batch
                )
                max_D, threshold_S = get_score_thresholds(target_index, all_index)

                # Move to GPU
                res = faiss.StandardGpuResources()
                all_index = faiss.index_cpu_to_gpu(res, 0, all_index)

                # Now gpu_index is ready for searching
                D, I = all_index.search(query_vectors, 2048)
                mol_scores = get_score(D, I, max_D, threshold_S, all_batch)
                sorted_mol_score_index = mol_scores.argsort(descending=True)
                sorted_active_mask = active_mask[sorted_mol_score_index]
                EFs = [get_EF(sorted_active_mask, sr) for sr in [0.005, 0.01, 0.05]]
                sample_ratios.append([target_id, EFs])

    print(f"time: {time.time()-start}")
    pickle.dump(sample_ratios, open("/home/jgaiser/dekois_EF_2.pkl", "wb"))


if __name__ == "__main__":
    args = get_args()
    main(args)
