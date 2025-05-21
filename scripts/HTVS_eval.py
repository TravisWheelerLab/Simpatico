import torch
import time
from copy import deepcopy
import sys
import pickle
import os
import numpy as np
import faiss
import argparse
from glob import glob


def get_args():
    parser = argparse.ArgumentParser(
        description="Convert PyG outputs of eval script into faiss index files."
    )

    parser.add_argument(
        "-i",
        "--input_path",
        type=str,
        help='Path to input csv file. Columns should be "/path/to/target.pyg, /path/to/actives.pyg"',
    )
    parser.add_argument(
        "-d",
        "--decoy_path",
        type=str,
        help='Path to directory containing decoy .pyg files. Expecting unix-style pathname in quotations (e.g. "/path/to/*.pyg")',
    )
    parser.add_argument("-o", "--output_path")
    parser.add_argument("--prefix", type=str, default="")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for training",
    )
    parser.add_argument("-m", "--max-D", type=float, default=0)
    parser.add_argument("-t", "--threshold-S", type=float, default=0)
    parser.add_argument("--no-actives", action="store_true")
    parser.add_argument("--scoring-params-out", type=str, default=None)
    parser.add_argument("--scoring-params-in", type=str, default=None)
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


def concat_actives_decoys(
    decoy_index, decoy_batch, active_embeds, active_batch, gpu=True
):
    decoy_index.add(active_embeds)
    all_batch = torch.hstack((decoy_batch, (active_batch + decoy_batch[-1] + 1)))

    decoy_mask = torch.zeros_like(decoy_batch.unique())
    active_mask = torch.ones_like(active_batch.unique())
    active_mol_mask = torch.hstack((decoy_mask, active_mask)).bool()

    if gpu:
        gpu_res = faiss.StandardGpuResources()
        decoy_index = faiss.index_cpu_to_gpu(gpu_res, 0, decoy_index)

    return decoy_index, all_batch, active_mol_mask


def get_scores(D, I, max_D, threshold_S, mol_batch, target_batch):
    D = torch.as_tensor(D)
    I = torch.as_tensor(I)

    S = torch.clamp(max_D.unsqueeze(1) - D - threshold_S.unsqueeze(1), min=0)
    mol_I = mol_batch[I].long()

    target_mol_scores = []

    for t_i in target_batch.unique():
        target_mask = target_batch == t_i
        S_t = S[target_mask].flatten()
        I_t = I[target_mask].flatten()
        mol_I = mol_batch[I_t].long()

        mol_scores = torch.zeros_like(mol_batch.unique()).float()
        mol_scores.scatter_add_(0, mol_I, S_t)
        target_mol_scores.append(mol_scores)

    target_mol_scores = torch.vstack((target_mol_scores))
    return target_mol_scores


def pyg_to_faiss(pyg_graph, gpu=True):
    atom_embeds = pyg_graph.x.detach().cpu().numpy().astype("float32")

    dim = atom_embeds.shape[1]
    index = faiss.IndexFlatL2(dim)

    if gpu:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)

    index.add(atom_embeds)
    return [index, pyg_graph.batch]


def vectors_to_faiss(vectors, gpu=True):
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)

    if gpu:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)

    index.add(vectors)
    return index


def concatenate_vectors(vectors):
    vector_batch = []
    batch_index = 0

    for v in vectors:
        vector_batch += [batch_index for _ in range(len(v))]
        batch_index += 1

    return np.vstack(vectors), vector_batch


def concatenate_batches(batches):
    batch_stack = []

    for b_i, b in enumerate(batches):
        if b_i != 0:
            b = b.clone() + batch_stack[-1][-1] + 1
        batch_stack.append(b)

    return torch.hstack(batch_stack)


def concatenate_index_pyg(faiss_index, faiss_batch, pyg_graph, gpu=True):
    pyg_embeds = pyg_graph.x.detach().cpu().numpy().astype("float32")

    if gpu:
        res = faiss.StandardGpuResources()
        faiss_index = faiss.index_cpu_to_gpu(res, 0, faiss_index)

    faiss_index.add(pyg_embeds)
    full_batch = torch.hstack((faiss_batch, pyg_graph.batch + faiss_batch[-1] + 1))
    return faiss_index, full_batch


def reduce_batch(mol_batch, target_batch):
    _, counts = mol_batch.unique_consecutive(return_counts=True)
    ptr = torch.cat([torch.tensor([0]), counts.cumsum(0)])
    reduced_target_batch = target_batch[ptr[:-1]]
    return reduced_target_batch


def screen(scoring_params, targets, decoys, actives=None):
    results = []

    if actives is not None:
        faiss_vectors = np.vstack((actives.vectors, decoys.vectors))
        faiss_mol_batch = torch.hstack(
            (actives.mol_batch, decoys.mol_batch + actives.mol_batch[-1] + 1)
        )
        faiss_target_batch = torch.hstack((actives.target_batch, decoys.target_batch))
    else:
        faiss_vectors = decoys.vectors
        faiss_mol_batch = decoys.mol_batch
        faiss_target_batch = decoys.target_batch

    faiss_index = vectors_to_faiss(faiss_vectors)

    if actives is not None:
        mol_target_batch = reduce_batch(faiss_mol_batch, faiss_target_batch)

    D, I = faiss_index.search(targets.vectors, 2048)

    max_D = scoring_params[0][targets.target_batch]
    threshold_S = scoring_params[1][targets.target_batch]

    scores = get_scores(D, I, max_D, threshold_S, faiss_mol_batch, targets.target_batch)

    for t_i, t_scores in enumerate(scores):
        if actives is not None:
            active_mask = mol_target_batch == t_i
            active_scores = t_scores[active_mask]
            decoy_scores = t_scores[~active_mask]

            results.append(
                (
                    (len(active_scores), active_scores[active_scores > 0]),
                    (len(decoy_scores), decoy_scores[decoy_scores > 0]),
                )
            )
        else:
            results.append((len(t_scores), t_scores[t_scores > 0]))

    return results


class VectorIndex:
    def __init__(self, pyg_files, decoys=False):
        self.vectors = []
        self.target_batch = []
        self.mol_batch = []

        last_mol = -1

        for f_i, pyg_f in enumerate(pyg_files):
            pyg_graph = torch.load(pyg_f, weights_only=False)
            pyg_vectors = pyg_graph.x.numpy()
            self.vectors.append(pyg_vectors)

            self.mol_batch.append(pyg_graph.batch + last_mol + 1)

            if decoys is False:
                self.target_batch.append(torch.zeros_like(pyg_graph.batch) + f_i)

            last_mol = self.mol_batch[-1][-1]

        self.vectors = np.vstack(self.vectors)
        self.mol_batch = torch.hstack(self.mol_batch)

        if decoys:
            self.target_batch = torch.ones_like(self.mol_batch) * (-1)
        else:
            self.target_batch = torch.hstack(self.target_batch)


def get_target_active_files(csv, ignore_actives=False):
    target_files = []
    active_files = []

    with open(args.input_path) as csv_in:
        for t_i, line in enumerate(csv_in):
            line_data = line.rstrip().split(",")
            target_files.append(line_data[0].strip())

            if ignore_actives:
                continue

            if len(line_data) > 1:
                active_files.append(line_data[1].strip())

    if len(active_files) != len(target_files):
        active_files = None

    return target_files, active_files


def get_score_thresholds(
    X1, X2, target_batch, n_random: int = 4000, q: float = 0.99, n_trials=5
):
    X1 = torch.as_tensor(X1, dtype=torch.float32)
    # pick sample size that works for both arrays
    n1, _ = X1.shape
    n2, _ = X2.shape
    k1 = min(n_random, n1)
    k2 = min(n_random, n2)

    target_max_D = torch.zeros(target_batch[-1] + 1)
    target_thresholds = torch.zeros_like(target_max_D)

    max_D_mean = 0
    threshold_mean = 0

    for trial_n in range(n_trials):
        idx2 = np.random.choice(n2, k2, replace=False).astype(np.int64)

        for target_index in torch.arange(target_max_D.size(0)):
            rv_1 = X1[target_batch == target_index]
            rv_2 = torch.as_tensor(X2[idx2], dtype=torch.float32)

            D = torch.cdist(rv_1, rv_2).flatten()
            max_D = D.max()
            S = max_D - D
            threshold = torch.quantile(S, q)

            target_max_D[target_index] += max_D / n_trials
            target_thresholds[target_index] += threshold / n_trials

    return target_max_D, target_thresholds


def get_scoring_params(target_index, decoy_index, active_index=None):
    if active_index is not None:
        decoy_pop = np.vstack((decoy_index.vectors, active_index.vectors))
    else:
        decoy_pop = decoy_index.vectors

    scoring_params = get_score_thresholds(
        target_index.vectors, decoy_pop, target_index.target_batch
    )

    return scoring_params


def main(args):
    target_files, active_files = get_target_active_files(
        args.input_path, args.no_actives
    )

    target_ids = [f.split("/")[-1].split("_")[0] for f in target_files]

    target_index = VectorIndex(target_files)
    active_index = None

    if active_files is not None:
        active_index = VectorIndex(active_files)

    decoy_files = glob(args.decoy_path)

    scoring_params = None

    if args.scoring_params_in:
        scoring_params = pickle.load(open(args.scoring_params_in, "rb"))

    if args.prefix != "":
        args.prefix += "_"

    for d_i, df in enumerate(decoy_files):
        start = time.time()
        decoy_ID = df.split("/")[-1].split("_")[0]

        outfile = f"{args.output_path}/{args.prefix}{decoy_ID}.pkl"

        if os.path.exists(outfile):
            continue

        decoy_index = VectorIndex([df], decoys=True)

        if active_index is not None:
            outfile = outfile.replace(".pkl", "_ACTIVES.pkl")

        if scoring_params is None:
            scoring_params = get_scoring_params(target_index, decoy_index, active_index)

            if args.scoring_params_out:
                pickle.dump(scoring_params, open(args.scoring_params_out, "wb"))

        results = screen(scoring_params, target_index, decoy_index, active_index)
        end = time.time()

        pickle.dump([target_ids, results, end - start], open(outfile, "wb"))
        print(outfile, end - start)


if __name__ == "__main__":
    args = get_args()
    main(args)
