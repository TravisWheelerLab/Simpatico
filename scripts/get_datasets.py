import argparse
from random import shuffle
import numpy as np
import sys
from pathlib import Path
import os
from glob import glob
import torch
from torch_geometric.data import Dataset


def main():

    parser = argparse.ArgumentParser(
        description="Generate train/validation PyG graph datasets from existing PyG graph files."
    )
    parser.add_argument(
        "-p",
        "--protein_file_structure",
        type=str,
        help="File structure for gathering all protein graph files. e.g. /path/to/proteins/pdb-id.pyg",
    )
    parser.add_argument(
        "-m",
        "--molecular_file_structure",
        type=str,
        help="File structure for gathering all molecule graph files. e.g. /path/to/molecules/pdb-id.pyg",
    )
    parser.add_argument(
        "-o",
        "--outpath",
        required=True,
        type=str,
        help="Ouptut file containing generated graph dataset. Number will be appended to filename if n is greater than 1. ",
    )
    parser.add_argument(
        "-n",
        default=1,
        type=int,
        help="Number of unique train/validation sets to generate",
    )
    parser.add_argument("-r", "--ratio", default=0.1)

    args = parser.parse_args()

    protein_files = glob(args.protein_file_structure)
    shuffle(protein_files)

    mol_files = glob(args.molecular_file_structure)

    protein_graphs = []
    molecular_graphs = []
    p_i = 0

    for pf in protein_files[:800]:
        p_i += 1
        pdb_id = pf.split("/")[-1].split("_")[0]
        for mf in mol_files:
            if mf.split("/")[-1].split("_")[0] == pdb_id:
                if p_i % 1000 == 0:
                    print(p_i)
                protein_graphs.append(torch.load(pf))
                # Molecular graph files contain batches by default
                molecular_graphs.append(torch.load(mf).to_data_list()[0])
                break

    for i in range(args.n):
        v_set_length = int(len(protein_graphs) * args.ratio)
        v_start_idx = i * v_set_length
        v_stop_idx = v_start_idx + v_set_length

        file_out = args.outpath
        if args.n > 1:
            file_out = file_out.split(".")[0] + f"_{i+1}." + file_out.split(".")[1]

        print(f"Saving set {i+1} of {args.n}: {file_out}")
        torch.save(
            (
                (
                    protein_graphs[:v_start_idx] + protein_graphs[v_stop_idx:],
                    molecular_graphs[:v_start_idx] + molecular_graphs[v_stop_idx:],
                ),
                (
                    protein_graphs[v_start_idx:v_stop_idx],
                    molecular_graphs[v_start_idx:v_stop_idx],
                ),
            ),
            file_out,
        )


if __name__ == "__main__":
    main()
