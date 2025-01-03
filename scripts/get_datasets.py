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
    parser.add_argument(
        "-v",
        "--validation_ids",
        type=str,
        help="path to line-separated file of ids to withold from training and use as validation.",
    )
    parser.add_argument("-r", "--ratio", default=0.1)

    args = parser.parse_args()

    if args.validation_ids is not None:
        validation_ids = []
        with open(args.validation_ids, "r") as v_in:
            for line in v_in:
                validation_ids.append(line.rstrip())

    protein_files = glob(args.protein_file_structure)
    shuffle(protein_files)

    mol_files = glob(args.molecular_file_structure)
    file_out = args.outpath

    protein_graphs = []
    molecular_graphs = []
    p_i = 0

    for pf in protein_files:
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

    if args.validation_ids is None:
        for i in range(args.n):
            v_set_length = int(len(protein_graphs) * args.ratio)
            v_start_idx = i * v_set_length
            v_stop_idx = v_start_idx + v_set_length

            if args.n > 1:
                file_out = (
                    args.out_path.split(".")[0]
                    + f"_{i+1}."
                    + args.out_path.split(".")[1]
                )

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
    else:
        train_proteins = []
        train_mols = []

        validation_proteins = []
        validation_mols = []

        for protein_graph, mol_graph in zip(protein_graphs, molecular_graphs):
            if protein_graph.name.split("_")[0] in validation_ids:
                validation_proteins.append(protein_graph)
                validation_mols.append(mol_graph)
            else:
                train_proteins.append(protein_graph)
                train_mols.append(mol_graph)

        torch.save(
            (
                (train_proteins, train_mols),
                (
                    validation_proteins,
                    validation_mols,
                ),
            ),
            file_out,
        )


if __name__ == "__main__":
    main()
