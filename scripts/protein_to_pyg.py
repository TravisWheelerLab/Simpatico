import argparse
import sys
from pathlib import Path
import os
from glob import glob
import torch
from simpatico.pdb_utils import pdb2pyg
from simpatico.mol_utils import molfile2pyg


def main():
    parser = argparse.ArgumentParser(
        description="""\
Iterate through PDBBind complex directories and generate
PyG graphs from the protein PDBs.
"""
    )
    parser.add_argument(
        "-f",
        "--file_structure",
        help="Specify file structure. If generating protein graphs from PDBBind, this will look something like /path/to/pdbbind/*/*_protein.pdb",
    )
    parser.add_argument(
        "-o",
        "--outpath",
        required=True,
        type=str,
        help="Directory where generated PyG graphs will be saved.",
    )
    parser.add_argument(
        "--no-overwrite", action="store_true", help="Do not overwrite existing files"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "-p", "--protein", action="store_true", help="Generate protein PyG graph files."
    )
    group.add_argument(
        "-m",
        "--molecule",
        action="store_true",
        help="Generate molecular PyG graph files.",
    )

    args = parser.parse_args()
    output_file_template = args.outpath + "/%s.pyg"
    structure_files = glob(args.file_structure)
    total = len(structure_files)

    for structure_i, structure_f in enumerate(structure_files):
        filename = structure_f.split("/")[-1].split(".")[0]
        graph_file_out = output_file_template % filename

        if args.no_overwrite:
            if os.path.exists(graph_file_out):
                continue
            else:
                # Create an empty file so parallel jobs skip this one.
                Path(graph_file_out).touch()

        # new_graph = pdb2pyg(structure_f) if args.protein else sdf2pyg(structure_f)
        new_graph = (
            pdb2pyg(structure_f)
            if args.protein
            else molfile2pyg(structure_f, get_pos=True)
        )

        if new_graph is None:
            continue

        torch.save(new_graph, graph_file_out)
        print(f"{structure_i} of {total}: {graph_file_out}")


if __name__ == "__main__":
    main()
