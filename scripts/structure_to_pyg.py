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
        description="Generate PyG graphs from protein PDBs, or molecular SDF/PDB/Mol files."
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

    # get list of all structure files that correspond to FILE_STRUCTURE parameter
    structure_files = glob(args.file_structure)
    total = len(structure_files)

    # for every structure file, convert to pyg and save in provided OUTPATH
    for structure_i, structure_f in enumerate(structure_files):
        # get name of file, ignoring parent directories and file extension
        filename = structure_f.split("/")[-1].split(".")[0]

        # insert filename into output file template to produce output path for final PyG file
        graph_file_out = output_file_template % filename

        if args.no_overwrite:
            if os.path.exists(graph_file_out):
                continue
            else:
                # create an empty file so parallel jobs know to skip current target
                Path(graph_file_out).touch()

        # use appropriate conversion method for protein or molecular input
        if args.protein:
            new_graph = pdb2pyg(structure_f)
        else:
            new_graph = molfile2pyg(structure_f, get_pos=True)

        # if conversion is unsuccessful, will return None
        if new_graph is None:
            continue

        torch.save(new_graph, graph_file_out)
        print(f"{structure_i} of {total}: {graph_file_out}")


if __name__ == "__main__":
    main()
