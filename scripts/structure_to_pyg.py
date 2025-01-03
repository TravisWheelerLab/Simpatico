import argparse
import sys
from pathlib import Path
import os
from glob import glob
import torch
from simpatico.utils.pdb_utils import pdb2pyg
from simpatico.utils.mol_utils import molfile2pyg


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
    parser.add_argument(
        "--dirname-level",
        type=int,
        default=1,
        help="""number of directory levels to use in naming file (1 is only filename, 2 is d).
        e.g. [--dirname-level 1] produces pyg files like 'pdb_filename.pyg'.
             [--dirname-level 2] produces pyg files like 'pdb_dir_pdb_filename.pyg'.
        """,
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
        filename = "_".join(structure_f.split("/")[-args.dirname_level :]).split(".")[0]
        graph_file_out = output_file_template % filename

        print(f"{structure_i} of {total}: {graph_file_out}")
        # get name of file, ignoring parent directories and file extension

        # insert filename into output file template to produce output path for final PyG file

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


if __name__ == "__main__":
    main()
