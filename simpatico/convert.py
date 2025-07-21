import argparse
from pathlib import Path
import torch
import sys
from glob import glob
from os import path
from simpatico import config
from simpatico.utils.pdb_utils import pdb2pyg
from simpatico.utils.mol_utils import molfile2pyg


def add_arguments(parser):
    parser.add_argument(
        "-i",
        "--input",
        help='List of file paths or quote-bound unix-style path (e.g. "/path/to/data/*.pdb") describing input data.',
    )
    parser.add_argument("-o", "--output_directory", required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "-p",
        "--protein",
        action="store_true",
        help="convert input files to protein graphs",
    )
    group.add_argument(
        "-m",
        "--molecule",
        action="store_true",
        help="convert input files to small molecule structures",
    )
    group.add_argument(
        "--suffix",
        default=None,
        help="optional suffix to insert between filename and extension in converted PyG files (file<suffix>.pyg)",
    )

    # sets parser's main function to the main function in this script
    parser.set_defaults(main=main)


def gather_structure_files(input_string) -> list[str]:
    """
    Parse input string and generate list of input structural files.
    Args:
        input_string (str): Path to input, either batch file (.txt, .csv)
    Returns:
        (list[str]): list of structure files to convert to PyG graphs.
    """
    _, input_filetype = path.splitext(input_string)
    # filetypes we directly convert
    structure_filetypes = config["protein_filetypes"] + config["molecule_filetypes"]

    # if input is a list, we need to process the file.
    # otherwise, gather files with glob function
    input_is_list = input_filetype not in structure_filetypes

    if input_is_list:
        with open(input_string, "r") as structure_file_list:
            # filter out any length 0 lines from structure file list
            structure_files = [
                l for l in structure_file_list.read().splitlines() if len(l)
            ]
        return structure_files
    else:
        return glob(input_string)


def new_filename(input_file, extension, output_dir, suffix=None):
    """
    Generate the the filename to be used for file converted to PyG.
    Args:
        input_file (str): path to input filename.
        extension (str): file extension to use (usually .pyg or .pkl).
        output_dir (str): directory where output file will be stored.
        suffix (str, optional): string to append to file name before extension.
    Returns:
        (str): path to new PyG file.
    """
    basename = path.splitext(path.basename(input_file))[0]
    output_dir = output_dir + "/" if output_dir[-1] != "/" else output_dir
    extension = "." + extension if extension[0] != "." else extension

    if suffix is None:
        suffix = ""

    new_path = f"{output_dir}{basename}{suffix}{extension}"
    return new_path


def main(args):
    structure_files = gather_structure_files(args.input)
    converter = pdb2pyg if args.protein else molfile2pyg

    Path(args.output_directory).mkdir(parents=True, exist_ok=True)

    for sf in structure_files:
        pyg_file_out = new_filename(sf, ".pyg", args.output_directory, args.suffix)
        pyg_graph = converter(sf)
        torch.save(pyg_graph, pyg_file_out)
