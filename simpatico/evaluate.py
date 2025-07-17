# scripts/train.py
import logging
import traceback
import os
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
    handle_no_overwrite,
)
from simpatico.models.molecule_encoder.MolEncoder import MolEncoder
from simpatico.models.protein_encoder.ProteinEncoder import ProteinEncoder
from simpatico.utils.pdb_utils import pdb2pyg
from simpatico import config

from typing import Callable
from glob import glob

import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def add_arguments(parser):
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to input file.\n"
        "- Protein: .pdb file\n"
        "- Small molecule: .smi, .ism, .sdf, or .pdb file\n"
        "- Batch: .csv file with one item per line:\n"
        "    For proteins:\n"
        "      <protein_structure_path>, <pocket_spec_path>\n"
        "      ..."
        "  For molecules:\n"
        "      <molecule_structure_path>"
        "      ...",
    )
    parser.add_argument(
        "output_path", type=str, help="Path to directory for outputting embed file."
    )
    parser.add_argument(
        "-w",
        "--weights-file",
        default=config["default_weights_path"],
        help="Non-default weights",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for training (gpu or cpu)",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="",
        help="Suffix to append to output files. For example, with `--suffix _test`, output.pyg becomes output_test.pyg",
    )
    parser.add_argument(
        "--pocket-coordinates",
        type=str,
        default=None,
        help="File specifying pocket coordinates (.csv or any compatible small-molecule file)",
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Skip evaluation if output file already exists.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "-p",
        "--protein",
        action="store_true",
        help="Indicates protein evaluation.",
    )
    group.add_argument(
        "-m",
        "--molecule",
        action="store_true",
        help="Indicates molecule evaluation.",
    )
    parser.add_argument(
        "-g",
        "--graph-in",
        action="store_true",
        help="Indicates that input file is PyG graph",
    )

    parser.set_defaults(main=main)
    return parser


def get_encoder(args):
    """
    input may be protein or molecular structural file, or batch file for either.
    get encoder according to input type.

    Args:
        args (ArgumentParser): script arguments
    Returns:
        encoder  (MolEncoder | ProteinEncoder): proper encoder module
    """
    device = args.device

    # weights for protein and ligand models are stored in one file.
    # first item is protein model weights, second is molecule model weights
    # so set `weight_index` accordingly
    if args.protein:
        encoder = ProteinEncoder().to(device)
        weight_index = 0

    elif args.molecule:
        encoder = MolEncoder().to(device)
        weight_index = 1

    encoder.load_state_dict(
        torch.load(args.weights_file, map_location=device)[weight_index]
    )
    encoder.eval()

    return encoder


def get_input_list(args):
    """
    protein/molecule structure files are stored in list, which we iterate through during eval.
    gather those files and produce list.

    input: script arguments (ArgumentParser object)
    output: list of paths to structural files
    """
    input_file = args.input_file

    _, input_filetype = path.splitext(args.input_file)

    # store input file paths here
    input_list = []

    # if input is a batch file, iterate through and store paths in `input_list`
    if input_filetype in [".txt", ".csv"]:
        with open(input_file, "r") as input_in:
            for line in input_in:
                line_content = [x.strip() for x in line.split(",")]
                input_list.append(line_content)
    # if input is an individual structure file, store in `input_list` to be compatible with remaining script.
    else:
        input_list = [[input_file, None]]

    check_files(input_list)

    return input_list


def check_files(input_list):
    for row in input_list:
        for input_file in row:
            if input_file is None:
                continue

            if not path.exists(input_file):
                raise FileNotFoundError(f"Input file does not exist: {input_file}")


def get_input_data_loader(input_line, args):
    """
    prepare item from input list for evaluation, according to input type logic
    input:
        1. line from input list data
        2. script arguments (ArgumentParser object)
    output: data loader object containing data for eval (DataLoader)
    """
    structure_file = input_line[0]

    if len(input_line) > 1:
        pocket_data = input_line[1]
    else:
        pocket_data = None

    if args.protein:
        # if single protein structural file is input, path to pocket coordinate file is supplied in args
        pocket_spec_file = pocket_data or args.pocket_coordinates
        pocket_spec = get_xyz_from_file(pocket_spec_file)

    if args.graph_in:
        input_g = torch.load(structure_file)
    elif args.protein:
        input_g = pdb2pyg(structure_file, pocket_coords=pocket_spec)
    elif args.molecule:
        input_g = molfile2pyg(structure_file, get_pos=True)

    if args.protein:
        input_g = Batch.from_data_list([input_g])

    input_data_loader = DataLoader(input_g, batch_size=1024, shuffle=False)
    input_data_loader.source_file = structure_file

    return input_data_loader


def evaluate_data(input_data_loader, outfile, encoder, args):
    """
    Produce embeddings for data in input data loader
    input:
        1. input data loader (DataLoader object)
        2. source structure file (str)
    output: data loader object containing data for eval (Batch object)
    """
    embed_failed = False
    data_out = []

    for batch in input_data_loader:
        if args.protein:
            if hasattr(batch, "pocket_mask") == False:
                log.warning(
                    "No pocket mask specified for %s", input_data_loader.source_file
                )
                embed_failed = True
                break

        with torch.no_grad():
            try:
                embeds = encoder(batch.to(args.device))
            except Exception as e:
                log.error("Error during embedding: %s", e)
                traceback.print_exc()
                embed_failed = True
                break

        if args.molecule:
            batch.x = embeds
            data_out += batch.cpu().to_data_list()

        if args.protein:
            data_out.append(
                Data(x=embeds[0].cpu(), pos=embeds[1].cpu(), batch=embeds[2].cpu())
            )

    if embed_failed:
        if args.no_overwrite:
            os.remove(outfile)
        return None

    embeds_out = Batch.from_data_list(data_out)
    return embeds_out


def main(args):
    encoder = get_encoder(args)

    input_list = get_input_list(args)

    output_path = args.output_path

    if output_path[-1] != "/":
        output_path += "/"

    Path(output_path).mkdir(parents=True, exist_ok=True)

    for input_line in input_list:
        structure_file = input_line[0]
        structure_file_basename, _ = path.splitext(path.basename(structure_file))

        outfile = (
            output_path + structure_file_basename + "_embeds" + args.suffix + ".pyg"
        )

        if args.no_overwrite:
            # creates an empty file so parallel jobs know to skip current target
            # returns False if file exists
            if handle_no_overwrite(outfile) is False:
                continue

        input_data_loader = get_input_data_loader(input_line, args)
        embeds_out = evaluate_data(input_data_loader, outfile, encoder, args)

        if embeds_out is None:
            continue

        embeds_out.source = structure_file

        torch.save(embeds_out, outfile)
        log.info("completed: %s", outfile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Produce atom embeddings for protein or small molecule structure"
    )
    add_arguments(parser)
    args = parser.parse_args()
    args.func(args)
