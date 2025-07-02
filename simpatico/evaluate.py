# scripts/train.py
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
)
from simpatico.models.molecule_encoder.MolEncoder import MolEncoder
from simpatico.models.protein_encoder.ProteinEncoder import ProteinEncoder
from simpatico.models import MolEncoderDefaults, ProteinEncoderDefaults
from simpatico.utils.pdb_utils import pdb2pyg
from simpatico.utils.utils import SmartFormatter
from simpatico import config

from typing import Callable
from glob import glob


def add_arguments(parser):
    parser.add_argument(
        "input_file",
        type=str,
        help="R|Path to input file.\n"
        "- Protein: .pdb file\n"
        "- Small molecule: .smi, .ism, .sdf, or .pdb file\n"
        "- Batch: .txt file with one path per line\n"
        "  * For proteins: each line = .pdb path, pocket spec (comma-separated)",
    )
    parser.add_argument("output_path")
    parser.add_argument("-w", "--weights-file", default=config["default_weights_path"])
    parser.add_argument(
        "--no-overwrite", action="store_true", help="Do not overwrite existing files"
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
        help="Indicates input file is PyG graph",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for training",
    )
    parser.add_argument(
        "--pocket-coordinates",
        type=str,
        default=None,
        help="File describing pocket coordinates (.csv or any compatible small-molecule file)",
    )
    parser.add_argument("--suffix", type=str, default="")

    parser.set_defaults(main=main)
    return parser


def main(args):
    device = args.device

    # Weights for protein and ligand models are stored in one file.
    # first item is protein model weights, second is molecule model weights
    if args.protein:
        encoder = ProteinEncoder(**ProteinEncoderDefaults).to(device)
        weight_index = 0

    elif args.molecule:
        encoder = MolEncoder(**MolEncoderDefaults).to(device)
        weight_index = 1

    encoder.load_state_dict(
        torch.load(args.weights_file, map_location=device)[weight_index]
    )

    encoder.eval()
    input_file = args.input_file
    _, input_filetype = path.splitext(args.input_file)

    input_list = []

    if input_filetype in [".txt", ".csv"]:
        with open(input_file, "r") as input_in:
            for line in input_in:
                line_content = [x.strip() for x in line.split(",")]
                input_list.append(line_content)
    else:
        input_list = [[input_file, None]]

    for file_i, structure_data in enumerate(input_list):
        structure_file = structure_data[0]

        if len(structure_data) > 1:
            pocket_data = structure_data[1]

        structure_file_basename, structure_filetype = path.splitext(
            path.basename(structure_file)
        )

        if args.protein:
            pocket_spec_file = pocket_data or args.pocket_coordinates
            pocket_spec = get_xyz_from_file(pocket_spec_file)

        embed_failed = False
        output_path = args.output_path

        Path(output_path).mkdir(parents=True, exist_ok=True)

        if output_path[-1] != "/":
            output_path += "/"

        outfile = (
            output_path + structure_file_basename + "_embeds" + args.suffix + ".pyg"
        )

        if args.no_overwrite:
            if os.path.exists(outfile) and os.path.getsize(outfile) > 0:
                continue
            else:
                # create an empty file so parallel jobs know to skip current target
                Path(outfile).touch()

        if args.graph_in:
            input_g = torch.load(structure_file)
        elif args.protein:
            input_g = pdb2pyg(structure_file, pocket_coords=pocket_spec)
        elif args.molecule:
            input_g = molfile2pyg(structure_file, get_pos=True)

        graph_source = None

        if hasattr(input_g, "source"):
            graph_source = input_g.source

        if args.protein:
            input_g = Batch.from_data_list([input_g])

        data_loader = DataLoader(input_g, batch_size=1024, shuffle=False)
        data_out = []

        for batch in data_loader:
            if args.protein:
                if hasattr(batch, "pocket_mask") == False:
                    print("No pocket mask specified for %s" % structure_file)

                    if args.no_overwrite:
                        os.remove(outfile)

                    embed_failed = True
                    continue

            with torch.no_grad():
                try:
                    embeds = encoder(batch.to(device))
                except:
                    embed_failed = True
                    continue

            if args.molecule:
                batch.x = embeds
                data_out += batch.cpu().to_data_list()

            if args.protein:
                data_out.append(
                    Data(x=embeds[0].cpu(), pos=embeds[1].cpu(), batch=embeds[2].cpu())
                )

        if embed_failed:
            continue

        embeds_out = Batch.from_data_list(data_out)

        if graph_source:
            embeds_out.source = graph_source

        torch.save(embeds_out, outfile)
        print(outfile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation tool")
    add_arguments(parser)
    args = parser.parse_args()
    args.func(args)
