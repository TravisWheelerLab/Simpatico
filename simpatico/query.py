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

from typing import Callable
from glob import glob


def add_arguments(parser):
    parser.add_argument(
        "-i",
        "--input_path",
        type=str,
        help="R|Path to input file.\n"
        "- Protein: .pdb file\n"
        "- Small molecule: .smi, .ism, .sdf, or .pdb file\n"
        "- Batch: .txt file with one path per line\n"
        "  * For proteins: each line = .pdb path, pocket spec (comma-separated)",
    )
    parser.add_argument("-w", "--weight_location")
    parser.add_argument("-o", "--output_path")
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query tool")
    add_arguments(parser)
    args = parser.parse_args()
    args.func(args)
