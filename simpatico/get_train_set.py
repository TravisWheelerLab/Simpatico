import argparse
import pickle
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
    parser.add_argument("-o", "--output_file", required=True)

    # sets parser's main function to the main function in this script
    parser.set_defaults(main=main)


def gather_structure_files(input_string):
    line_data = []
    with open(input_string, "r") as structure_file_list:
        # filter out any length 0 lines from structure file list
        for line in structure_file_list:
            line = line.strip()
            columns = [x.strip() for x in line.split(",")]

            if len(columns) != 3:
                continue
            line_data.append(columns)
    return line_data


def main(args):
    structure_data = gather_structure_files(args.input)

    train_pairs = []
    validation_pairs = []

    for columns in structure_data:
        split = columns[0].lower()

        # TODO: Implement support for multiple ligands per single protein structure.
        ligand_graph = molfile2pyg(columns[2]).to_data_list()[0]
        protein_graph = pdb2pyg(columns[1], ligand_pos=ligand_graph.pos)

        if split == "t":
            train_pairs.append((protein_graph, ligand_graph))
        elif split == "v":
            validation_pairs.append((protein_graph, ligand_graph))

    with open(args.output_file, "wb") as tv_set_out:
        pickle.dump((train_pairs, validation_pairs), tv_set_out)
