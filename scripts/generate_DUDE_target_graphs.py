import argparse
import torch
import os
import sys
from glob import glob
from simpatico.utils.mol_utils import molfile2pyg
from simpatico.utils.pdb_utils import pdb2pyg
import subprocess
from torch_geometric.nn import radius

parser = argparse.ArgumentParser(
    description="Generate DUDE protein graphs and set pocket mask data."
)
parser.add_argument(
    "-d",
    "--directory",
    help="root DUDE directory",
)
parser.add_argument("-o", "--out", help="Output directory for new protein graphs.")

args = parser.parse_args()


def main(args):
    target_dirs = glob(args.directory + "/*/")

    for td in target_dirs:
        target_id = td.split("/")[-2]
        mol_graph = molfile2pyg(td + "/crystal_ligand.mol2")

        if mol_graph is None:
            continue

        graph_file_out = f"{args.out}/{target_id}_receptor.pyg"
        pdb_graph = pdb2pyg(td + "/receptor.pdb")
        proximal_protein_atoms = radius(mol_graph.pos, pdb_graph.pos, 5)[0].unique()
        pdb_graph.pocket_mask = torch.zeros(pdb_graph.x.size(0)).bool()
        pdb_graph.pocket_mask[proximal_protein_atoms] = True

        torch.save(pdb_graph, graph_file_out)
        print(graph_file_out)


main(args)
