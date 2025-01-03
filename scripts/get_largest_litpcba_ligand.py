import argparse
import torch
import os
import sys
from glob import glob
from simpatico.utils.mol_utils import molfile2pyg
from simpatico.utils.pdb_utils import pdb2pyg
import subprocess
from torch_geometric.nn import radius


def main():
    parser = argparse.ArgumentParser(
        description="identify the largest provided ligand for pocket-specification of each LITPCBA structure"
    )
    parser.add_argument(
        "-d",
        "--directory",
        help="root litpcba directory",
    )
    parser.add_argument(
        "-o",
        "--out_directory",
        help="Directory for storing generated .pyg files",
    )

    args = parser.parse_args()

    target_dirs = glob(args.directory + "/*/")

    for td in target_dirs:
        largest_graph = None

        for f_i, ligand_file in enumerate(glob(td + "*_ligand.mol2")):
            mol_graph = molfile2pyg(ligand_file)

            if mol_graph is None:
                continue

            if largest_graph is None:
                largest_graph = mol_graph
                continue

            if mol_graph.x.size(0) > largest_graph.x.size(0):
                largest_graph = mol_graph

        largest_id = largest_graph.name[0].split("_")[0]

        os.system(
            f"curl https://files.rcsb.org/download/{largest_id}.pdb -o {td}/{largest_id}.pdb"
        )

        pdb_graph = pdb2pyg(f"{td}/{largest_id}.pdb")
        pocket_mask = torch.zeros(pdb_graph.x.size(0)).bool()
        proximal_protein_atoms = radius(largest_graph.pos, pdb_graph.pos, 4)[0].unique()
        pocket_mask[proximal_protein_atoms] = True
        pdb_graph_filename = td.split("/")[-2] + f".pyg"
        torch.save(pdb_graph, args.out_directory + "/" + pdb_graph_filename)


main()
