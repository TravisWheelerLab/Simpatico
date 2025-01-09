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
    description="Generate LITPCBA protein graphs and set pocket mask data."
)
parser.add_argument(
    "-d",
    "--directory",
    help="root litpcba directory",
)
parser.add_argument("-o", "--out", help="Output directory for new protein graphs.")

args = parser.parse_args()


def main(args):
    target_dirs = glob(args.directory + "/*/")

    for td in target_dirs:
        target_id = td.split("/")[-2]

        for f_i, ligand_file in enumerate(glob(td + "*_ligand.mol2")):
            mol_graph = molfile2pyg(ligand_file)

            if mol_graph is None:
                continue

            pdb_id = mol_graph.name[0].split("_")[0]

            pdb_dl_location = f"{td}/{pdb_id}.pdb"
            graph_file_out = f"{args.out}/{target_id}_{pdb_id}.pyg"

            os.system(
                f"curl https://files.rcsb.org/download/{pdb_id}.pdb -o {pdb_dl_location}"
            )

            pdb_graph = pdb2pyg(pdb_dl_location)
            proximal_protein_atoms = radius(mol_graph.pos, pdb_graph.pos, 5)[0].unique()
            pdb_graph.pocket_mask = torch.zeros(pdb_graph.x.size(0)).bool()
            pdb_graph.pocket_mask[proximal_protein_atoms] = True

            torch.save(pdb_graph, graph_file_out)
            print(graph_file_out)
            os.remove(pdb_dl_location)


main(args)
