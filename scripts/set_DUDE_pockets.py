import argparse
import sys
import torch
from pathlib import Path
import os
from glob import glob
from rdkit.Chem import MolFromMol2File
from torch_geometric.nn import radius_graph, radius
from simpatico.utils.graph_utils import get_pocket_mask


def main():
    parser = argparse.ArgumentParser(
        description="Get pocket atom mask from provided ligand."
    )
    parser.add_argument(
        "-pf",
        "--protein_file_structure",
        help="Specify protein file structure.",
    )
    parser.add_argument(
        "-mf",
        "--mol_file_structure",
        help="Specify corresponding mol file structure.",
    )

    args = parser.parse_args()

    mol_files = glob(args.mol_file_structure)
    prot_graph_files = glob(args.protein_file_structure)

    for pg_file in prot_graph_files:
        pdb_id = pg_file.split("/")[-1].split("_")[0]
        pg = torch.load(pg_file)

        for mf in mol_files:
            if pdb_id == mf.split("/")[-2]:
                m = MolFromMol2File(mf, sanitize=False)
                m.GetConformer()

                conformer = m.GetConformer()
                # placeholder for positional data
                mol_pos = []

                for atom in m.GetAtoms():
                    atom_idx = atom.GetIdx()
                    atom_pos = conformer.GetAtomPosition(atom_idx)
                    mol_pos.append([atom_pos.x, atom_pos.y, atom_pos.z])

                mol_pos = torch.tensor(mol_pos)

                pocket_center_atom = torch.cdist(mol_pos, mol_pos).mean(1).argmin()
                pg.pocket_mask = get_pocket_mask(pg, mol_pos[pocket_center_atom])
                pg.name = pdb_id
                torch.save(pg, pg_file)


if __name__ == "__main__":
    main()
