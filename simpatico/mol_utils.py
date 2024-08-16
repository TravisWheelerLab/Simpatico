import re
import torch
import numpy as np
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import AllChem, AddHs
from torch_geometric.utils import subgraph, to_undirected
from torch_geometric.data import Batch

from simpatico import config
from simpatico.utils import to_onehot, get_k_hop_neighborhoods


def get_mol_atom_features(m):
    atom_count = m.getNumAtoms()


def mol2pyg(m):
    try:
        m = AddHs(m)
    except:
        return None

    x = get_mol_atom_features(m)


def molfile2pyg(m_file, get_pos=True, k=3):
    filename, filetype = m_file.split("/")[-1].split(".")

    if filetype == "sdf":
        mols = Chem.SDMolSupplier(m_file, sanitize=False)

    if filetype == "pdb":
        mols = [Chem.MolFromPDBFile(m_file, sanitize=False)]

    k_hop_vocab = [x + 1 for x in range(k)]
    mol_batch = []

    for m_i, m in enumerate(mols):
        try:
            m = AddHs(m)
        except:
            continue

        atom_count = m.GetNumAtoms()
        am = Chem.GetAdjacencyMatrix(m)

        if get_pos:
            conformer = m.GetConformer()
            pos = [[] for _ in range(atom_count)]

        x = [[] for _ in range(atom_count)]
        atom_H_count = [0 for _ in range(atom_count)]
        heavy_atoms = [0 for _ in range(atom_count)]

        edge_index = [[], []]
        edge_attr = []

        for b in m.GetBonds():
            edge_index[0].append(b.GetBeginAtomIdx())
            edge_index[1].append(b.GetEndAtomIdx())

        for atom in m.GetAtoms():
            a_i = atom.GetIdx()

            if get_pos:
                apos = conformer.GetAtomPosition(a_i)
                pos[a_i] = [apos.x, apos.y, apos.z]

            x[a_i] = to_onehot(atom.GetSymbol(), config.get("mol_atom_vocab"))

            if atom.GetSymbol() == "H":
                atom_H_count[list(am[a_i]).index(1)] += 1
            else:
                heavy_atoms[a_i] = 1

        for a_i in range(atom_count):
            x[a_i] += to_onehot(atom_H_count[a_i], config.get("H_count_vocab"))

        heavy_atoms = torch.where(torch.tensor(heavy_atoms) == 1)[0]

        x = torch.tensor(x)[heavy_atoms]

        edge_index, edge_attr = subgraph(
            heavy_atoms,
            torch.tensor(edge_index),
            relabel_nodes=True,
        )

        edge_index = to_undirected(edge_index)

        # For now, throw away bond type information on edges.
        final_edge_index = torch.tensor([[], []])
        final_edge_attr = []

        for atom_idx in torch.unique(edge_index.flatten()):
            for k_i, k_neighborhood in enumerate(
                get_k_hop_neighborhoods(atom_idx.item(), k, edge_index)
            ):
                final_edge_attr += [
                    to_onehot(k_i + 1, k_hop_vocab)
                    for _ in range(k_neighborhood.size(0))
                ]
                final_edge_index = torch.hstack(
                    (
                        final_edge_index,
                        torch.vstack(
                            (atom_idx.repeat(k_neighborhood.size(0)), k_neighborhood)
                        ),
                    )
                )

        final_edge_attr = torch.tensor(final_edge_attr)

        final_edge_index, final_edge_attr = to_undirected(
            final_edge_index, final_edge_attr, reduce="max"
        )

        g = Data(
            x=x,
            edge_index=final_edge_index,
            edge_attr=final_edge_attr,
            name=(filename + f"_{m_i}"),
        )

        if get_pos:
            g.pos = torch.tensor(pos)[heavy_atoms]

        mol_batch.append(g)

    if len(mol_batch) == 0:
        return None

    return Batch.from_data_list(mol_batch)
