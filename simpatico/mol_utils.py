import re
import torch
import numpy as np
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import AllChem, AddHs
from torch_geometric.utils import subgraph, to_undirected
from torch_geometric.data import Batch

from simpatico import config
from simpatico.utils import to_onehot, get_k_hop_edges


def get_mol_atom_features(m):
    atom_count = m.GetNumAtoms()
    x = [[] for _ in range(atom_count)]

    for atom in m.GetAtoms():
        a_i = atom.GetIdx()
        x[a_i] = to_onehot(atom.GetSymbol(), config.get("mol_atom_vocab"))

    return torch.tensor(x)


def get_mol_pos(m):
    atom_count = m.GetNumAtoms()
    conformer = m.GetConformer()
    pos = [[] for _ in range(atom_count)]

    for atom in m.GetAtoms():
        a_i = atom.GetIdx()
        apos = conformer.GetAtomPosition(a_i)
        pos[a_i] = [apos.x, apos.y, apos.z]

    return torch.tensor(pos)


def get_mol_edges(m, k=3):
    edge_index = [[], []]

    for b in m.GetBonds():
        edge_index[0].append(b.GetBeginAtomIdx())
        edge_index[1].append(b.GetEndAtomIdx())

    edge_index_tensor = torch.tensor(edge_index)
    final_edge_index, final_edge_attr = get_k_hop_edges(edge_index_tensor)

    return final_edge_index, final_edge_attr


def get_H_counts(x, edge_index, edge_attr=None, k_graph=True):
    H_count_features = torch.tensor(
        [[0 for _ in config.get("H_count_vocab")] for _ in x]
    )
    ei = edge_index.clone()
    if k_graph:
        ei = ei[:, torch.where(edge_attr[:, 0] == 1)[0]]
    ei = to_undirected(ei)
    H_idx = config.get("mol_atom_vocab").index("H")
    H_atom_index = torch.where(x[:, H_idx] == 1)[0]
    H_node_sinks = torch.where(
        (H_atom_index.unsqueeze(0) == ei[1].unsqueeze(1)).any(dim=1)
    )[0]
    H_neighbors, H_counts = ei[:, H_node_sinks][0].unique(return_counts=True)
    H_count_features[H_neighbors.int(), H_counts.int()] = 1
    return H_count_features


def mol2pyg(m, removeHs=True):
    try:
        m = AddHs(m)
    except:
        return None

    x = get_mol_atom_features(m)
    pos = get_mol_pos(m)
    edge_index, edge_attr = get_mol_edges(m)
    H_count_features = get_H_counts(x, edge_index, edge_attr)
    x = torch.hstack((x, H_count_features))

    if removeHs:
        H_idx = config.get("mol_atom_vocab").index("H")
        heavy_atom_index = torch.where(x[:, H_idx] != 1)[0]
        edge_index, edge_attr = subgraph(
            heavy_atom_index, edge_index, edge_attr, relabel_nodes=True
        )
        x = x[heavy_atom_index]
        pos = pos[heavy_atom_index]

    mol_graph = Data(x=x, pos=pos, edge_index=edge_index, edge_attr=edge_attr)
    return mol_graph


def molfile2pyg(m_file, get_pos=True, k=3):
    filename, filetype = m_file.split("/")[-1].split(".")

    if filetype == "sdf":
        mols = Chem.SDMolSupplier(m_file, sanitize=False)

    if filetype == "pdb":
        mols = [Chem.MolFromPDBFile(m_file, sanitize=False)]

    mol_batch = []

    for m_i, m in enumerate(mols):
        mg = mol2pyg(m)

        if mg is None:
            continue
        else:
            mg.name = filename + f"_{m_i}"
            mol_batch.append(mg)

    if len(mol_batch) == 0:
        return None

    return Batch.from_data_list(mol_batch)
