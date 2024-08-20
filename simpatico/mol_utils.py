import re
import torch
import numpy as np
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import AllChem, AddHs
from torch_geometric.utils import subgraph, to_undirected
from torch_geometric.data import Batch
from from rdkit.Chem.rdchem import Mol
from torch import Tensor
from typing import List, Tuple, Optional

from simpatico import config
from simpatico.utils import to_onehot, get_k_hop_edges


def get_mol_atom_features(m: Mol, atom_vocab: List[str]) -> torch.Tensor:
    '''
    Generates one-hot feature tensor to reflect atomic symbol of each atom.
    Args:
        m (Mol): rdkit Molecule object.
        atom_vocab (List[str]): List of possible atomic symbols. 
    Returns:
       torch.Tensor: (M,N) shaped pytorch tensor where M is number of atoms and N is length of atom vocab. 
    '''
    atom_count = m.GetNumAtoms()

    # placeholder for one-hot rows
    x = [0 for _ in range(atom_count)]

    for atom in m.GetAtoms():
        atom_idx = atom.GetIdx()
        # produce list of 0s except for 1 at index corresponding to atomic symbol 
        onehot_row = to_onehot(atom.GetSymbol(), atom_vocab) 
        x[atom_idx] = onehot_row

    return torch.tensor(x)


def get_mol_pos(m: Mol) -> torch.Tensor:
    '''
    Get xyz coordinates for each atom in rdkit Molecule object.
    Args:
        m (Mol): rdkit Molecule object.
    Returns:
        torch.Tensor: (M,3) shaped pytorch tensor.
    '''
    atom_count = m.GetNumAtoms()
    # retrieve 3D coordinates describing molecular conformer
    conformer = m.GetConformer()
    # placeholder for positional data
    pos = [0 for _ in range(atom_count)]

    for atom in m.GetAtoms():
        atom_idx = atom.GetIdx()
        atom_pos = conformer.GetAtomPosition(atom_idx)
        pos[atom_idx] = [atom_pos.x, atom_pos.y, atom_pos.z]

    return torch.tensor(pos)


def get_mol_edges(m: Mol, k: int=3) -> Tuple[torch.Tensor, torch.Tensor]:
    '''
    Get pyg graph style molecular edge index.
    Args:
        m (Mol): rdkit Molecule object.
        k (int): featured edges will be generated for any atoms k bonds away from each other.
    Returns:
        Tuple[torch.Tensor, Torch.Tensor]: 
            (2, EDGES) shaped edge index tensor and (EDGES, k) shaped feature tensor.
    '''
    edge_index = [[], []]

    # get covalent bonds from list of (from_idx, to_idx) bond descriptions
    for b in m.GetBonds():
        edge_index[0].append(b.GetBeginAtomIdx())
        edge_index[1].append(b.GetEndAtomIdx())

    # 'get_k_hop_edges' accepts an edge index pytorch tensor
    edge_index_tensor = torch.tensor(edge_index)

    # generate edges and edge features for atoms k covalent bonds away from each other
    final_edge_index, final_edge_attr = get_k_hop_edges(edge_index_tensor)

    return final_edge_index, final_edge_attr


def get_H_counts(x: torch.Tensor, 
                 edge_index: torch.Tensor,
                 atom_vocab: List[str]
                 ) -> torch.Tensor:
    '''
    Generates one-hot feature tensor to reflect number of hydrogens attached to each atom.
    Args:
        x (torch.Tensor): one-hot feature vector reflecting atomic symbol of each atom for molecular graph 
        edge_index (torch.Tensor): pyg style edge index reflecting covalent bonds (k-hop edges must be trimmed)
        atom_vocab (List[str]): List of possible atomic symbols. 
    Returns:
        Tuple[torch.Tensor, Torch.Tensor]: 
            (2, EDGES) shaped edge index tensor and (EDGES, k) shaped feature tensor.
    '''
    # one-hot vocab for possible number of hydrogens connected to single atoms
    H_count_vocab = [v for v in range(4)] 

    # initialize H count feature vector
    # for each row in x, generate row of zeros equal in length to H_count_vocab
    H_count_features = torch.zeros(x.size(0), len(H_count_vocab))

    # use undirected_graph to remove any duplicate edges and standardize edge index
    ei = to_undirected(edge_index)
    H_idx = atom_vocab.index("H")

    # get indices of hydrogen atoms in x
    H_atom_index = torch.where(x[:, H_idx] == 1)[0]

    # get indices of all edges where hydrogen is the sink node (heavy-to-hydrogen edges)
    H_node_sinks = torch.where(
        # produce heavy-to-hydrogen mask by mapping each sink value as follows:
        # if idx value is in list of hydrogen indices (H_atom_index), return True
        # otherwise return False
        (ei[1].unsqueeze(1) == H_atom_index.unsqueeze(0)).any(dim=1)
    )[0]

    heavy_to_hydrogen_edges = ei[:, H_node_sinks]

    # instance count of heavy atom index in heavy_to_hydrogen_edges reflects its H count
    H_neighbors, H_counts = heavy_to_hydrogen_edges[0].unique(return_counts=True)

    # update one hot to reflect number of hydrogens connected to each heavy atom
    H_count_features[H_neighbors.int(), H_counts.int()] = 1
    return H_count_features


def mol2pyg(m, removeHs=True):
    try:
        m = AddHs(m)
    except:
        return None

    mol_atom_vocab = config.get('mol_atom_vocab')
    x = get_mol_atom_features(m, mol_atom_vocab)
    pos = get_mol_pos(m)
    edge_index, edge_attr = get_mol_edges(m)
    covalent_index = torch.where(edge_attr[:,0]==1)[0]
    covalent_edge_index = edge_index[:, covalent_index]
    H_count_features = get_H_counts(x, covalent_index, mol_atom_vocab)
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
