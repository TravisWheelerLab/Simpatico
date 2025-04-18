import re
import sys
import torch
from torch_geometric.data import Data
from torch_geometric.nn import radius
from simpatico import config
from simpatico.utils.utils import to_onehot
from typing import List, Tuple, Optional


def get_pdb_line_data(line: str) -> Tuple[List[float], List[float]]:
    """
    Takes an ATOM line from PDB file and returns one-hot feature and positional tensors for PyG graphs.
    Args:
        line (str): Line from PDB file
    Returns:
        Tuple[List[float], List[float]]: Per-atom one-hot feature and positional tensors
    """
    atom_vocab = config.get("atom_vocab")
    res_vocab = config.get("res_vocab")

    # Specify the (start,stop) indices of features to extract from PDB line
    atom_name = (12, 16)
    res_name = (17, 20)

    x_pos = (30, 38)
    y_pos = (38, 46)
    z_pos = (46, 54)

    onehot = []
    pos = []

    # Get one-hot feature vectors given corresponding feature vocab
    for f, v in ([atom_name, atom_vocab], [res_name, res_vocab]):
        feature_val = line[f[0] : f[1]].strip()
        onehot += to_onehot(feature_val, v)

    for p in [x_pos, y_pos, z_pos]:
        pos_val = float(line[p[0] : p[1]].strip())
        pos.append(pos_val)

    return onehot, pos


def pdb2pyg(pdb_path: str, pocket_ligand=None) -> Data:
    """
    Converts PDB file int PyG graph
    Args:
        pdb_path (str): path to PDB file to be converted
    Returns:
        Data: PyG graph object
    """
    graph_x = []
    graph_pos = []
    pocket_pos = []

    # Use the filename (without extension) as graph name
    pdb_name = pdb_path.split("/")[-1].split(".")[0]

    with open(pdb_path) as pdb_in:
        for line in pdb_in:
            if pocket_ligand is not None:
                if line[0:6] == "HETATM":
                    if line[17:20].strip() == pocket_ligand:
                        _, pos = get_pdb_line_data(line)
                        pocket_pos.append(pos)

            # Only interested in ATOM lines
            if line[0:4] == "ATOM":
                # Skip hydrogens.
                if re.match(r"^(\d+H|H)", line[12:16].strip()):
                    continue
                if line[76:78].strip() == "H":
                    continue
                else:
                    x, pos = get_pdb_line_data(line)
                    graph_x.append(x)
                    graph_pos.append(pos)

    g = Data(x=torch.tensor(graph_x), pos=torch.tensor(graph_pos), name=pdb_name)

    if pocket_ligand is not None:
        pocket_pos = torch.tensor(pocket_pos)
        pocket_mask = torch.zeros(g.x.size(0)).bool()
        proximal_atoms, _ = radius(pocket_pos, g.pos, 5)
        pocket_mask[proximal_atoms] = True
        g.pocket_mask = pocket_mask

    return g
