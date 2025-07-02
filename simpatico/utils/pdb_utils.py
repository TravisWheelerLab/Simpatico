import re
from os import path
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
    atom_name = slice(12, 16)
    res_name = slice(17, 20)

    x_pos = slice(30, 38)
    y_pos = slice(38, 46)
    z_pos = slice(46, 54)

    onehot = []
    pos = []

    # Get one-hot feature vectors given corresponding feature vocab
    onehot += to_onehot(line[atom_name].strip(), atom_vocab)
    onehot += to_onehot(line[res_name].strip(), res_vocab)

    for pos_slice in [x_pos, y_pos, z_pos]:
        pos_val = float(line[pos_slice].strip())
        pos.append(pos_val)

    return onehot, pos


def pdb2pyg(pdb_path: str, ligand_pos=None, pocket_coords=None) -> Data:
    """
    Converts PDB file to PyG graph
    Args:
        pdb_path (str): path to PDB file to be converted
    Returns:
        Data: PyG graph object
    """
    graph_x = []
    graph_pos = []

    # Use the filename (without extension) as graph name
    pdb_name = path.splitext(path.basename(pdb_path))[0]

    with open(pdb_path) as pdb_in:
        for line in pdb_in:
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

    g = Data(
        x=torch.tensor(graph_x),
        pos=torch.tensor(graph_pos),
        name=pdb_name,
        source=pdb_path,
    )

    if ligand_pos is not None:
        proximal_atoms = radius(ligand_pos, g.pos, 12.5)[0].unique()
        g.x = g.x[proximal_atoms]
        g.pos = g.pos[proximal_atoms]

    if pocket_coords is not None:
        pocket_mask = torch.zeros(len(g.x)).bool()

        close_enough = radius(pocket_coords, g.pos, 5)[0].unique()
        pocket_mask[close_enough] = True

        too_close = radius(pocket_coords, g.pos, 2)[0].unique()
        pocket_mask[too_close] = False

        g.pocket_mask = pocket_mask

    return g
