import re
from copy import deepcopy
import torch
import numpy as np
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import AllChem, AddHs
from torch_geometric.utils import subgraph, to_undirected
from torch_geometric.data import Batch
from rdkit.Chem.rdchem import Mol
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from torch import Tensor
from typing import List, Tuple, Optional
from molvs import Standardizer

from simpatico import config
from simpatico.utils.utils import to_onehot, get_k_hop_edges


def get_mol_atom_features(m: Mol, atom_vocab: List[str]) -> torch.Tensor:
    """
    Generates one-hot feature tensor to reflect atomic symbol of each atom.
    Args:
        m (Mol): rdkit Molecule object.
        atom_vocab (List[str]): List of possible atomic symbols.
    Returns:
       torch.Tensor: (M,N) shaped pytorch tensor where M is number of atoms and N is length of atom vocab.
    """
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
    """
    Get xyz coordinates for each atom in rdkit Molecule object.
    Args:
        m (Mol): rdkit Molecule object.
    Returns:
        torch.Tensor: (M,3) shaped pytorch tensor.
    """
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


def get_mol_edges(m: Mol, k: int = 3) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get pyg graph style molecular edge index.
    Args:
        m (Mol): rdkit Molecule object.
        k (int): featured edges will be generated for any atoms k bonds away from each other.
    Returns:
        Tuple[torch.Tensor, Torch.Tensor]:
            (2, EDGES) shaped edge index tensor and (EDGES, k) shaped feature tensor.
    """
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


def get_H_counts(
    x: torch.Tensor, edge_index: torch.Tensor, atom_vocab: List[str]
) -> torch.Tensor:
    """
    Generates one-hot feature tensor to reflect number of hydrogens attached to each atom.
    Args:
        x (torch.Tensor): one-hot feature vector reflecting atomic symbol of each atom for molecular graph
        edge_index (torch.Tensor): pyg style edge index reflecting covalent bonds (k-hop edges must be trimmed)
        atom_vocab (List[str]): List of possible atomic symbols.
    Returns:
        Tuple[torch.Tensor, Torch.Tensor]:
            (2, EDGES) shaped edge index tensor and (EDGES, k) shaped feature tensor.
    """
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
    H_count_features[H_neighbors.long(), H_counts.long()] = 1
    # For atoms not connect to a hydrogen, set first index of onehot to 1 to indicate zero Hs
    H_count_features[H_count_features.sum(1) == 0, 0] = 1
    return H_count_features


def mol2pyg(
    m: Mol, ignore_pos: bool = False, removeHs: bool = True, get_scaffold=True
) -> Optional[Data]:
    """
    Converts an RDKit Mol object into a PyG Data object representing the molecular graph.
    Args:
        m (Mol): RDKit Molecule object.
        removeHs (bool): If True, hydrogen atoms are removed from the final graph.
    Returns:
        Optional[Data]: PyG Data object containing the graph representation of the molecule, or None if conversion fails.
    """
    standardizer = Standardizer()

    # If we cannot standardize molecule, we won't be able to extract the scaffold
    try:
        m = standardizer.standardize(m)
    except Exception as e:
        get_scaffold = False

    # Attempt to add hydrogen atoms so that we are working with a standardized molecule going forward
    try:
        m = AddHs(m)
    except:
        print("Could not add hydrogens")
        return None

    if get_scaffold:
        try:
            scaffold = MurckoScaffold.GetScaffoldForMol(m)
            scaffold_atoms = m.GetSubstructMatch(scaffold)

            if len(scaffold_atoms) == 0:
                get_scaffold = False
            else:
                scaffold_atoms = torch.tensor(list(scaffold_atoms))
        except Exception as e:
            print(e)
            get_scaffold = False

    mol_atom_vocab = config.get("mol_atom_vocab")

    # Generate one-hot encoded atom features
    atom_species_onehots = get_mol_atom_features(m, mol_atom_vocab)

    if get_scaffold:
        scaffold_mask = torch.zeros(atom_species_onehots.size(0)).bool()
        scaffold_mask[scaffold_atoms] = True

    if ignore_pos is False:
        # Get the 3D coordinates of atoms in the molecule
        pos = get_mol_pos(m)

    # Generate edge indices and edge attributes for the molecular graph
    edge_index, edge_attr = get_mol_edges(m)

    if edge_index.size(1) == 0:
        return None

    # Identify covalent bonds (edges where k = 1)
    covalent_index = torch.where(edge_attr[:, 0] == 1)[0]
    covalent_edge_index = edge_index[:, covalent_index]

    # Generate features representing the number of hydrogens attached to each atom
    H_count_features = get_H_counts(
        atom_species_onehots, covalent_edge_index, mol_atom_vocab
    )

    # Concatenate the hydrogen count features to the atomc species onehots
    x = torch.hstack((atom_species_onehots, H_count_features))

    if removeHs:
        H_idx = mol_atom_vocab.index("H")
        # Get indices of non-hydrogen (heavy) atoms
        heavy_atom_index = torch.where(x[:, H_idx] != 1)[0].long()

        try:
            # Remove hydrogen-adjacent edges
            edge_index, edge_attr = subgraph(
                heavy_atom_index, edge_index, edge_attr, relabel_nodes=True
            )
            x = x[heavy_atom_index]
        except:
            return None

        if get_scaffold:
            scaffold_mask = scaffold_mask[heavy_atom_index]

        if ignore_pos is False:
            pos = pos[heavy_atom_index]

    edge_index, edge_attr = to_undirected(edge_index, edge_attr, reduce="mean")

    # Create a PyG Data object representing the molecular graph
    mol_graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    if get_scaffold:
        mol_graph.scaffold = scaffold_mask

    if ignore_pos is False:
        mol_graph.pos = pos

    return mol_graph


def molfile2pyg(m_file: str, get_pos: bool = True, k: int = 3) -> Optional[Batch]:
    """
    Converts a molecular file (e.g., SDF, PDB) into a batch of molecular PyG graphs.
    Args:
        m_file (str): Path to the molecular file.
        get_pos (bool): If True, 3D coordinates of atoms are included in the graph.
        k (int): Maximum number of covalent bonds between two atoms for which edges are generated.
    Returns:
        Optional[Batch]: A batch of PyG Data objects, or None if conversion fails.
    """
    # Extract the filename and filetype from the input file path
    filename, filetype = m_file.split("/")[-1].split(".")
    ignore_pos = False

    # Use appropriate RDKit method for generating Molecule object from file
    if filetype == "sdf":
        mols = Chem.SDMolSupplier(m_file, sanitize=False)

    elif filetype == "pdb":
        mols = [Chem.MolFromPDBFile(m_file, sanitize=False)]

    elif filetype in ["ism", "smi"]:
        mols = Chem.SmilesMolSupplier(m_file, sanitize=False)
        ignore_pos = True

    elif filetype == "mol2":
        mols = [Chem.MolFromMol2File(m_file, sanitize=False)]

    # List to store individual PyG graph objects
    mol_batch = []

    for m_i, m in enumerate(mols):
        # Convert each molecule to a PyG graph
        mg = mol2pyg(m, ignore_pos)

        if mg is None:
            # Skip molecules that failed to convert
            continue
        else:
            # Assign a name to the graph based on the file name and molecule index
            mg.name = filename + f"_{m_i}"
            mol_batch.append(mg)

    if len(mol_batch) == 0:
        # Return None if no valid graphs were generated
        return None

    # Create a batch of PyG Data objects
    return Batch.from_data_list(mol_batch)
