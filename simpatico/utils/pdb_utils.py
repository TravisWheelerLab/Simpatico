import re
import tempfile
from os import path
import sys
import torch
from torch_geometric.data import Data
from torch_geometric.nn import radius
from simpatico import config
from simpatico.utils.utils import to_onehot
from simpatico.utils.mol_utils import molfile2pyg, mol2pyg
from typing import List, Tuple, Optional
from collections import defaultdict
from Bio.PDB import PDBParser, MMCIFParser, is_aa, PDBIO
from rdkit import Chem
from rdkit.Chem.rdDetermineBonds import DetermineConnectivity




def include_residues(protein_graph, selection):
    """
    Expands selection of protein atoms to include all atoms from residues with atoms in the selection. 
    Args:
        protein_graph (PyG Graph): PyG graph of protein/pocket.
        selection (torch.tensor): 1D tensor of selected atoms from protein_graph.
    Returns:
        (torch.tensor): updated 1D selection tensor to include all atoms from incident residues. 
    """
    residue_keys = torch.vstack((protein_graph.residue[selection], 
                                 protein_graph.chain[selection])).T.unique(dim=0)

    all_keys = torch.vstack((protein_graph.residue, 
                             protein_graph.chain)).T

    mask = (all_keys[:, None, :] == residue_keys[None, :, :]).all(dim=2) 
    matches = mask.any(dim=1) 
    indices = torch.nonzero(matches, as_tuple=True)[0]
    return indices

def trim_protein_graph(protein_graph, target_pos, r=12.5):
    proximal_atoms = radius(target_pos, protein_graph.pos, 12.5)[0].unique()
    proximal_atoms = include_residues(protein_graph, proximal_atoms)

    protein_graph.x = protein_graph.x[proximal_atoms]
    protein_graph.pos = protein_graph.pos[proximal_atoms]
    protein_graph.residue = protein_graph.residue[proximal_atoms]
    protein_graph.chain = protein_graph.chain[proximal_atoms]
    return protein_graph

def get_atom_features(atom) -> Tuple[List[float], List[float]]:
    """
    Takes a Bio.PDB.Atom object from a Bio.PDB.Structure and returns one-hot feature and positional tensors for PyG graphs.
    Args:
        atom (Bio.PDB.Atom): atom object from structure.
    Returns:
        Tuple[List[float], List[float]]: Per-atom one-hot feature and positional tensors
    """
    atom_vocab = config.get("atom_vocab")
    res_vocab = config.get("res_vocab")
    
    onehot = []
    pos = atom.get_coord()

    residue = atom.parent

    # Get one-hot feature vectors given corresponding feature vocab
    onehot += to_onehot(atom.get_name(), atom_vocab)
    onehot += to_onehot(residue.get_resname(), res_vocab)

    chain_val = residue.parent.id
    residue_number = residue.id[1]

    return onehot, pos, chain_val, residue_number

def pdb2pyg(pdb_path, ligand_pos=None, pocket_coords=None) -> Data:
    """
    Converts PDB file to PyG graph
    Args:
        pdb_path (str): path to PDB file to be converted
    Returns:
        Data: PyG graph object
    """
    if isinstance(pdb_path, str):
        pdb_structure = pdb_to_bio(pdb_path)
    else:
        pdb_structure = pdb_path
        pdb_path = pdb_structure.id

    pdb_name = path.splitext(path.basename(pdb_path))[0]

    graph_x = []
    graph_pos = []
    chain_vals = []
    res_numbers = []

    for atom in pdb_structure.get_atoms():
        if atom.element == 'H':
            continue
        if is_aa(atom.get_parent()) == False:
            continue

        x, pos, chain, res_number = get_atom_features(atom)

        graph_x.append(x)
        graph_pos.append(pos)
        chain_vals.append(chain)
        res_numbers.append(res_number)

    
    # with open(pdb_path) as pdb_in:
    #     for line in pdb_in:
    #         # Only interested in ATOM lines
    #         if line[0:4] == "ATOM":
    #             # Skip hydrogens.
    #             if re.match(r"^(\d+H|H)", line[12:16].strip()):
    #                 continue
    #             if line[76:78].strip() == "H":
    #                 continue
    #             else:
    #                 x, pos, chain, res_number = get_pdb_line_data(line)
    #                 x, pos, chain, res_number = get_atom_features(atom)
    #                 graph_x.append(x)
    #                 graph_pos.append(pos)
    #                 chain_vals.append(chain)
    #                 res_numbers.append(res_number)
    
    unique_chain_vals = list(set(chain_vals))
    chain_vals = [unique_chain_vals.index(x) for x in chain_vals]

    g = Data(
        x=torch.tensor(graph_x),
        pos=torch.tensor(graph_pos),
        residue=torch.tensor(res_numbers).int(),
        chain=torch.tensor(chain_vals).int(),
        name=pdb_name,
        source=pdb_path,
    )

    if ligand_pos is not None:
        g = trim_protein_graph(g, ligand_pos)

    if pocket_coords is not None:
        pocket_mask = torch.zeros(len(g.x)).bool()

        close_enough = radius(pocket_coords, g.pos, 5)[0].unique()
        pocket_mask[close_enough] = True

        too_close = radius(pocket_coords, g.pos, 2)[0].unique()
        pocket_mask[too_close] = False

        g.pocket_mask = pocket_mask

    return g

def pdb_to_bio(pdb_file):
    pdb_filetype = path.splitext(path.basename(pdb_file))[1]

    if pdb_filetype == '.pdb':
        parser = PDBParser()
    else:
        parser = MMCIFParser()

    # Use the filename (without extension) as graph name
    pdb_name = path.splitext(path.basename(pdb_file))[0]
    pdb_data = parser.get_structure(pdb_name, pdb_file)
    return pdb_data



def extract_ligands(pdb_path: str, ligand_id: str):
    ligand_blocks = defaultdict(list)
    ligand_graphs = []

    atom_type_slice = slice(0,6)
    res_name_slice = slice(17,20)
    res_chain_slice = 21
    res_number_slice = slice(22,26)

    with open(pdb_path, 'r') as pdb_in:
        pdb_id = pdb_path.split('/')[-1].split('.')[0]

        for line in pdb_in:
            atom_type = line[atom_type_slice]

            if atom_type != 'HETATM':
                continue

            res_name = line[res_name_slice].strip()
            res_number = line[res_number_slice].strip()
            res_chain = line[res_chain_slice]

                        
            if res_name[-len(ligand_id):] == ligand_id:
                ligand_blocks[res_chain + res_number.strip()].append(line)
        
    if len(ligand_blocks) == 0:
        return ligand_graphs 

    for k,v in ligand_blocks.items():
        with open('temp_ligand.pdb', 'w') as ligand_out:
            ligand_out.write(''.join(v))
        
        try:
            mol_g = molfile2pyg('temp_ligand.pdb')
        except Exception as e:
            mol_g = None

        if mol_g is None:
            continue

        mol_g = mol_g.to_data_list()[0]
        mol_g.name = f"{pdb_id}_{ligand_id}_{k}"
        ligand_graphs.append(mol_g)

    return ligand_graphs 

def extract_ligands(pdb_path: str, ligand_id: str):
    if isinstance(pdb_path, str):
        pdb_structure = pdb_to_bio(pdb_path)
    else:
        pdb_structure = pdb_path
        pdb_path = pdb_structure.id

    pdb_filename = pdb_path.split('/')[-1].split('.')[0]

    ligand_graphs = []

    for r_i, res in enumerate(pdb_structure.get_residues()):
        if is_aa(res):
            continue
            
        if res.get_resname() == ligand_id:
            mol = residue_to_rdkit(res)

            if mol is None:
                continue

            g = mol2pyg(mol)

            if g is not None:
                g.name = f'{pdb_filename}_{ligand_id}_{res.parent.id}_{res.id[1]}_{r_i}'
                ligand_graphs.append(g)
                    
    return ligand_graphs

def residue_to_rdkit(residue):
    no_carbon = True 
    mol = Chem.RWMol()
    atom_map = {}

    # Add atoms
    for atom in residue.get_atoms():
        atom_element = atom.element.capitalize()
        if atom_element == 'C':
            no_carbon = False 
        idx = mol.AddAtom(Chem.Atom(atom_element))
        atom_map[atom] = idx
    
    if no_carbon:
        'No carbon present in ligand.'
        return None

    # Add coordinates
    conf = Chem.Conformer(len(atom_map))
    for atom, idx in atom_map.items():
        x, y, z = atom.coord
        conf.SetAtomPosition(idx, (float(x), float(y), float(z)))
    mol.AddConformer(conf)
    try:
        DetermineConnectivity(mol)
    except:
        print("could not assign bonds.")
        return None
   
    # Guess bonds
    # rdDetermineBonds.DetermineConnectivity()
    # rdmolops.SanitizeMol(mol, catchErrors=True)

    return mol


