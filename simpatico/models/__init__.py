# models/__init__.py
import json
import os

# Define the path to the JSON file
mol_hparam_path = os.path.join(
    os.path.dirname(__file__), "molecule_encoder/MolEncoder.json"
)
protein_hparam_path = os.path.join(
    os.path.dirname(__file__), "protein_encoder/ProteinEncoder.json"
)

with open(mol_hparam_path, "r") as file:
    MolEncoderDefaults = json.load(file)

with open(protein_hparam_path, "r") as file:
    ProteinEncoderDefaults = json.load(file)
