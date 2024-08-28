import json
import os
from .models.molecule_encoder.MolEncoder import MolEncoder
from .models.protein_encoder.ProteinEncoder import ProteinEncoder

config_path = os.path.join(os.path.dirname(__file__), "config.json")

with open(config_path, "r") as config_file:
    config = json.load(config_file)
