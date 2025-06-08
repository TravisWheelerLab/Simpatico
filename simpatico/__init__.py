import json
import os
from .models.molecule_encoder.MolEncoder import MolEncoder
from .models.protein_encoder.ProteinEncoder import ProteinEncoder
import importlib.resources as pkg_resources

with pkg_resources.open_text("simpatico.data", "config.json") as f:
    config = json.load(f)
