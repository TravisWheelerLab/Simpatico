import json
import os
from .models.molecule_encoder.MolEncoder import MolEncoder
from .models.protein_encoder.ProteinEncoder import ProteinEncoder
import importlib.resources as pkg_resources
from . import data  # assuming config.json is in simpatico/data/

with pkg_resources.open_text(data, "config.json") as f:
    config = json.load(f)
