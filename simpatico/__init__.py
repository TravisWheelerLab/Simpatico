import json
import os
from .models.molecule_encoder.MolEncoder import MolEncoder
from .models.protein_encoder.ProteinEncoder import ProteinEncoder
import importlib.resources as pkg_resources
import simpatico

with pkg_resources.open_text("simpatico", "config.json") as f:
    config = json.load(f)

with pkg_resources.path(
    "simpatico.models.weights", config["default_weights"]
) as weight_path:
    config["default_weights_path"] = str(weight_path)
