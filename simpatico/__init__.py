import json
import os
import importlib.resources as pkg_resources
import warnings
import logging
from Bio import BiopythonWarning
from Bio.PDB.PDBExceptions import PDBConstructionWarning

# Suppress *only* PDBConstructionWarnings
warnings.filterwarnings("ignore", category=PDBConstructionWarning)

# Suppress PyG warnings
warnings.filterwarnings("ignore", module="torch_geometric")

# Suppress FAISS info logs
logging.getLogger("faiss").setLevel(logging.ERROR)

with pkg_resources.open_text("simpatico", "config.json") as f:
    config = json.load(f)

with pkg_resources.path(
    "simpatico.models.weights", config["default_weights"]
) as weight_path:
    config["default_weights_path"] = str(weight_path)

from .models.molecule_encoder.MolEncoder import MolEncoder
from .models.protein_encoder.ProteinEncoder import ProteinEncoder