# scripts/train.py
import os
from pathlib import Path
import sys
import argparse
import torch
from typing import List, Tuple, Optional
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import radius
from simpatico.utils.mol_utils import molfile2pyg

from simpatico.utils.data_utils import (
    ProteinLigandDataLoader,
    TrainingOutputHandler,
)
from simpatico.models.molecule_encoder.MolEncoder import MolEncoder
from simpatico.models.protein_encoder.ProteinEncoder import ProteinEncoder
from simpatico.models import MolEncoderDefaults, ProteinEncoderDefaults
from simpatico.utils.pdb_utils import pdb2pyg

from typing import Callable
from glob import glob


def get_args():
    parser = argparse.ArgumentParser(description="Training Script")
    parser.add_argument(
        "-d",
        "--data_path",
        type=str,
        help="Path to the dataset",
    )
    parser.add_argument("-w", "--weight_location")
    parser.add_argument("-o", "--output_path")
    parser.add_argument(
        "--no-overwrite", action="store_true", help="Do not overwrite existing files"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "-p", "--protein", action="store_true", help="Get protein atom embeddings."
    )
    group.add_argument(
        "-m",
        "--molecule",
        action="store_true",
        help="Get molecule atom embeddings",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for training",
    )
    parser.add_argument(
        "-c",
        "--center",
        type=str,
        help="Protein pocket center coordinates in the form X,Y,Z e.g '-c 20.3,-12,10.25",
    )
    parser.add_argument("-r", "--radius", type=float, help="Protein pocket radius")
    parser.add_argument("-b", "--batch_size", type=int, default=1024)
    parser.add_argument(
        "--pocket-ligand",
        type=str,
        default=None,
        help="Directory to ligand graphs that describe pocket.",
    )
    parser.add_argument(
        "--pocket-id", type=str, default=None, help="Pocket ligand name"
    )

    return parser.parse_args()


def main(args):
    device = args.device

    if args.protein:
        encoder = ProteinEncoder(**ProteinEncoderDefaults).to(device)
        weight_index = 0

    elif args.molecule:
        encoder = MolEncoder(**MolEncoderDefaults).to(device)
        weight_index = 1

    encoder.load_state_dict(
        torch.load(args.weight_location, map_location=device)[weight_index]
    )

    encoder.eval()

    graph_files = glob(args.data_path)

    if args.pocket_ligand:
        pocket_ligands = glob(args.pocket_ligand)

    for gf in graph_files:
        embed_failed = False
        outfile = (
            args.output_path + "/" + gf.split("/")[-1].split(".")[0] + "_embeds.pyg"
        )

        if args.no_overwrite:
            if os.path.exists(outfile) and os.path.getsize(outfile) > 0:
                continue
            else:
                # create an empty file so parallel jobs know to skip current target
                Path(outfile).touch()

        if gf.split(".")[-1] == "pdb":
            input_g = pdb2pyg(gf, args.pocket_id)
        elif gf.split(".")[-1] in ["sdf", "ism", "smi"]:
            input_g = molfile2pyg(gf, get_pos=True)
        else:
            input_g = torch.load(gf)

        if args.pocket_ligand:
            for lf in pocket_ligands:
                lf_id = lf.split("/")[-1].split("_")[0]

                if gf.split("/")[-1].split("_")[0] == lf_id:
                    if lf.split(".")[-1] == "txt":
                        with open(lf) as lf_txt:
                            pos = []
                            for line in lf_txt:
                                xyz = [float(v) for v in line.split(" ")[:3]]
                                pos.append(xyz)
                        lg = Data(pos=torch.tensor(pos))
                    else:
                        lg = torch.load(lf)

                    pocket_mask = torch.zeros(input_g.x.size(0)).bool()
                    proximal_atoms = radius(lg.pos, input_g.pos, 4)[0].unique()
                    pocket_mask[proximal_atoms] = True
                    input_g.pocket_mask = pocket_mask

        if args.protein:
            input_g = Batch.from_data_list([input_g])

        data_loader = DataLoader(input_g, batch_size=args.batch_size, shuffle=False)
        data_out = []

        for batch in data_loader:
            if args.protein:
                if hasattr(batch, "pocket_mask") == False:
                    print("No pocket mask specified for %s" % gf)

                    if args.no_overwrite:
                        os.remove(outfile)

                    embed_failed = True
                    continue

            with torch.no_grad():
                try:
                    embeds = encoder(batch.to(device))
                except:
                    embed_failed = True
                    continue

            if args.molecule:
                batch.x = embeds
                data_out += batch.cpu().to_data_list()

            if args.protein:
                data_out.append(
                    Data(x=embeds[0].cpu(), pos=embeds[1].cpu(), batch=embeds[2].cpu())
                )

        if embed_failed:
            continue

        embeds_out = Batch.from_data_list(data_out)
        torch.save(embeds_out, outfile)
        print(outfile)


if __name__ == "__main__":
    args = get_args()
    main(args)
