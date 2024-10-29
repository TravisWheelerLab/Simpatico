# scripts/train.py
import sys
import argparse
import torch
from typing import List, Tuple, Optional
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import radius
from simpatico.utils.data_utils import (
    ProteinLigandDataLoader,
    TrainingOutputHandler,
)
from simpatico.models.molecule_encoder.MolEncoder import MolEncoder
from simpatico.models.protein_encoder.ProteinEncoder import ProteinEncoder
from simpatico.models import MolEncoderDefaults, ProteinEncoderDefaults
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
        "c",
        "--center",
        type=str,
        help="Protein pocket center coordinates in the form X,Y,Z e.g '-c 20.3,-12,10.25",
    )
    parser.add_argument("-r", "--radius", type=float, help="Protein pocket radius")
    parser.add_argument("-b", "--batch_size", type=int, default=1024)

    return parser.parse_args()


def main(args):
    device = args.device

    if args.protein:
        encoder = ProteinEncoder(**ProteinEncoderDefaults).to(device)
        weight_key = "protein_encoder"

    elif args.molecule:
        encoder = MolEncoder(**MolEncoderDefaults).to(device)
        weight_key = "mol_encoder"

    encoder.load_state_dict(
        torch.load(args.weight_location, map_location=device)[weight_key]
    )

    encoder.eval()

    graph_files = glob(args.data_path)

    for gf in graph_files:
        outfile = (
            args.output_path + "/" + gf.split("/")[-1].split(".")[0] + "_embeds.pyg"
        )
        input_g = torch.load(gf)

        if args.protein:
            input_g = Batch.from_data_list([input_g])

        data_loader = DataLoader(input_g, batch_size=args.batch_size, shuffle=False)
        data_out = []

        for batch in data_loader:
            with torch.no_grad():
                embeds = encoder(batch.to(device))

            if args.molecule:
                batch.x = embeds
                data_out += batch.cpu().to_data_list()

            if args.protein:
                data_out.append(
                    Data(x=embeds[0].cpu(), pos=embeds[1].cpu(), batch=embeds[2].cpu())
                )

        embeds_out = Batch.from_data_list(data_out)
        torch.save(embeds_out, outfile)
        print(outfile)


if __name__ == "__main__":
    args = get_args()
    main(args)
