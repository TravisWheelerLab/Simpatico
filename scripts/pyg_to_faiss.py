import torch
import pickle
import os
import numpy as np
import faiss
import argparse
from glob import glob


def get_args():
    parser = argparse.ArgumentParser(
        description="Convert PyG outputs of eval script into faiss index files."
    )

    parser.add_argument(
        "-i",
        "--input_path",
        type=str,
        help='Path to input file. May use unix style pathname patterns if enclosed in quotes. e.g. "/path/to/directory/*.pyg".',
    )
    parser.add_argument("-o", "--output_path")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for training",
    )
    return parser.parse_args()


def main(args):
    input_files = glob(args.input_path)
    output_path = args.output_path
    os.makedirs(output_path, exist_ok=True)

    for pyg_f in input_files:
        pyg_graph = torch.load(pyg_f, weights_only=False)
        atom_embeds = pyg_graph.x.detach().cpu().numpy().astype("float32")

        dim = atom_embeds.shape[1]
        index = faiss.IndexFlatL2(dim)

        # if args.device == "cuda":
        #     res = faiss.StandardGpuResources()
        #     index = faiss.index_cpu_to_gpu(res, 0, index)

        index.add(atom_embeds)
        basename, _ = os.path.splitext(os.path.basename(pyg_f))
        index_path_out = f"{output_path}/{basename}.faiss"
        batch_path_out = f"{output_path}/{basename}.batch"
        faiss.write_index(index, index_path_out)

        with open(batch_path_out, "wb") as batch_out:
            pickle.dump(pyg_graph.batch.cpu(), batch_out)


if __name__ == "__main__":
    args = get_args()
    main(args)
