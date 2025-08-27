import sys
import random
from os import path
import pickle
import argparse
import torch
from typing import List, Tuple, Optional
from simpatico.utils.data_utils import (
    ProteinLigandDataLoader,
    TrainingOutputHandler,
)
from simpatico.models.molecule_encoder.MolEncoder import MolEncoder
from simpatico.models.protein_encoder.ProteinEncoder import ProteinEncoder
from simpatico.models.molecule_refiner.MolRefiner import MolRefiner
from simpatico.models.protein_refiner.ProteinRefiner import ProteinRefiner
from simpatico.get_train_set import construct_tv_set
from simpatico.models import MolEncoderDefaults, ProteinEncoderDefaults
from torch_geometric.nn import knn
from torch_geometric.data import Batch
from typing import Callable


def add_arguments(parser):
    parser.add_argument('train_type', type=int, help='Train encoders and refiners (0), just encoders (1), or just refiners (2).')
    parser.add_argument(
        "input",
        type=str,
        help="Path to train-eval dataset",
    )
    parser.add_argument('-we',"--encoder-weight-path"),
    parser.add_argument("-wr", "--refiner-weight-path"),
    parser.add_argument("-o", "--output", type=str, help="Model performance output")
    parser.add_argument(
        "-b", "--batch_size", type=int, default=16, help="Input batch size for training"
    )
    parser.add_argument(
        "-e", "--epochs", type=int, default=100, help="Number of epochs to train"
    )
    parser.add_argument(
        "-lr", "--learning_rate", type=float, default=0.0001, help="Learning rate"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for training",
    )
    parser.add_argument(
        "-lew",
        "--load_encoder",
        help="Load previously trained encoder weights",
    )
    parser.add_argument(
        "-lrw",
        "--load_refiner",
        help="Load previously trained refiner weights",
    )
    parser.add_argument("--epoch_start", type=int, default=1)

    parser.set_defaults(main=main)
    return parser


def positive_margin_loss(anchors, positives, negatives, m=1.0, d=3):
    positive_distances = torch.norm(anchors - positives, dim=1)
    anchors = anchors.repeat(negatives.size(0) // anchors.size(0), 1)

    negative_distances = torch.norm(anchors - negatives, dim=1)
    positive_loss = torch.clamp(positive_distances - m, min=0)
    negative_loss = torch.clamp(m * d - negative_distances, min=0)

    return positive_loss.mean() + negative_loss.mean()


def hard_negative_scheduler(target_epoch, target_difficulty):
    def scheduler(epoch):
        d_modifier = min(epoch / target_epoch, 1)
        return 1 - (1 - target_difficulty) * d_modifier

    return scheduler


def logger(output_path):
    def log_text(message):
        if output_path is not None:
            with open(output_path, "a") as f_out:
                f_out.write(f"{message}\n")
        else:
            print(message)

    return log_text

def refinement_loss(p_embeds, p_pos, p_batch, m_embeds, m_pos, m_batch):
    criterion = torch.nn.MSELoss()
    m_index, p_index = knn(p_pos, m_pos, 1, p_batch, m_batch)
    loss = criterion(m_embeds[m_index], p_embeds[p_index])
    return loss

def derangement(n: int):
    if n == 1:
        raise ValueError("No derangement possible for n=1")
    while True:
        perm = torch.randperm(n)
        if not torch.any(perm == torch.arange(n)):
            return perm.tolist()

def training_step(
    training_type, data_loader, protein_encoder, mol_encoder, protein_refiner, mol_refiner, difficulty_value, prot_loss=True
):
    device = next(protein_encoder.parameters()).device

    protein_batch, molecule_batch = data_loader.get_random_batch()
    protein_batch = protein_batch.clone()
    protein_batch.pos += torch.randn_like(protein_batch.pos) * 0.25

    protein_batch = protein_batch.to(device)
    molecule_batch = molecule_batch.to(device)

    if training_type in [2,3]:
        with torch.no_grad():
            protein_out = protein_encoder(protein_batch)
            mol_out = mol_encoder(molecule_batch)
    else:
        protein_out = protein_encoder(protein_batch)
        mol_out = mol_encoder(molecule_batch)

    KV = protein_refiner(*protein_out)

    mbatch = molecule_batch.clone().to(device)
    positive_mol_batch = mbatch.batch.clone()
    mbatch.x = mol_out

    pos_mol_out = mol_refiner(mbatch, KV, protein_out[2])

    mol_graph_list = mbatch.to_data_list()
    neg_graph_batch = Batch.from_data_list([mol_graph_list[i] for i in derangement(len(mol_graph_list))])

    mol_nearest_negatives, prot_nearest_negatives = knn(protein_out[0], neg_graph_batch.x, 1, protein_out[2], neg_graph_batch.batch)
    target_negative_distances = torch.norm(mbatch.x[mol_nearest_negatives] - protein_out[0][prot_nearest_negatives], dim=1)

    negative_mol_out = mol_refiner(mbatch, KV, protein_out[2])
    negative_distances = torch.norm(negative_mol_out[mol_nearest_negatives] - protein_out[0][prot_nearest_negatives], dim=1)

    neg_distance_criterion = torch.nn.MSELoss()

    # If ligand is not active, refinement should hold embedding values stationary
    l2 = neg_distance_criterion(negative_distances, target_negative_distances)

    
    l1 = refinement_loss(*protein_out, pos_mol_out, mbatch.pos, positive_mol_batch)

    # output_handler = TrainingOutputHandler(
    #     *protein_out,
    #     final_mol_out,
    #     molecule_batch.pos,
    #     molecule_batch.batch,
    # )

    # anchor_samples, positive_samples, negative_samples = (
    #     output_handler.get_anchors_positives_negatives(
    #         prot_anchor=prot_loss, difficulty=difficulty_value
    #     )
    # )

    # loss = positive_margin_loss(anchor_samples, positive_samples, negative_samples)
    return l1,l2


def validate(
    data_loader, protein_encoder, mol_encoder, protein_refiner, mol_refiner, batch_size=16
):
    validation_loss_vals = []
    l1_vals = []
    l2_vals = []

    for prot_loss in [True]:
        for batch_idx in range(data_loader.size // batch_size):
            if batch_idx > 10:
                break
            with torch.no_grad():
                l1,l2 = training_step(
                    3, data_loader, protein_encoder, mol_encoder, protein_refiner, mol_refiner, 1, prot_loss
                )
                loss = l1+l2
                validation_loss_vals.append(loss)
                l1_vals.append(l1)
                l2_vals.append(l2)

    return sum(validation_loss_vals) / len(validation_loss_vals), sum(l1_vals) / len(l1_vals), sum(l2_vals) / len(l2_vals), 


def main(args):
    device = args.device
    log_text = logger(args.output)

    _, input_filetype = path.splitext(args.input)
    print('loading dataset...')

    if input_filetype in [".pkl", '.tv']:
        with open(args.input, "rb") as train_validate_data:
            train_data, validation_data = pickle.load(train_validate_data)

    elif input_filetype == ".csv":
        train_data, validation_data = construct_tv_set(args.input)

    if args.output is not None:
        with open(args.output, "w"):
            True

    print('...dataset loaded')

    train_loader = ProteinLigandDataLoader(train_data, batch_size=args.batch_size)
    validation_loader = ProteinLigandDataLoader(
        validation_data, batch_size=args.batch_size
    )

    protein_encoder = ProteinEncoder(**ProteinEncoderDefaults).to(device)
    protein_refiner = ProteinRefiner().to(device)

    mol_encoder = MolEncoder(**MolEncoderDefaults).to(device)
    mol_refiner = MolRefiner().to(device)

    get_hard_negative_difficulty = hard_negative_scheduler(50, 0.05)

    if args.load_encoder:
        protein_model_weights, mol_model_weights = torch.load(args.load_encoder)

        protein_encoder.load_state_dict(protein_model_weights)
        mol_encoder.load_state_dict(mol_model_weights)

    if args.load_refiner:
        protein_refiner_weights, mol_refiner_weights = torch.load(args.load_refiner)

        protein_refiner.load_state_dict(protein_refiner_weights)
        mol_refiner.load_state_dict(mol_refiner_weights)

    print('starting initial validation')
    best_validation_loss, l1, l2 = validate(
        validation_loader, protein_encoder, mol_encoder, protein_refiner, mol_refiner, batch_size=args.batch_size
    )
    log_text(f"Best validation loss: {best_validation_loss}")

    encoder_params = list(protein_encoder.parameters()) + list(mol_encoder.parameters())
    refiner_params = list(protein_refiner.parameters()) + list(mol_refiner.parameters())

    if args.train_type == 0:
        param_list = encoder_params + refiner_params
    elif args.train_type == 1:
        param_list = encoder_params
    elif args.train_type == 2:
        param_list = refiner_params 

    optimizer = torch.optim.AdamW(param_list, lr=args.learning_rate)
    prot_loss = True

    for epoch in range(args.epoch_start, args.epochs + 1):
        difficulty_value = get_hard_negative_difficulty(epoch)
        log_message = f"Epoch {epoch} difficulty: {difficulty_value}"
        log_text(log_message)
        loss_vals = []
        l1_vals = []
        l2_vals = []

        for batch_idx in range(train_loader.size // args.batch_size):
            prot_loss = not prot_loss
            l1,l2 = training_step(
                args.train_type, train_loader, protein_encoder, mol_encoder, protein_refiner, mol_refiner, difficulty_value, prot_loss
            )
            loss = l1+l2

            l1_vals.append(l1.item())
            l2_vals.append(l2.item())
            loss_vals.append(loss.item())

            if batch_idx % 10 == 0:
                loss_avg = torch.tensor(loss_vals).mean().item()
                l1_avg = torch.tensor(l1_vals).mean().item()
                l2_avg = torch.tensor(l2_vals).mean().item()
                log_text(f"Epoch {epoch}, batch {batch_idx} loss: {loss_avg} {l1_avg} {l2_avg}")
                
                l1_vals = []
                l2_vals = []
                loss_vals = []

            loss.backward()

            if prot_loss:
                optimizer.step()
                optimizer.zero_grad()
                torch.cuda.empty_cache()

        epoch_validation_loss, epoch_l1, epoch_l2 = validate(
            validation_loader, protein_encoder, mol_encoder, protein_refiner, mol_refiner, batch_size=args.batch_size
        )

        log_text(f"Epoch {epoch} validation loss: {epoch_validation_loss}")

        if epoch_validation_loss < best_validation_loss:
            if args.train_type in [0,1]:
                torch.save(
                    [protein_encoder.state_dict(), mol_encoder.state_dict()],
                    args.encoder_weight_path,
                )

            if args.train_type in [0,2]:
                torch.save(
                    [protein_refiner.state_dict(), mol_refiner.state_dict()],
                    args.refiner_weight_path,
                )

            best_validation_loss = epoch_validation_loss

            log_text(f"Weights updated")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train")
    add_arguments(parser)
    args = parser.parse_args()
    args.func(args)
