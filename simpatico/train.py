import sys
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
from simpatico.models import MolEncoderDefaults, ProteinEncoderDefaults
from typing import Callable


def add_arguments(parser):
    parser.add_argument(
        "input_data",
        type=str,
        help="Path to train-eval dataset",
    )
    parser.add_argument("weight_path"),
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
        "-l",
        "--load_model",
        action="store_true",
        help="Load previously trained weights",
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


def training_step(
    data_loader, protein_encoder, mol_encoder, difficulty_value, prot_loss=True
):
    device = next(protein_encoder.parameters()).device

    protein_batch, molecule_batch = data_loader.get_random_batch()
    protein_batch = protein_batch.clone()
    protein_batch.pos += torch.randn_like(protein_batch.pos) * 0.25

    protein_batch = protein_batch.to(device)
    molecule_batch = molecule_batch.to(device)

    protein_out = protein_encoder(protein_batch)
    mol_out = mol_encoder(molecule_batch)

    output_handler = TrainingOutputHandler(
        *protein_out,
        mol_out,
        molecule_batch.pos,
        molecule_batch.batch,
    )

    if prot_loss:
        anchor_samples, positive_samples, negative_samples = (
            output_handler.get_protein_anchor_pairs(difficulty=difficulty_value)
        )
    else:
        anchor_samples, positive_samples, negative_samples = (
            output_handler.get_mol_anchor_pairs(difficulty=difficulty_value)
        )

    loss = positive_margin_loss(anchor_samples, positive_samples, negative_samples)
    return loss


def validate(
    data_loader, protein_encoder, mol_encoder, difficulty_value, batch_size=16
):
    validation_loss_vals = []

    for prot_loss in [True, False]:
        for batch_idx in range(data_loader.size // batch_size):
            with torch.no_grad():
                loss = training_step(
                    data_loader,
                    protein_encoder,
                    mol_encoder,
                    difficulty_value,
                    prot_loss,
                )
                validation_loss_vals.append(loss)

    return sum(validation_loss_vals) / len(validation_loss_vals)


def main(args):
    device = args.device
    log_text = logger(args.output)
    # Load data
    with open(args.input_data, "rb") as train_validate_data:
        train_data, validation_data = pickle.load(train_validate_data)

    if args.output is not None:
        with open(args.output, "w"):
            True

    train_loader = ProteinLigandDataLoader(train_data, batch_size=args.batch_size)
    validation_loader = ProteinLigandDataLoader(
        validation_data, batch_size=args.batch_size
    )

    protein_encoder = ProteinEncoder(**ProteinEncoderDefaults).to(device)
    mol_encoder = MolEncoder(**MolEncoderDefaults).to(device)

    if args.load_model:
        protein_model_weights, mol_model_weights = torch.load(args.weight_path)

        protein_encoder.load_state_dict(protein_model_weights)
        mol_encoder.load_state_dict(mol_model_weights)

        initial_validation_score, _ = validate(
            protein_encoder,
            mol_encoder,
            validation_loader,
            device,
        )
        log_text(f"Best validation score: {initial_validation_score}")

    optimizer = torch.optim.AdamW(
        list(protein_encoder.parameters()) + list(mol_encoder.parameters()),
        lr=args.learning_rate,
    )

    get_hard_negative_difficulty = hard_negative_scheduler(50, 0.05)
    prot_loss = True
    best_validation_score = None

    for epoch in range(args.epoch_start, args.epochs + 1):
        difficulty_value = get_hard_negative_difficulty(epoch)

        if best_validation_score is None:
            best_validation_score = validate(
                validation_loader, protein_encoder, mol_encoder, difficulty_value
            )

        log_message = f"Epoch {epoch} difficulty: {difficulty_value}"
        log_text(log_message)
        loss_vals = []

        for batch_idx in range(train_loader.size // args.batch_size):
            prot_loss = not prot_loss
            loss = training_step(
                train_loader, protein_encoder, mol_encoder, difficulty_value, prot_loss
            )

            loss_vals.append(loss)

            if batch_idx % 10 == 0:
                loss_avg = torch.hstack(loss_vals).mean().item()
                log_text(f"Epoch {epoch}, batch {batch_idx}: {loss_avg}")
                loss_vals = []

            loss.backward()

            if prot_loss:
                optimizer.step()
                optimizer.zero_grad()
                torch.cuda.empty_cache()

        epoch_validation_score = validate(
            validation_loader, protein_encoder, mol_encoder, difficulty_value
        )

        log_text(f"Epoch {epoch} validation: {epoch_validation_score}")

        if epoch_validation_score > best_validation_score:
            torch.save(
                [protein_encoder.state_dict(), mol_encoder.state_dict()],
                args.weight_path,
            )

            log_text(f"Weights updated")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation tool")
    add_arguments(parser)
    args = parser.parse_args()
    args.func(args)
