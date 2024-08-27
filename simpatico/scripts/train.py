# scripts/train.py

import argparse
import torch
from torch.utils.data import DataLoader
from models import model1  # Import your model
from utils import metrics, visualization
from data.data_utils import load_data
from config import config  # Import configurations

# API: simpatico train --t /path/to/trainloader --v path/to/validationloader


def train(model, train_loader, optimizer, criterion, device):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print(
                f"Train Epoch: [{batch_idx * len(data)}/{len(train_loader.dataset)}]\tLoss: {loss.item():.6f}"
            )


def main(args):
    # Load data
    train_data = load_data(args.data_path, train=True)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

    # Model, criterion, optimizer
    model = model1.Model().to(args.device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # Train the model
    for epoch in range(1, args.epochs + 1):
        train(model, train_loader, optimizer, criterion, args.device)
        # Save checkpoint
        if args.save_model:
            torch.save(
                model.state_dict(), f"../experiments/checkpoints/model_epoch_{epoch}.pt"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training Script")
    parser.add_argument(
        "--data_path",
        type=str,
        default="../data/processed/",
        help="Path to the dataset",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Input batch size for training"
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of epochs to train"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.001, help="Learning rate"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for training",
    )
    parser.add_argument(
        "--save_model", action="store_true", help="For Saving the current Model"
    )

    args = parser.parse_args()
    main(args)
