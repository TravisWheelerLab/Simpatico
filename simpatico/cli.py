import argparse
from simpatico import train, evaluate
from simpatico.utils.utils import SmartFormatter


def main():
    parser = argparse.ArgumentParser(
        description="CLI interface", formatter_class=SmartFormatter
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Eval
    eval_parser = subparsers.add_parser("eval", help="Evaluate something")
    evaluate.add_arguments(eval_parser)

    # Train
    train_parser = subparsers.add_parser("train", help="Train something")
    train.add_arguments(train_parser)

    args = parser.parse_args()
    args.func(args)
