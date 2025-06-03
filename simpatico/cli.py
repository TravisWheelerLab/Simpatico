import argparse
from simpatico import train, evaluate, convert, get_train_set
from simpatico.utils.utils import SmartFormatter


def main():
    parser = argparse.ArgumentParser(
        description="CLI interface", formatter_class=SmartFormatter
    )

    # first argument defines which parser to use.
    subparsers = parser.add_subparsers(dest="command", required=True)

    eval_parser = subparsers.add_parser("eval", help="Evaluate something")
    # construct parser from specification in evaluate.py
    # and sets parser's 'main' function to evaluate.main
    evaluate.add_arguments(eval_parser)

    train_parser = subparsers.add_parser("train", help="Train something")
    train.add_arguments(train_parser)

    convert_parser = subparsers.add_parser(
        "convert", help="Convert protein or small molecule structures to PyG graphs"
    )
    convert.add_arguments(convert_parser)

    get_train_set_parser = subparsers.add_parser(
        "get-train-set",
        help="Converts protein and molecule files into PyG graphs and consolidates in train-validation set.",
    )
    get_train_set.add_arguments(get_train_set_parser)

    args = parser.parse_args()
    # run main function from correct script
    args.main(args)
