import argparse
import pickle
from simpatico.utils.data_utils import report_results


def add_arguments(parser):
    parser.add_argument(
        "input",
        help="Path to results .pkl file produced by `simpatico query`.",
    )
    parser.add_argument(
        "-n",
        "--limit",
        type=int,
        default=0,
        help="Print at most this many molecules per query (0 = all).",
    )
    # sets parser's main function to the main function in this script
    parser.set_defaults(main=main)


def main(args):
    with open(args.input, "rb") as pkl_in:
        results_data = pickle.load(pkl_in)

    report_results(results_data, limit=args.limit)
