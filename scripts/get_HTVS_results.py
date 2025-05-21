import torch
import time
from copy import deepcopy
import sys
import pickle
import os
import numpy as np
import faiss
import argparse
from glob import glob


def get_args():
    parser = argparse.ArgumentParser(
        description="Calculate EF scores from all HTVS outputs"
    )

    parser.add_argument(
        "-i", "--input-dir", type=str, help="Path to HTVS result files."
    )
    parser.add_argument("--ef-in", type=str)

    parser.add_argument("--decoy-values-out", type=str, default=None)
    parser.add_argument("--decoy-values-in", type=str, default=None)
    parser.add_argument("--ef-out", type=str, default=None)
    parser.add_argument("--prefix", type=str, default="")

    return parser.parse_args()


def get_decoy_values(args):
    max_pos = 0
    total_time = 0
    n_positives_estimate = int(70e6)
    result_files = glob(args.input_dir + "/*.pkl")

    all_decoy_values = {}

    for f_i, f in enumerate(result_files):
        if f_i % 500 == 0:
            print(f_i)

        if "ACTIVE" in f:
            continue

        result_data = pickle.load(open(f, "rb"))

        if len(result_data[0]) != len(result_data[1]):
            continue

        for target_id, decoy_data in zip(*result_data[:2]):
            if target_id not in all_decoy_values:
                all_decoy_values[target_id] = {
                    "total": 0,
                    "total_positives": 0,
                    "scores": torch.zeros(n_positives_estimate),
                }

            all_decoy_values[target_id]["total"] += decoy_data[0]
            start_idx = all_decoy_values[target_id]["total_positives"]
            end_idx = start_idx + decoy_data[1].size(0)

            all_decoy_values[target_id]["scores"][start_idx:end_idx] = decoy_data[1]

            all_decoy_values[target_id]["total_positives"] = end_idx

    for k, v in all_decoy_values.items():
        v["scores"] = v["scores"][: v["total_positives"]]

    pickle.dump(all_decoy_values, open(args.decoy_values_out, "wb"))


def get_EF(
    active_data,
    decoy_data,
    sample_ratios=[
        0.000001,
        0.000005,
        0.00001,
        0.00005,
        0.0001,
        0.0005,
        0.001,
        0.005,
        0.01,
        0.05,
    ],
):
    EF_vals = []
    hit_sample_counts = []

    active_count = active_data[0]
    decoy_count = decoy_data[0]
    total_count = active_count + decoy_count

    true_ratio = active_count / total_count

    active_vals = active_data[1]
    decoy_vals = decoy_data[1]

    all_vals = torch.hstack((active_vals, decoy_vals))
    active_mask = torch.hstack(
        (torch.ones_like(active_vals), torch.zeros_like(decoy_vals))
    )
    sorted_index = all_vals.argsort(descending=True)
    sorted_mask = active_mask[sorted_index]

    for sr in sample_ratios:
        sample_size = int(sr * total_count)
        hit_count = sorted_mask[:sample_size].sum()
        EF = (hit_count / sample_size) / true_ratio
        EF_vals.append(EF)
        hit_sample_counts.append(
            (int(hit_count), sample_size, active_count, total_count)
        )

    return (EF_vals, hit_sample_counts)


def get_active_results(args):
    decoy_data = pickle.load(open(args.decoy_values_in, "rb"))
    result_files = glob(args.input_dir + "/*ACTIVE*.pkl")
    active_trials = []
    if args.prefix != "":
        args.prefix += "_"

    for f in result_files:
        trial_data = {}
        result_data = pickle.load(open(f, "rb"))

        for target_id, screening_results in zip(result_data[0], result_data[1]):
            trial_data[target_id] = screening_results[0]

        active_trials.append(trial_data)

    for a_i, at in enumerate(active_trials):
        print(f"TRIAL {a_i}")
        trial_results = {}

        for k, v in at.items():
            print(a_i, k)
            EF_vals = get_EF(v, (decoy_data[k]["total"], decoy_data[k]["scores"]))
            trial_results[k] = EF_vals

        ef_outfile = f"{args.ef_out}/{args.prefix}{a_i}.pkl"
        print(ef_outfile)
        pickle.dump(trial_results, open(ef_outfile, "wb"))


def get_summary_stats(args):
    ef_files = glob(args.ef_in + "*.pkl")
    decoy_files = [f for f in glob(args.input_dir + "*.pkl") if "ACTIVE" not in f]

    target_data = {}

    for ef_f in sorted(ef_files):
        ef_data = pickle.load(open(ef_f, "rb"))

        for k, v in ef_data.items():
            print(k)
            print(v[1])
        sys.exit()
    #         if k not in target_data:
    #             target_data[k] = []
    #         else:
    #             target_data[k].append(v)

    # for k, v in target_data.items():
    #     print(k)
    #     for trial in v:
    #         print(trial)
    #         break


def main(args):
    if args.decoy_values_out is not None:
        get_decoy_values(args)

    if args.decoy_values_in is not None:
        get_active_results(args)

    if args.ef_in is not None:
        get_summary_stats(args)


if __name__ == "__main__":
    args = get_args()
    main(args)
