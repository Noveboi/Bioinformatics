#!/usr/bin/python

import argparse
import json
import logging
from pathlib import Path

import alignment
import synthesis
from cachelib import Cache
from hmm import ProfileHMM

log = logging.getLogger("main")


def load_sequences(path: Path) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


# Part (i)
def run_synthesis_program(args) -> None:
    if not args.seed:
        raise ValueError("Seed is not defined!")

    patterns_path = Path(args.patterns)

    if patterns_path.suffix != ".txt":
        raise ValueError(f"Expected .txt file for the patterns file at {patterns_path}")

    log.info("Running SYNTHESIS")

    patterns = load_sequences(patterns_path)
    cache = Cache(args.cache)  # load early to catch path errorrs

    log.info("Building datasets using patterns: %s", patterns)
    datasets = synthesis.build_datasets(patterns, args.seed)

    cache.save_json("datasets", datasets.to_dict())


# Part (ii)
def run_msa(args) -> None:
    if not args.id:
        raise ValueError("Student ID is required")

    alpha = 2 - args.id % 2  # (2 - 21115 mod 2) = 1 for me

    log.info("Using α = %d", alpha)

    cache = Cache(args.cache)
    datasets = synthesis.DatasetCollection(**cache.load_json("datasets"))
    dataset = datasets.datasetA

    log.info("Starting MSA of %d sequences", len(dataset))
    msa = alignment.multiple_align(dataset, alpha=alpha)

    cache.save_json("msa", {"alignments": msa})


# Part (iii)
def run_profiling(args) -> None:
    cache = Cache(args.cache)

    datasets = synthesis.DatasetCollection(**cache.load_json("datasets"))
    msa: list[str] = cache.load_json("msa")["alignments"]
    dataset = datasets.datasetB

    log.info("Constructing HMM profile for %d aligned sequences", len(msa))
    profile = ProfileHMM(msa)

    log.info(
        "Profile matched %d out of %d columns", profile.match_column_count, len(msa[0])
    )

    log.info("Training HMM profile on %d sequences", len(dataset))

    profile.train(dataset)
    cache.save_pickle("profile", profile)


# Part (iv)
def run_alignment_scores(args) -> None:
    if not args.seed:
        raise ValueError("Random sequence generation requires a seed")

    cache = Cache(args.cache)
    datasets = synthesis.DatasetCollection(**cache.load_json("datasets"))

    dataset_sequences = datasets.datasetC
    random_sequences = synthesis.generate_random_sequences(datasets.datasetC, args.seed)

    profile: ProfileHMM = cache.load_pickle("profile")

    results = {"dataset": {}, "random": {}}

    log.info(
        "Calculating alignment scores for %d datasetC sequences", len(dataset_sequences)
    )
    for sequence in dataset_sequences:
        results["dataset"][sequence] = profile.forward(sequence)

    log.info(
        "Calculating alignment scores for %d random sequences", len(random_sequences)
    )
    for sequence in random_sequences:
        results["random"][sequence] = profile.forward(sequence)

    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d] (%(name)s) %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    synthesis_parser = subparsers.add_parser(
        name="synthesize",
        help="Generate datasets of ATGC sequences that are derived from specific patterns.",
    )

    synthesis_parser.add_argument(
        "-p",
        "--patterns",
        type=str,
        dest="patterns",
        help="The text file containing the ATGC patterns to use. Each pattern should be in a new line",
        required=True,
    )

    synthesis_parser.add_argument(
        "--cache",
        type=str,
        dest="cache",
        help="The path of the cache directory",
        default="_cache",
    )

    synthesis_parser.add_argument(
        "--seed",
        type=int,
        dest="seed",
        help="Seed the randon number generator used for synthesis",
        default=777,
    )

    synthesis_parser.set_defaults(func=run_synthesis_program)

    args = parser.parse_args()

    log.info("Running program '%s'", args.command)
    log.info("Using arguments: %s", vars(args))

    args.func(args)


if __name__ == "__main__":
    main()
