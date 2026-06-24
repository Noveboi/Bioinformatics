#!/usr/bin/python

import argparse
import dataclasses
import logging
import statistics
from pathlib import Path

import alignment
import synthesis
from cachelib import Cache
from hmm import ProfileHMM

log = logging.getLogger("main")


def load_sequences(path: Path) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def add_cache_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--cache",
        type=str,
        dest="cache",
        help="The path of the cache directory",
        default="_cache",
    )


def add_seed_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--seed",
        type=int,
        dest="seed",
        help="Seed the randon number generator used for synthesis",
        default=777,
    )


# Part (i)
def run_synthesis(args) -> None:
    if args.seed is None:
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
    if args.id is None:
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
        "Profile matched %d out of %d columns",
        profile.profile_column_count,
        len(msa[0]),
    )

    log.info("Training HMM profile on %d sequences", len(dataset))

    def maybe_store_paths(path_name: str):
        if args.store_paths:
            paths = {}
            for seq in dataset:
                score, path = profile.viterbi(seq)
                paths[seq] = {
                    "score": score,
                    "path": [dataclasses.asdict(path) for path in path],
                }

            cache.save_json(path_name, paths)

    maybe_store_paths("viterbi_path")
    profile.train(dataset)
    maybe_store_paths("viterbi_path_trained")

    cache.save_pickle("profile", profile)


# Part (iv)
def run_alignment_scores(args) -> None:
    if args.seed is None:
        raise ValueError("Random sequence generation requires a seed")

    cache = Cache(args.cache)

    datasets = synthesis.DatasetCollection(**cache.load_json("datasets"))
    profile: ProfileHMM = cache.load_pickle("profile")

    dataset_sequences = datasets.datasetC
    random_sequences = synthesis.generate_random_sequences(
        datasets.datasetC,
        args.seed,
    )

    results = {
        "summary": {},
        "dataset": [],
        "random": [],
    }

    log.info(
        "Calculating alignment scores for %d datasetC sequences",
        len(dataset_sequences),
    )

    for sequence in dataset_sequences:
        score = profile.forward(sequence)
        results["dataset"].append(
            {
                "sequence": sequence,
                "length": len(sequence),
                "score": score,
                "normalized_score": score / len(sequence),
            }
        )

    log.info(
        "Calculating alignment scores for %d random sequences",
        len(random_sequences),
    )

    for sequence in random_sequences:
        score = profile.forward(sequence)
        results["random"].append(
            {
                "sequence": sequence,
                "length": len(sequence),
                "score": score,
                "normalized_score": score / len(sequence),
            }
        )

    # Highest-scoring sequences first
    results["dataset"].sort(key=lambda x: x["normalized_score"], reverse=True)
    results["random"].sort(key=lambda x: x["normalized_score"], reverse=True)

    results["summary"] = {
        "dataset": {
            "count": len(results["dataset"]),
            "mean": statistics.mean(r["normalized_score"] for r in results["dataset"]),
            "best": max(r["normalized_score"] for r in results["dataset"]),
            "worst": min(r["normalized_score"] for r in results["dataset"]),
        },
        "random": {
            "count": len(results["random"]),
            "mean": statistics.mean(r["normalized_score"] for r in results["random"]),
            "best": max(r["normalized_score"] for r in results["random"]),
            "worst": min(r["normalized_score"] for r in results["random"]),
        },
    }

    cache.save_json("results", results)


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

    msa_parser = subparsers.add_parser(
        name="msa",
        help="Perform multiple sequence alignment on datasetA, which is generated from the synthesize sub-program.",
    )

    hmm_parser = subparsers.add_parser(
        name="profile",
        help="Generate a profile HMM (Hidden Markov Model) initially based upon the multiple sequence alignment. Then train the profile"
        " on datasetB.",
    )

    score_parser = subparsers.add_parser(
        name="score",
        help="Calculate alignment scores for sequences using the any-path method. Results are output to a JSON file.",
    )

    add_cache_arg(synthesis_parser)
    add_cache_arg(msa_parser)
    add_cache_arg(hmm_parser)
    add_cache_arg(score_parser)

    add_seed_arg(synthesis_parser)
    add_seed_arg(score_parser)

    synthesis_parser.add_argument(
        "-p",
        "--patterns",
        type=str,
        dest="patterns",
        help="The text file containing the ATGC patterns to use. Each pattern should be in a new line",
        required=True,
    )

    msa_parser.add_argument(
        "--id",
        type=int,
        dest="id",
        help="The student ID to be used in determine the α parameter (used in the global alignment penalties)",
        required=True,
    )

    hmm_parser.add_argument(
        "--save-paths",
        action=argparse.BooleanOptionalAction,
        dest="store_paths",
        default=False,
        help="Whether to store the Viterbi best paths on disk.",
    )

    synthesis_parser.set_defaults(func=run_synthesis)
    msa_parser.set_defaults(func=run_msa)
    hmm_parser.set_defaults(func=run_profiling)
    score_parser.set_defaults(func=run_alignment_scores)

    args = parser.parse_args()

    log.info("Running program '%s'", args.command)
    log.info("Using arguments: %s", vars(args))

    args.func(args)


if __name__ == "__main__":
    main()
