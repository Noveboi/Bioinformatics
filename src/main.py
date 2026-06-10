#!/usr/bin/python

import argparse
import logging
from pathlib import Path

from cachelib import Cache

log = logging.getLogger("main")


def load_patterns(path: Path) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def run_synthesis_program(args) -> None:
    import synthesis

    if not args.seed:
        raise ValueError("Seed is not defined!")

    patterns_path = Path(args.patterns)

    if patterns_path.suffix != ".txt":
        raise ValueError(f"Expected .txt file for the patterns file at {patterns_path}")

    patterns = load_patterns(patterns_path)
    cache = Cache(args.cache)

    datasets = synthesis.build_datasets(patterns, args.seed)

    cache.save("datasets", datasets.to_dict())


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
