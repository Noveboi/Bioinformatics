import dataclasses
from random import Random

from common import ALPHABET


@dataclasses.dataclass(frozen=True)
class DatasetCollection:
    datasetA: list[str]
    datasetB: list[str]
    datasetC: list[str]

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def _mutate_pattern(
    pattern: str,
    rng: Random,
    max_changes: int = 2,
) -> str:
    """
    Apply [0, max_changes] random edits to a pattern.

    Edit types:
      - substitute a symbol with a random symbol from the alphabet, or
      - delete the symbol.
    """
    chars = list(pattern)

    n_changes = rng.randint(0, max_changes)
    n_chars = len(chars)

    for _ in range(n_changes):
        if not chars:
            break

        pos = rng.randrange(n_chars)

        if rng.random() < 0.5:
            chars[pos] = rng.choice(ALPHABET)
        else:
            del chars[pos]
            n_chars -= 1

    return "".join(chars)


def _synthesize_sequence(patterns: list[str], rng: Random) -> str:
    prefix = "".join(rng.choice(ALPHABET) for _ in range(rng.randint(1, 3)))

    body_parts = [_mutate_pattern(pattern, rng) for pattern in patterns]

    suffix = "".join(rng.choice(ALPHABET) for _ in range(rng.randint(1, 3)))

    return prefix + "".join(body_parts) + suffix


def build_datasets(
    patterns: list[str],
    seed: int | None = None,
) -> DatasetCollection:
    """
    Build a small set of datasets containing partially random sequences
    """
    rng = Random(seed)

    sequences = [_synthesize_sequence(patterns, rng) for _ in range(200)]
    rng.shuffle(sequences)

    return DatasetCollection(
        datasetA=sequences[:20],
        datasetB=sequences[20:160],
        datasetC=sequences[160:],
    )
