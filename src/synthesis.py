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
    """
    Generates a sequence partially random sequence in a 3-step process:

    Step #1: Prefix
        A random short sequence is used as the start of the whole sequence

    Step #2: Pattern Mutation
        Using the given ``patterns``, the algorithm will append each pattern to the body of the whole sequence.
        Before appending, each pattern is mutated randomly at randomly varying degrees.

    Step #3: Suffix
        A random short sequence is used as the end of the whole sequence. Same process as the prefix.

    Parameters
    --------
    patterns : list of str
        Base sequences of A,T,G,C to use for constructing/synthesizing the sequence.
        The resulting sequence will be very similar to the given patterns but will differ at some points
        due to the prefix, suffix and mutations that occur.

    rng: Random
        Random number generator for creating the prefix, suffix and defining the mutations.
    """
    prefix = "".join(rng.choice(ALPHABET) for _ in range(rng.randint(1, 3)))

    body_parts = [_mutate_pattern(pattern, rng) for pattern in patterns]

    suffix = "".join(rng.choice(ALPHABET) for _ in range(rng.randint(1, 3)))

    return prefix + "".join(body_parts) + suffix


def build_datasets(
    patterns: list[str],
    seed: int | None = None,
) -> DatasetCollection:
    """
    Build a small set of datasets containing sequences derived from the given ``patterns``
    """
    rng = Random(seed)

    sequences = [_synthesize_sequence(patterns, rng) for _ in range(200)]
    rng.shuffle(sequences)

    return DatasetCollection(
        datasetA=sequences[:20],
        datasetB=sequences[20:160],
        datasetC=sequences[160:],
    )
