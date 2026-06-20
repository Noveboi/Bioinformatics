import math

ALPHABET = ("A", "C", "G", "T")
GAP = "-"
INFINITY = 1e30


class ProbabilityDistribution:
    """
    Encapsulates a dictionary with symbols as keys and probabilities as values.

    Enforces the invariant that all the probabilities in the distribution must sum to 1.
    """

    def __init__(self, distribution: dict[str, float]):
        prob_sum = sum(distribution.values())

        if abs(prob_sum - 1) > 1e-9:
            raise ValueError(
                f"Probability distribution {distribution} has sum {prob_sum}"
            )

        self.probabilities = distribution

    @classmethod
    def uniform(cls, symbols: list[str] | tuple[str, ...]) -> "ProbabilityDistribution":
        p = 1 / len(symbols)

        return ProbabilityDistribution({x: p for x in symbols})

    def copy(self) -> "ProbabilityDistribution":
        return ProbabilityDistribution(self.probabilities.copy())


def safe_log(p: float) -> float:
    """
    Compute the logarithm of a number safely. Returns negative infinity for
    zero or near-zero numbers.
    """
    return math.log(p) if p > 1e-300 else -INFINITY


def smooth_counts(counts: dict[str, float]) -> dict[str, float]:
    """
    Perform additive smoothing on the count dictionary.

    This prevents any one event/symbol having zero probability and is useful
    for making all events at least a tiny bit likely rather than completely unlikely.

    Sources
    --------
    https://en.wikipedia.org/wiki/Additive_smoothing
    """
    PSEUDOCOUNT: float = 1.0  # The smoothing parameter
    d = len(counts)

    total = sum(counts.values()) + PSEUDOCOUNT * d  # denominator: N + αd

    return {k: (v + PSEUDOCOUNT) / total for k, v in counts.items()}
