import math

ALPHABET = ("A", "C", "G", "T")
GAP = "-"
INFINITY = 1e30


def _safe_log(p: float) -> float:
    """
    Compute the logarithm of a number safely. Returns negative infinity for
    zero or near-zero numbers.
    """
    return math.log(p) if p > 1e-300 else -INFINITY


class ProbabilityDistribution[T]:
    """
    Encapsulates a dictionary with symbols (of type T) as keys and probabilities as values.

    Enforces the invariant that all the probabilities in the distribution must sum to 1.
    """

    def __init__(self, distribution: dict[T, float]):
        prob_sum = sum(distribution.values())

        if abs(prob_sum - 1) > 1e-3:
            raise ValueError(
                f"Probability distribution {distribution} has sum {prob_sum}"
            )

        self.probabilities = distribution

    def __str__(self) -> str:
        return self.probabilities.__str__()

    def __repr__(self) -> str:
        return self.__str__()

    def display(self, decimal_places: int = 3) -> str:
        return {
            k: round(v, decimal_places) for k, v in self.probabilities.items()
        }.__str__()

    @classmethod
    def from_counts[TSymbol](
        cls,
        counts: dict[TSymbol, int],
        smooth: bool,
    ) -> "ProbabilityDistribution[TSymbol]":
        """
        Construct a probability distribution from a dictionary of symbol counts.

        Parameters
        --------
        counts : dict[TSymbol, int]
            The count dictionary with the keys containing the symbols and the values
            containing the corresponding counts.
        smooth : bool, optional
            Whether to perform additive/laplace smoothing on the ``counts`` before calculating
            the probabilities.

        Sources
        --------
        1. Additive Smoothing - https://en.wikipedia.org/wiki/Additive_smoothing
        """
        total = sum(counts.values())
        d = len(counts)

        pseudocount = 1 if smooth else 0

        probalities: dict[TSymbol, float] = {
            symbol: (count + pseudocount) / (total + pseudocount * d)
            for symbol, count in counts.items()
        }

        return ProbabilityDistribution(probalities)

    @classmethod
    def uniform[TSymbol](
        cls,
        symbols: Sequence[TSymbol],
    ) -> "ProbabilityDistribution[TSymbol]":
        """
        Construct a uniform probability distribution for a set of symbols.

        Examples
        --------
        ```
        symbols = ['A', 'B', 'C', 'D']
        probs = ProbabilityDistribution.uniform(symbols)
        probs.display() # { 'A': 0.25, 'B': 0.25, 'C': 0.25, 'D': 0.25 }
        ```
        """
        p = 1 / len(symbols)

        return ProbabilityDistribution({x: p for x in symbols})

    def get(self, symbol: T) -> float:
        return self.probabilities[symbol]

    def log(self, symbol: T) -> float:
        return _safe_log(self.get(symbol))

    def copy(self) -> "ProbabilityDistribution[T]":
        return ProbabilityDistribution(self.probabilities.copy())
