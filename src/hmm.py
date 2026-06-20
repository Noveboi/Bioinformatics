from common import ALPHABET, GAP, ProbabilityDistribution, smooth_counts


def _get_match_columns(msa: list[str], threshold: float) -> list[int]:
    """
    Returns the indices of 'match' columns in the MSA.

    A column is a 'match' column if the fraction of gap symbols is
    below the given ``threshold``. That means a lower ``threshold`` allows more
    'match' columns to be identified, a higher ``threshold`` is stricter.
    """
    n_cols = len(msa[0])
    n_rows = len(msa)

    return [
        j
        for j in range(n_cols)
        if sum(seq[j] == GAP for seq in msa) / n_rows < threshold
    ]


class ProfileHMM:
    def __init__(self, msa: list[str]):
        """
        Construct the HMM profile from an MSA (Multi-Sequence Alignment) result.
        """
        THRESHOLD: float = 0.5

        match_columns = _get_match_columns(msa, threshold=THRESHOLD)
        L = len(match_columns)

        # ! if you hit this, lower the threshold little-by-ltitle
        if L == 0:
            raise ValueError(
                f"No match columns detected from MSA. (threshold={THRESHOLD}"
            )

        counts: list[dict[str, float]] = [{} for _ in range(L)]
        emit_match: list[ProbabilityDistribution] = [
            ProbabilityDistribution.uniform(ALPHABET) for _ in range(L)
        ]

        for sequence in msa:
            for i, col in enumerate(match_columns):
                symbol = sequence[col]
                if symbol != GAP:
                    counts[i][symbol] = counts[i].get(symbol, 0) + 1

        for i in range(L):
            normalized = smooth_counts({c: counts[i].get(c, 0.0) for c in ALPHABET})
            emit_match[i] = ProbabilityDistribution(normalized)

        self.emit_match = emit_match
        self.match_column_count = L
