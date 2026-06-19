from common import GAP


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
        n_cols = len(msa[0])

        match_columns = _get_match_columns(msa, threshold=THRESHOLD)
        L = len(match_columns)

        # ! if you hit this, lower the threshold little-by-ltitle
        if L == 0:
            raise ValueError(
                f"No match columns detected from MSA. (threshold={THRESHOLD}"
            )

        column_labels: list[tuple[str, int]] = []
        match_count: int = 0

        # Label every MSA column:
        # - ('M', k)
        # - ('I', k)
        # - ('D', -1)
        # where the 2-tuple is (<symbol>, <index>)
        for j in range(n_cols):
            if j in match_columns:
                label = ("M", match_count)
                match_count += 1
            elif 0 < match_count < L:
                label = ("I", match_count - 1)
            else:
                label = ("SKIP", -1)

            column_labels.append(label)

        # Create the graph
        pass
