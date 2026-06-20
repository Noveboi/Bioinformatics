import math
from dataclasses import dataclass
from enum import Enum

from common import (
    ALPHABET,
    GAP,
    INFINITY,
    ProbabilityDistribution,
    safe_log,
    smooth_counts,
)

TINY = 1e-10
NEGATIVE_INFINITY = -INFINITY

# Fixed transition probabilities
P_MM, P_MD = 0.9, 0.1  # from Match
P_DM, P_DD = 0.9, 0.1  # from Delete
P_BM, P_BD = 0.9, 0.1  # from Begin

LOG_MM, LOG_MD = math.log(P_MM), math.log(P_MD)
LOG_DM, LOG_DD = math.log(P_DM), math.log(P_DD)
LOG_BM, LOG_BD = math.log(P_BM), math.log(P_BD)


class Node(Enum):
    NONE = -1
    MATCH = 0
    DELETE = 1


@dataclass(frozen=True)
class ViterbiNode:
    type: Node
    profile_col: int
    sequence_pos: int


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

    def viterbi(self, sequence: str) -> tuple[float, list[ViterbiNode]]:
        """
        Finds the most likely state path for a sequence
        """
        L, n = self.match_column_count, len(sequence)
        emit = self.emit_match

        # Dynamic programming matrcies
        dp_m: list[list[float]] = [[NEGATIVE_INFINITY] * (n + 1) for _ in range(L)]
        dp_d: list[list[float]] = [[NEGATIVE_INFINITY] * (n + 1) for _ in range(L)]

        # Traceback matrices -- these will define the best path
        trace_m: list[list[tuple[Node, int]]] = [
            [(Node.NONE, -1)] * (n + 1) for _ in range(L)
        ]
        trace_d: list[list[tuple[Node, int]]] = [
            [(Node.NONE, -1)] * (n + 1) for _ in range(L)
        ]

        # Initial Condition / Seed
        if n >= 1:
            probability = emit[0].probabilities.get(sequence[0], TINY)
            dp_m[0][1] = LOG_BM + safe_log(probability)

        dp_d[0][0] = LOG_BD

        # Main DP loop
        for i in range(L):
            for j in range(n + 1):
                # Case: D[i, j] -- do not emit, j does not incremenet
                if i > 0:
                    candidates: list[tuple[float, Node]] = []

                    if dp_m[i - 1][j] > NEGATIVE_INFINITY:
                        match_score = (dp_m[i - 1][j] + LOG_MD, Node.MATCH)
                        candidates.append(match_score)
                    if dp_d[i - 1][j] > NEGATIVE_INFINITY:
                        del_score = (dp_d[i - 1][j] + LOG_DD, Node.DELETE)
                        candidates.append(del_score)

                    if len(candidates) > 0:
                        best_score, node_type = max(candidates, key=lambda x: x[0])

                        if best_score > dp_d[i][j]:
                            dp_d[i][j] = best_score
                            trace_d[i][j] = (node_type, i - 1)

                # Case: M[i, j] -- emits sequence[j-1], so j increments
                if i > 0 and j > 0:
                    candidates: list[tuple[float, Node]] = []
                    emit_prob = emit[i].probabilities.get(sequence[j - 1], TINY)

                    if dp_m[i - 1][j - 1] > NEGATIVE_INFINITY:
                        match_score = (dp_m[i - 1][j - 1] + LOG_MM, Node.MATCH)
                        candidates.append(match_score)
                    if dp_d[i - 1][j - 1] > NEGATIVE_INFINITY:
                        del_score = (dp_d[i - 1][j - 1] + LOG_DM, Node.DELETE)
                        candidates.append(del_score)

                    if len(candidates) > 0:
                        best_score, node_type = max(candidates, key=lambda x: x[0])
                        nv = best_score + safe_log(emit_prob)

                        if nv > dp_m[i][j]:
                            dp_m[i][j] = nv
                            trace_m[i][j] = (node_type, i - 1)

        # Determine the best final state after processing all `n` symbols
        final_candidates: list[tuple[float, Node]] = []

        if dp_m[L - 1][n] > NEGATIVE_INFINITY:
            final_candidates.append((dp_m[L - 1][n], Node.MATCH))
        if dp_d[L - 1][n] > NEGATIVE_INFINITY:
            final_candidates.append((dp_d[L - 1][n], Node.DELETE))

        if len(final_candidates) == 0:
            return NEGATIVE_INFINITY, []

        score, node_type = max(final_candidates, key=lambda x: x[0])
        i, j = L - 1, n

        # Construct the path from end to beginning
        path: list[ViterbiNode] = []

        while True:
            path.append(ViterbiNode(node_type, profile_col=i, sequence_pos=j))

            if node_type == Node.MATCH:
                trace = trace_m[i][j]
                j -= 1
            elif node_type == Node.DELETE:
                trace = trace_d[i][j]
                # j unchanged
            else:
                break

            node_type, i = trace

        path.reverse()
        return score, path
