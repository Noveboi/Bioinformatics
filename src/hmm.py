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
P_MM, P_MI, P_MD = 0.8, 0.1, 0.1  # from MATCH to X
P_IM, P_II, P_ID = 0.1, 0.8, 0.1  # from INSERT to X
P_DM, P_DI, P_DD = 0.8, 0.1, 0.1  # from DELETE to X
P_BM, P_BI, P_BD = 0.8, 0.1, 0.1  # from B to X

LOG_MM, LOG_MI, LOG_MD = math.log(P_MM), math.log(P_MI), math.log(P_MD)
LOG_IM, LOG_II, LOG_ID = math.log(P_IM), math.log(P_II), math.log(P_ID)
LOG_DM, LOG_DI, LOG_DD = math.log(P_DM), math.log(P_DI), math.log(P_DD)
LOG_BM, LOG_BI, LOG_BD = math.log(P_BM), math.log(P_BI), math.log(P_BD)


class Node(Enum):
    NONE = -1
    BEGIN = 0
    MATCH = 1
    INSERT = 2
    DELETE = 3


@dataclass(frozen=True)
class ViterbiNode:
    type: Node
    profile_column: int
    sequence_position: int


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

        Sources
        --------
        https://www.ebi.ac.uk/training/online/courses/pfam-creating-protein-families/what-are-profile-hidden-markov-models-hmms/
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
        emit_match = [ProbabilityDistribution.uniform(ALPHABET) for _ in range(L)]

        for sequence in msa:
            for i, col in enumerate(match_columns):
                symbol = sequence[col]
                if symbol != GAP:
                    counts[i][symbol] = counts[i].get(symbol, 0) + 1

        for i in range(L):
            normalized = smooth_counts({c: counts[i].get(c, 0.0) for c in ALPHABET})
            emit_match[i] = ProbabilityDistribution(normalized)

        self.match_emission_probs = emit_match
        self.insert_emission_probs = ProbabilityDistribution.uniform(ALPHABET)
        self.match_column_count = L

    def _emite_match_log_prob(self, i: int, symbol: str) -> float:
        prob = self.match_emission_probs[i].probabilities.get(symbol, TINY)
        return safe_log(prob)

    def _emit_insert_log_prob(self, symbol: str) -> float:
        prob = self.insert_emission_probs.probabilities.get(symbol, TINY)
        return safe_log(prob)

    def viterbi(self, sequence: str) -> tuple[float, list[ViterbiNode]]:
        """
        Finds the most likely state path for a sequence
        """
        L, n = self.match_column_count, len(sequence)

        # Dynamic programming matrcies
        # 1. dp_m[i][j] : best score ending in MATCH at profile column i after emitting j symbols
        # 2. dp_i[i][j] : best score ending in INSERT at profile column i after emitting j symbols
        # 3. dp_d[i][j] : best score ending in DELETE at profile column i after emmitting j symbols
        dp_m = [[NEGATIVE_INFINITY] * (n + 1) for _ in range(L + 1)]
        dp_i = [[NEGATIVE_INFINITY] * (n + 1) for _ in range(L + 1)]
        dp_d = [[NEGATIVE_INFINITY] * (n + 1) for _ in range(L + 1)]

        # Traceback matrices -- these will define the best path
        trace_m = [[(Node.NONE, -1)] * (n + 1) for _ in range(L + 1)]
        trace_i = [[(Node.NONE, -1)] * (n + 1) for _ in range(L + 1)]
        trace_d = [[(Node.NONE, -1)] * (n + 1) for _ in range(L + 1)]

        # Initial Condition / Seed
        trace_m[1][1] = (Node.BEGIN, 0)
        trace_i[0][1] = (Node.BEGIN, 0)
        trace_d[1][0] = (Node.BEGIN, 0)

        # Main DP loop
        for i in range(L + 1):
            for j in range(n + 1):
                # INSERT[i][j] -- emits sequence[j-1], j does not increment
                if j > 0:
                    candidates: list[tuple[float, Node]] = []

                    if i == 0 and j == 1:
                        candidates.append((LOG_BI, Node.BEGIN))

                    if dp_m[i][j - 1] > NEGATIVE_INFINITY:
                        candidates.append((dp_m[i][j - 1] + LOG_MI, Node.MATCH))
                    if dp_i[i][j - 1] > NEGATIVE_INFINITY:
                        candidates.append((dp_i[i][j - 1] + LOG_II, Node.INSERT))
                    if dp_d[i][j - 1] > NEGATIVE_INFINITY:
                        candidates.append((dp_d[i][j - 1] + LOG_DI, Node.DELETE))

                    if len(candidates) > 0:
                        best_score, prev_node_type = max(candidates, key=lambda x: x[0])
                        nv = best_score + self._emit_insert_log_prob(sequence[j - 1])

                        if nv > dp_i[i][j]:
                            dp_i[i][j] = nv
                            trace_i[i][j] = (prev_node_type, i)

                # MATCH[i][j] -- emits sequence[j-1], j increments
                if i > 0 and j > 0:
                    candidates: list[tuple[float, Node]] = []

                    if i == 1 and j == 1:
                        candidates.append((LOG_BM, Node.BEGIN))

                    if dp_m[i - 1][j - 1] > NEGATIVE_INFINITY:
                        candidates.append((dp_m[i - 1][j - 1] + LOG_MM, Node.MATCH))
                    if dp_i[i - 1][j - 1] > NEGATIVE_INFINITY:
                        candidates.append((dp_i[i - 1][j - 1] + LOG_IM, Node.INSERT))
                    if dp_d[i - 1][j - 1] > NEGATIVE_INFINITY:
                        candidates.append((dp_d[i - 1][j - 1] + LOG_DM, Node.DELETE))

                    if len(candidates) > 0:
                        best_score, prev_node_type = max(candidates, key=lambda x: x[0])
                        nv = best_score + self._emite_match_log_prob(
                            i - 1, sequence[j - 1]
                        )

                        if nv > dp_m[i][j]:
                            dp_m[i][j] = nv
                            trace_m[i][j] = (prev_node_type, i - 1)

                # DELETE[i, j] -- do not emit, j does not incremenet
                if i > 0:
                    candidates: list[tuple[float, Node]] = []

                    if i == 1 and j == 0:
                        candidates.append((LOG_BD, Node.BEGIN))

                    if dp_m[i - 1][j] > NEGATIVE_INFINITY:
                        candidates.append((dp_m[i - 1][j] + LOG_MD, Node.MATCH))
                    if dp_i[i - 1][j] > NEGATIVE_INFINITY:
                        candidates.append((dp_i[i - 1][j] + LOG_ID, Node.INSERT))
                    if dp_d[i - 1][j] > NEGATIVE_INFINITY:
                        candidates.append((dp_d[i - 1][j] + LOG_DD, Node.DELETE))

                    if len(candidates) > 0:
                        best_score, prev_node_type = max(candidates, key=lambda x: x[0])

                        if best_score > dp_d[i][j]:
                            dp_d[i][j] = best_score
                            trace_d[i][j] = (prev_node_type, i - 1)

        # Determine the best final state after processing all `n` symbols
        final_candidates: list[tuple[float, Node]] = []

        if dp_m[L][n] > NEGATIVE_INFINITY:
            final_candidates.append((dp_m[L][n], Node.MATCH))
        if dp_i[L][n] > NEGATIVE_INFINITY:
            final_candidates.append((dp_i[L][n], Node.INSERT))
        if dp_d[L][n] > NEGATIVE_INFINITY:
            final_candidates.append((dp_d[L][n], Node.DELETE))

        if len(final_candidates) == 0:
            return NEGATIVE_INFINITY, []

        score, node_type = max(final_candidates, key=lambda x: x[0])
        i, j = L, n

        # Construct the path from end to beginning
        path: list[ViterbiNode] = []

        while node_type != Node.BEGIN:
            path.append(ViterbiNode(node_type, profile_column=i, sequence_position=j))

            if node_type == Node.MATCH:
                trace = trace_m[i][j]
                j -= 1
            elif node_type == Node.INSERT:
                trace = trace_i[i][j]
                j -= 1
            elif node_type == Node.DELETE:
                trace = trace_d[i][j]
                # j unchanged
            else:
                raise RuntimeError(f"Unexpected trackback node: {node_type.name}")

            node_type, i = trace

        path.reverse()
        return score, path


if __name__ == "__main__":
    from alignment import multiple_align
    from cachelib import Cache
    from synthesis import DatasetCollection

    cache = Cache("_cache")
    datasets = DatasetCollection(**cache.load("datasets"))

    msa = multiple_align(datasets.datasetB, alpha=1)
    profile = ProfileHMM(msa)

    print(f"Symbols: {len(msa[0])}")
    print(f"Matches: {profile.match_column_count}\n{'-' * 40}")

    print("\nEmission Probability Distributions per Match")
    for i, emission in enumerate(profile.match_emission_probs):
        print(f"{i + 1}: {emission.display()}")

    seq = datasets.datasetC[2]
    seq2 = seq[:15] + "A" + seq[15:]

    def v(s):
        print(f"\nViterbi Path for {s}")
        score, path = profile.viterbi(s)

        print(f"Score: {score}")
        print(f"Path: {len(path)}")
        for node in path:
            print(f"{node.type.name} ({node.profile_column}, {node.sequence_position})")

    v(seq)
    v(seq2)
