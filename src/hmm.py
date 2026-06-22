from dataclasses import dataclass
from enum import Enum

from common import (
    ALPHABET,
    GAP,
    INFINITY,
    ProbabilityDistribution,
)

NEGATIVE_INFINITY = -INFINITY


class State(Enum):
    NONE = -1
    BEGIN = 0
    MATCH = 1
    INSERT = 2
    DELETE = 3


# shorthand syntax for more compact code
# the ``State`` class is kept for strong type inference
BEGIN = State.BEGIN
MATCH = State.MATCH
INSERT = State.INSERT
DELETE = State.DELETE


@dataclass(frozen=True)
class ViterbiNode:
    state: State
    profile_column: int
    sequence_position: int


class TransitionProbabilities:
    def __init__(self):
        # HMM Transition probability distribution modelled as P(S_n = x | S_{n-1} = y).
        # A nested dictionary is used for encapsulating each conditioned probability space Ω in a ``ProbabilityDistribution``,
        # thus enforcing the probability theory invariants.
        self._probs = {
            BEGIN: ProbabilityDistribution.uniform([MATCH, INSERT, DELETE]),
            MATCH: ProbabilityDistribution.uniform([MATCH, INSERT, DELETE]),
            INSERT: ProbabilityDistribution.uniform([MATCH, INSERT, DELETE]),
            DELETE: ProbabilityDistribution.uniform([MATCH, INSERT, DELETE]),
        }

    def get(self, previous_state: State, current_state: State):
        return self._probs[previous_state].get(current_state)

    def log(self, previous_state: State, current_state: State):
        return self._probs[previous_state].log(current_state)


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
    """
    A specific type of hidden Markov model (HMM) with three states:
        - INSERT (I)
        - MATCH (M)
        - DELETE (D)

    The purpose of the profile HMM is to emit sequences (A, T, G, C) by first predicting the
    overall structure of a given set of related sequences

    Sources
    --------
        1. https://www.ebi.ac.uk/training/online/courses/pfam-creating-protein-families/what-are-profile-hidden-markov-models-hmms/
    """

    def __init__(self, msa: list[str]):
        """
        Construct the HMM profile from an MSA (Multi-Sequence Alignment) result.
        """
        THRESHOLD: float = 0.5

        match_columns = _get_match_columns(msa, threshold=THRESHOLD)
        match_columns_set = set(match_columns)
        L = len(match_columns)

        if L == 0:  # ! if you hit this, lower the threshold
            raise ValueError(
                f"No match columns detected from MSA. (threshold={THRESHOLD}"
            )

        match_counts: list[dict[str, int]] = [
            {c: 0 for c in ALPHABET} for _ in range(L)
        ]
        insert_counts: dict[str, int] = {c: 0 for c in ALPHABET}

        for sequence in msa:
            for j, col in enumerate(match_columns):
                symbol = sequence[col]

                if symbol != GAP:
                    match_counts[j][symbol] += 1

        for sequence in msa:
            for col in range(len(sequence)):
                symbol = sequence[col]
                if col not in match_columns_set and symbol != GAP:
                    insert_counts[symbol] += 1

        self.match_column_count = L
        self.transition_probs = TransitionProbabilities()

        self.match_emit_probs = [
            ProbabilityDistribution.from_counts(c, True) for c in match_counts
        ]
        self.insert_emit_probs = ProbabilityDistribution.from_counts(
            insert_counts, True
        )

    def viterbi(self, sequence: str) -> tuple[float, list[ViterbiNode]]:
        """
        Finds the most likely state path for a sequence
        """
        L, n = self.match_column_count, len(sequence)
        tp = self.transition_probs.log

        # Dynamic programming matrcies
        # 1. dp_m[i][j] : best score ending in MATCH at profile column i after emitting j symbols
        # 2. dp_i[i][j] : best score ending in INSERT at profile column i after emitting j symbols
        # 3. dp_d[i][j] : best score ending in DELETE at profile column i after emmitting j symbols
        dp_m = [[NEGATIVE_INFINITY] * (n + 1) for _ in range(L + 1)]
        dp_i = [[NEGATIVE_INFINITY] * (n + 1) for _ in range(L + 1)]
        dp_d = [[NEGATIVE_INFINITY] * (n + 1) for _ in range(L + 1)]

        # Traceback matrices -- these will define the best path
        trace_m = [[(State.NONE, -1)] * (n + 1) for _ in range(L + 1)]
        trace_i = [[(State.NONE, -1)] * (n + 1) for _ in range(L + 1)]
        trace_d = [[(State.NONE, -1)] * (n + 1) for _ in range(L + 1)]

        # Main DP loop
        for i in range(L + 1):
            for j in range(n + 1):
                # INSERT[i][j] -- emits sequence[j-1], j does not increment
                if j > 0:
                    candidates: list[tuple[float, State]] = []

                    if i == 0 and j == 1:  # INSERT Initial Condition
                        candidates.append((tp(BEGIN, INSERT), BEGIN))

                    if dp_m[i][j - 1] > NEGATIVE_INFINITY:
                        candidates.append((dp_m[i][j - 1] + tp(MATCH, INSERT), MATCH))
                    if dp_i[i][j - 1] > NEGATIVE_INFINITY:
                        candidates.append((dp_i[i][j - 1] + tp(INSERT, INSERT), INSERT))
                    if dp_d[i][j - 1] > NEGATIVE_INFINITY:
                        candidates.append((dp_d[i][j - 1] + tp(DELETE, INSERT), DELETE))

                    if len(candidates) > 0:
                        score, prev_state = max(candidates, key=lambda x: x[0])
                        best_emission_score = score + self.insert_emit_probs.log(
                            sequence[j - 1]
                        )

                        if best_emission_score > dp_i[i][j]:
                            dp_i[i][j] = best_emission_score
                            trace_i[i][j] = (prev_state, i)

                # MATCH[i][j] -- emits sequence[j-1], j increments
                if i > 0 and j > 0:
                    candidates: list[tuple[float, State]] = []

                    if i == 1 and j == 1:  # MATCH Initial Condition
                        candidates.append((tp(BEGIN, MATCH), BEGIN))

                    if dp_m[i - 1][j - 1] > NEGATIVE_INFINITY:
                        candidates.append(
                            (dp_m[i - 1][j - 1] + tp(MATCH, MATCH), MATCH)
                        )
                    if dp_i[i - 1][j - 1] > NEGATIVE_INFINITY:
                        candidates.append(
                            (dp_i[i - 1][j - 1] + tp(INSERT, MATCH), INSERT)
                        )
                    if dp_d[i - 1][j - 1] > NEGATIVE_INFINITY:
                        candidates.append(
                            (dp_d[i - 1][j - 1] + tp(DELETE, MATCH), DELETE)
                        )

                    if len(candidates) > 0:
                        score, prev_state = max(candidates, key=lambda x: x[0])
                        best_emission_score = score + self.match_emit_probs[i - 1].log(
                            sequence[j - 1]
                        )

                        if best_emission_score > dp_m[i][j]:
                            dp_m[i][j] = best_emission_score
                            trace_m[i][j] = (prev_state, i - 1)

                # DELETE[i, j] -- do not emit, j does not incremenet
                if i > 0:
                    candidates: list[tuple[float, State]] = []

                    if i == 1 and j == 0:  # DELETE Initial Condition
                        candidates.append((tp(BEGIN, DELETE), BEGIN))

                    if dp_m[i - 1][j] > NEGATIVE_INFINITY:
                        candidates.append((dp_m[i - 1][j] + tp(MATCH, DELETE), MATCH))
                    if dp_i[i - 1][j] > NEGATIVE_INFINITY:
                        candidates.append((dp_i[i - 1][j] + tp(INSERT, DELETE), INSERT))
                    if dp_d[i - 1][j] > NEGATIVE_INFINITY:
                        candidates.append((dp_d[i - 1][j] + tp(DELETE, DELETE), DELETE))

                    if len(candidates) > 0:
                        score, prev_state = max(candidates, key=lambda x: x[0])

                        if score > dp_d[i][j]:
                            dp_d[i][j] = score
                            trace_d[i][j] = (prev_state, i - 1)

        # Determine the best final state after processing all `n` symbols
        final_candidates: list[tuple[float, State]] = []

        if dp_m[L][n] > NEGATIVE_INFINITY:
            final_candidates.append((dp_m[L][n], MATCH))
        if dp_i[L][n] > NEGATIVE_INFINITY:
            final_candidates.append((dp_i[L][n], INSERT))
        if dp_d[L][n] > NEGATIVE_INFINITY:
            final_candidates.append((dp_d[L][n], DELETE))

        if len(final_candidates) == 0:
            return NEGATIVE_INFINITY, []

        score, node_type = max(final_candidates, key=lambda x: x[0])
        i, j = L, n

        # Construct the path from end to beginning
        path: list[ViterbiNode] = []

        while node_type != BEGIN:
            path.append(ViterbiNode(node_type, profile_column=i, sequence_position=j))

            if node_type == MATCH:
                trace = trace_m[i][j]
                j -= 1
            elif node_type == INSERT:
                trace = trace_i[i][j]
                j -= 1
            elif node_type == DELETE:
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

    # print("\nEmission Probability Distributions per Match")
    # for i, emission in enumerate(profile.match_emit_probs):
    #     print(f"{i + 1}: {emission.display()}")

    seq = datasets.datasetC[2]
    seq2 = seq[:15] + "A" + seq[15:]

    def v(s):
        print(f"\nViterbi Path for {s}")
        score, path = profile.viterbi(s)

        print(f"Score: {score}")
        print(f"Path: {len(path)}")
        for node in path:
            print(
                f"{node.state.name} ({node.profile_column}, {node.sequence_position})"
            )

    v(seq)
    v(seq2)
