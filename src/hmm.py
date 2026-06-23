import math
from dataclasses import dataclass
from enum import Enum

from common import (
    ALPHABET,
    GAP,
    NEGATIVE_INFINITY,
    ProbabilityDistribution,
)


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
        self._probs: dict[State, ProbabilityDistribution[State]] = {
            BEGIN: ProbabilityDistribution.uniform([MATCH, INSERT, DELETE]),
            MATCH: ProbabilityDistribution.uniform([MATCH, INSERT, DELETE]),
            INSERT: ProbabilityDistribution.uniform([MATCH, INSERT, DELETE]),
            DELETE: ProbabilityDistribution.uniform([MATCH, INSERT, DELETE]),
        }

    @classmethod
    def from_counts(
        cls,
        counts: dict[State, dict[State, int]],
    ) -> "TransitionProbabilities":
        probs = {
            s: ProbabilityDistribution.from_counts(c, True) for s, c in counts.items()
        }

        t = cls()
        t._probs = probs
        return t

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


def _logsumexp(values: list[float]) -> float:
    values = [v for v in values if v > NEGATIVE_INFINITY]
    if not values:
        return NEGATIVE_INFINITY

    m = max(values)
    return m + math.log(sum(math.exp(v - m) for v in values))


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
        insert_counts: dict[str, int] = {x: 0 for x in ALPHABET}

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

    def train(self, sequences: list[str]) -> None:
        """
        Train the profile HMM using the best-path method.
        """
        L = self.match_column_count

        transition_counts = {
            s1: {s2: 0 for s2 in [MATCH, INSERT, DELETE]}
            for s1 in [BEGIN, MATCH, INSERT, DELETE]
        }

        match_counts = [{x: 0 for x in ALPHABET} for _ in range(L)]

        insert_counts = {x: 0 for x in ALPHABET}

        for sequence in sequences:
            _, best_path = self.viterbi(sequence)

            prev_state = BEGIN
            j = 0

            for node in best_path:
                curr_state = node.state

                transition_counts[prev_state][curr_state] += 1

                if curr_state == MATCH:
                    match_counts[node.profile_column - 1][sequence[j]] += 1
                    j += 1

                elif curr_state == INSERT:
                    insert_counts[sequence[j]] += 1
                    j += 1

                prev_state = curr_state

        self.transition_probs = TransitionProbabilities.from_counts(transition_counts)
        self.match_emit_probs = [
            ProbabilityDistribution.from_counts(c, True) for c in match_counts
        ]
        self.insert_emit_probs = ProbabilityDistribution.from_counts(
            insert_counts, True
        )

    # The forward algorithm has very similar DP structure to Viterbi (hmm.py).
    def forward(self, sequence: str) -> float:
        """
        Any-path score for a sequence under the profile HMM.
        Returns log P(sequence | HMM).
        """
        L, n = self.match_column_count, len(sequence)
        tp = self.transition_probs.log

        # Forward DP tables in log-space
        f_m = [[NEGATIVE_INFINITY] * (n + 1) for _ in range(L + 1)]
        f_i = [[NEGATIVE_INFINITY] * (n + 1) for _ in range(L + 1)]
        f_d = [[NEGATIVE_INFINITY] * (n + 1) for _ in range(L + 1)]

        for i in range(L + 1):
            for j in range(n + 1):
                # INSERT[i][j]: emits sequence[j-1], stays at same profile column
                if j > 0:
                    prev_scores: list[float] = []

                    if i == 0 and j == 1:
                        prev_scores.append(tp(BEGIN, INSERT))

                    if f_m[i][j - 1] > NEGATIVE_INFINITY:
                        prev_scores.append(f_m[i][j - 1] + tp(MATCH, INSERT))
                    if f_i[i][j - 1] > NEGATIVE_INFINITY:
                        prev_scores.append(f_i[i][j - 1] + tp(INSERT, INSERT))
                    if f_d[i][j - 1] > NEGATIVE_INFINITY:
                        prev_scores.append(f_d[i][j - 1] + tp(DELETE, INSERT))

                    if prev_scores:
                        f_i[i][j] = _logsumexp(
                            prev_scores
                        ) + self.insert_emit_probs.log(sequence[j - 1])

                # MATCH[i][j]: emits sequence[j-1], advances profile column
                if i > 0 and j > 0:
                    prev_scores = []

                    if i == 1 and j == 1:
                        prev_scores.append(tp(BEGIN, MATCH))

                    if f_m[i - 1][j - 1] > NEGATIVE_INFINITY:
                        prev_scores.append(f_m[i - 1][j - 1] + tp(MATCH, MATCH))
                    if f_i[i - 1][j - 1] > NEGATIVE_INFINITY:
                        prev_scores.append(f_i[i - 1][j - 1] + tp(INSERT, MATCH))
                    if f_d[i - 1][j - 1] > NEGATIVE_INFINITY:
                        prev_scores.append(f_d[i - 1][j - 1] + tp(DELETE, MATCH))

                    if prev_scores:
                        f_m[i][j] = _logsumexp(prev_scores) + self.match_emit_probs[
                            i - 1
                        ].log(sequence[j - 1])

                # DELETE[i][j]: emits nothing, advances profile column
                if i > 0:
                    prev_scores = []

                    if i == 1 and j == 0:
                        prev_scores.append(tp(BEGIN, DELETE))

                    if f_m[i - 1][j] > NEGATIVE_INFINITY:
                        prev_scores.append(f_m[i - 1][j] + tp(MATCH, DELETE))
                    if f_i[i - 1][j] > NEGATIVE_INFINITY:
                        prev_scores.append(f_i[i - 1][j] + tp(INSERT, DELETE))
                    if f_d[i - 1][j] > NEGATIVE_INFINITY:
                        prev_scores.append(f_d[i - 1][j] + tp(DELETE, DELETE))

                    if prev_scores:
                        f_d[i][j] = _logsumexp(prev_scores)

        # Any-path score = sum over all valid ending states
        return _logsumexp([f_m[L][n], f_i[L][n], f_d[L][n]])
