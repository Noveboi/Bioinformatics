import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Callable, TypeAlias

from common import (
    ALPHABET,
    GAP,
    NEGATIVE_INFINITY,
    ProbabilityDistribution,
)


class State(str, Enum):
    NONE = "NONE"
    BEGIN = "BEGIN"
    MATCH = "MATCH"
    INSERT = "INSERT"
    DELETE = "DELETE"
    END = "END"


# shorthand syntax for more compact code
# the ``State`` class is kept for strong type inference
BEGIN = State.BEGIN
MATCH = State.MATCH
INSERT = State.INSERT
DELETE = State.DELETE
END = State.END


@dataclass(frozen=True)
class PathNode:
    state: State
    profile_column: int
    sequence_position: int


class TransitionProbabilities:
    def __init__(
        self,
        probabilities: list[dict[State, ProbabilityDistribution[State]]],
    ) -> None:
        self._probs = probabilities

    @classmethod
    def uniform(cls, profile_column_count: int) -> "TransitionProbabilities":
        probs = [
            {
                BEGIN: ProbabilityDistribution.uniform([INSERT, MATCH, DELETE]),
                INSERT: ProbabilityDistribution.uniform([INSERT, MATCH]),
            }
        ]

        for _ in range(1, profile_column_count):
            probs.append(
                {
                    MATCH: ProbabilityDistribution.uniform([MATCH, INSERT, DELETE]),
                    INSERT: ProbabilityDistribution.uniform([INSERT, MATCH]),
                    DELETE: ProbabilityDistribution.uniform([DELETE, MATCH]),
                }
            )

        probs.append(
            {
                MATCH: ProbabilityDistribution.uniform([INSERT, END]),
                INSERT: ProbabilityDistribution.uniform([INSERT, END]),
                DELETE: ProbabilityDistribution.uniform([END]),
            }
        )

        return cls(probs)

    @classmethod
    def from_counts(
        cls,
        counts_per_profile_column: list[dict[State, dict[State, int]]],
    ) -> "TransitionProbabilities":
        probs = [
            {s: ProbabilityDistribution.from_counts(c, True) for s, c in counts.items()}
            for counts in counts_per_profile_column
        ]

        return cls(probs)

    def log(self, profile_column: int, current_state: State, next_state: State):
        return self._probs[profile_column][current_state].log(next_state)


_DPCandidate: TypeAlias = tuple[float, State, int]


class DynamicProgramSolver:
    _STATE_IDX = {
        MATCH: 0,
        INSERT: 1,
        DELETE: 2,
    }  # perf optimization

    # state -> (previous_state, Δ(profile_column), Δ(sequence_position))
    _PREDECESSORS = {
        MATCH: [
            (MATCH, -1, -1),
            (INSERT, -1, -1),
            (DELETE, -1, -1),
        ],
        INSERT: [
            (MATCH, 0, -1),
            (INSERT, 0, -1),
        ],
        DELETE: [
            (MATCH, -1, 0),
            (DELETE, -1, 0),
        ],
    }

    _STATES = [MATCH, INSERT, DELETE]

    def __init__(
        self,
        profile_column_count: int,
        combine: Callable[[Sequence[_DPCandidate]], _DPCandidate],
        match_emission_probs: list[ProbabilityDistribution[str]],
        insert_emission_probs: ProbabilityDistribution[str],
        transition_probabilities: TransitionProbabilities,
    ) -> None:
        self._profile_column_count = profile_column_count
        self._combine = combine
        self._transition_probs = transition_probabilities

        def prob(state: State, profile_column: int, symbol: str) -> float:
            match state:
                case State.MATCH:
                    return match_emission_probs[profile_column].log(symbol)
                case State.INSERT:
                    return insert_emission_probs.log(symbol)
                case _:
                    return 0

        self._probs = prob

    def solve(
        self,
        observations: str,
        trace_path: bool = True,
    ) -> tuple[float, Sequence[PathNode]]:
        L, n = self._profile_column_count, len(observations)
        tp = self._transition_probs.log

        dp = [[[NEGATIVE_INFINITY] * 3 for _ in range(n + 1)] for _ in range(L + 1)]
        trace = (
            [[[(State.NONE, -1)] * 3 for _ in range(n + 1)] for _ in range(L + 1)]
            if trace_path
            else None
        )

        for i in range(L + 1):
            for j in range(n + 1):
                for state in self._STATES:
                    # Structural validity of profile-HMM states.
                    if state == MATCH and (i == 0 or j == 0):
                        continue

                    if state == INSERT and j == 0:
                        continue

                    if state == DELETE and i == 0:
                        continue

                    idx = self._STATE_IDX[state]
                    candidates: list[_DPCandidate] = []
                    begin_added = False

                    # use only the valid incoming edges
                    for prev_state, di, dj in self._PREDECESSORS[state]:
                        # p = previous profile column
                        # s = previous sequence position
                        p, s = i + di, j + dj

                        if p < 0 or s < 0:
                            continue

                        if p == 0 and s == 0:
                            if not begin_added:
                                candidates.append((tp(0, BEGIN, state), BEGIN, 0))
                                begin_added = True
                            continue

                        prev_idx = self._STATE_IDX[prev_state]
                        score = dp[p][s][prev_idx]

                        if score <= NEGATIVE_INFINITY:
                            continue

                        candidates.append(
                            (
                                score + tp(p, prev_state, state),
                                prev_state,
                                p,
                            )
                        )

                    if len(candidates) == 0:
                        continue

                    best_score, best_state, p = self._combine(candidates)
                    score = best_score + self._probs(state, i - 1, observations[j - 1])

                    if score > dp[i][j][idx]:
                        dp[i][j][idx] = score

                        if trace is not None:
                            trace[i][j][idx] = best_state, p

        final_candidates: list[_DPCandidate] = [
            (dp[L][n][self._STATE_IDX[MATCH]] + tp(L, MATCH, END), MATCH, L),
            (dp[L][n][self._STATE_IDX[INSERT]] + tp(L, INSERT, END), INSERT, L),
            (dp[L][n][self._STATE_IDX[DELETE]] + tp(L, DELETE, END), DELETE, L),
        ]

        best_final_score, current_state, _ = self._combine(final_candidates)

        if trace is None:
            return best_final_score, []

        p, s = L, n

        path: list[PathNode] = [PathNode(END, L + 1, s)]

        while current_state != BEGIN:
            path.append(
                PathNode(
                    current_state,
                    profile_column=p,
                    sequence_position=s,
                )
            )

            idx = self._STATE_IDX[current_state]

            if current_state == MATCH:
                t = trace[p][s][idx]
                s -= 1
            elif current_state == INSERT:
                t = trace[p][s][idx]
                s -= 1
            elif current_state == DELETE:
                t = trace[p][s][idx]
            else:
                raise RuntimeError()

            current_state, p = t

        path.reverse()

        return best_final_score, path


def _get_match_columns(msa: list[str], threshold: float) -> list[int]:
    """
    Returns the indices of 'match' columns in the MSA.

    A column is a 'match' column if the fraction of gap symbols is
    below the given ``threshold``. That means a lower ``threshold`` allows more
    'match' columns to be identified, a lower ``threshold`` is stricter.
    """
    n_cols = len(msa[0])
    n_rows = len(msa)

    return [
        j
        for j in range(n_cols)
        if sum(seq[j] == GAP for seq in msa) / n_rows < threshold
    ]


def _logsumexp(candidates: Sequence[_DPCandidate]) -> _DPCandidate:
    candidates = [v for v in candidates if v[0] > NEGATIVE_INFINITY]

    m = max(candidates, key=lambda x: x[0])

    return m[0] + math.log(sum(math.exp(v[0] - m[0]) for v in candidates)), m[1], m[2]


def _empty_transition_counts(
    profile_column_count: int,
) -> list[dict[State, dict[State, int]]]:
    """
    Return zero-filled transition-count dictionaries for the same topology used by the
    dynamic program.

    The list index is the source profile column. For example, index 0 contains
    BEGIN and I_0 transitions, index i contains transitions leaving M_i, I_i and D_i,
    and index L contains transitions into END.
    """
    L = profile_column_count

    counts: list[dict[State, dict[State, int]]] = [
        {
            BEGIN: {INSERT: 0, MATCH: 0, DELETE: 0},
            INSERT: {INSERT: 0, MATCH: 0, DELETE: 0},
        }
    ]

    for _ in range(1, L):
        counts.append(
            {
                MATCH: {MATCH: 0, INSERT: 0, DELETE: 0},
                INSERT: {INSERT: 0, MATCH: 0, DELETE: 0},
                DELETE: {DELETE: 0, MATCH: 0, INSERT: 0},
            }
        )

    counts.append(
        {
            MATCH: {INSERT: 0, END: 0},
            INSERT: {INSERT: 0, END: 0},
            DELETE: {INSERT: 0, END: 0},
        }
    )

    return counts


def _transition_counts_from_msa(
    msa: list[str],
    match_columns: list[int],
) -> list[dict[State, dict[State, int]]]:
    """
    Convert every aligned MSA row into a profile-HMM state path and count transitions.

    Match columns generate either M_i or D_i, depending on whether the row has a
    residue or a gap at that column. Non-match columns generate I_i only when the row
    has a residue there; gaps in insertion columns do not correspond to an HMM state.
    """
    L = len(match_columns)
    match_columns_set = set(match_columns)
    counts = _empty_transition_counts(L)

    for sequence in msa:
        prev_state = BEGIN
        prev_profile_col = 0
        profile_col = 0

        for msa_col, symbol in enumerate(sequence):
            if msa_col in match_columns_set:
                profile_col += 1
                curr_state = DELETE if symbol == GAP else MATCH
                curr_profile_col = profile_col
            else:
                if symbol == GAP:
                    continue

                curr_state = INSERT
                curr_profile_col = profile_col

            counts[prev_profile_col][prev_state][curr_state] += 1
            prev_state = curr_state
            prev_profile_col = curr_profile_col

        counts[prev_profile_col][prev_state][END] += 1

    return counts


class ProfileHMM:
    """
    A profile hidden Markov model consisting of:

        - BEGIN -> entry state
        - MATCH -> consensus columns
        - INSERT -> insertion regions
        - DELETE -> skipped consensus columns
        - END -> terminal state

    MATCH and INSERT emit symbols.
    DELETE, BEGIN and END are silent states.

    The purpose of the profile HMM is to emit sequences (A, T, G, C) by first predicting the
    overall structure of a given set of related sequences (given by MSA)

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
        L, n = len(match_columns), len(msa[0])

        if L == 0:  # ! if you hit this, lower the threshold
            raise ValueError(
                f"No match columns detected from MSA. (threshold={THRESHOLD}"
            )

        if len(set(len(s) for s in msa)) > 1:
            raise ValueError("All MSA sequences are expected to be of equal length.")

        valid_symbols = set(ALPHABET).union("-")
        if len(set(chr for s in msa for chr in s) - valid_symbols):
            raise ValueError(
                f"Expected all MSA sequences to only contain symbols: {valid_symbols}"
            )

        match_counts = [{x: 0 for x in ALPHABET} for _ in range(L)]
        insert_counts = {x: 0 for x in ALPHABET}

        for sequence in msa:
            for j, col in enumerate(match_columns):
                symbol = sequence[col]

                if symbol != GAP:
                    match_counts[j][symbol] += 1

        for sequence in msa:
            for col in range(n):
                symbol = sequence[col]
                if col not in match_columns_set and symbol != GAP:
                    insert_counts[symbol] += 1

        self.profile_column_count = L
        self.transition_probs = TransitionProbabilities.from_counts(
            _transition_counts_from_msa(msa, match_columns)
        )

        self.match_emit_probs = [
            ProbabilityDistribution.from_counts(c, True) for c in match_counts
        ]
        self.insert_emit_probs = ProbabilityDistribution.from_counts(
            insert_counts, True
        )

    def viterbi(self, sequence: str) -> tuple[float, Sequence[PathNode]]:
        """
        Finds the most likely state path for a sequence
        """
        solver = DynamicProgramSolver(
            profile_column_count=self.profile_column_count,
            combine=lambda x: max(x, key=lambda y: y[0]),
            transition_probabilities=self.transition_probs,
            match_emission_probs=self.match_emit_probs,
            insert_emission_probs=self.insert_emit_probs,
        )

        return solver.solve(sequence, trace_path=True)

    # The forward algorithm has very similar DP structure to Viterbi.
    def forward(self, sequence: str) -> float:
        """
        Any-path score for a sequence under the profile HMM.
        Returns log P(sequence | HMM).
        """
        solver = DynamicProgramSolver(
            profile_column_count=self.profile_column_count,
            combine=_logsumexp,
            match_emission_probs=self.match_emit_probs,
            insert_emission_probs=self.insert_emit_probs,
            transition_probabilities=self.transition_probs,
        )

        return solver.solve(sequence, trace_path=False)[0]

    def train(self, sequences: list[str]) -> None:
        """
        Train the profile HMM using the best-path (Viterbi) method.
        """
        L = self.profile_column_count

        transition_counts = _empty_transition_counts(L)

        match_counts = [{x: 0 for x in ALPHABET} for _ in range(L)]

        insert_counts = {x: 0 for x in ALPHABET}

        for sequence in sequences:
            _, best_path = self.viterbi(sequence)

            prev_state = BEGIN
            prev_profile_col = 0
            seq_pos = 0

            for node in best_path:
                curr_state = node.state

                transition_counts[prev_profile_col][prev_state][curr_state] += 1

                if curr_state == END:
                    break

                if curr_state == MATCH:
                    match_counts[node.profile_column - 1][sequence[seq_pos]] += 1
                    seq_pos += 1

                elif curr_state == INSERT:
                    insert_counts[sequence[seq_pos]] += 1
                    seq_pos += 1

                prev_profile_col = node.profile_column
                prev_state = curr_state

        self.transition_probs = TransitionProbabilities.from_counts(transition_counts)
        self.match_emit_probs = [
            ProbabilityDistribution.from_counts(c, True) for c in match_counts
        ]
        self.insert_emit_probs = ProbabilityDistribution.from_counts(
            insert_counts, True
        )
