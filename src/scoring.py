import math

from common import NEGATIVE_INFINITY
from hmm import BEGIN, DELETE, INSERT, MATCH


def _logsumexp(values: list[float]) -> float:
    values = [v for v in values if v > NEGATIVE_INFINITY]
    if not values:
        return NEGATIVE_INFINITY

    m = max(values)
    return m + math.log(sum(math.exp(v - m) for v in values))


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
                    f_i[i][j] = _logsumexp(prev_scores) + self.insert_emit_probs.log(
                        sequence[j - 1]
                    )

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
