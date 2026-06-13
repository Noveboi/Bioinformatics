import dataclasses
from enum import Enum

GAP = "-"


class Move(Enum):
    NONE = 0  # default uninitialized value
    UP = 1
    LEFT = 2
    DIAGONAL = 3


@dataclasses.dataclass(frozen=True)
class AlignmentResult:
    aligned_seq1: str
    aligned_seq2: str
    score: float
    score_matrix: list[list[float]]
    moves: list[Move]


def align(seq1: str, seq2: str, alpha: int) -> AlignmentResult:
    """
    Perform global sequence alignment using the Needleman-Wunsch algorithm.

    Parameters
    --------
    seq1 : str
        The first sequence of length ``m`` that will be aligned
    seq2 : str
        The second sequence of length ``n`` that will be aligned
    alpha : int
        Parameter that defines the penalties for gaps and mismatches.
        The higher the value, the more harsh the penalty is.
    """
    m = len(seq1)
    n = len(seq2)

    match_reward = 1
    gap_penalty = -alpha
    mismatch_penalty = -alpha / 2

    def s(a: str, b: str):
        if a == b:
            return match_reward
        else:
            return mismatch_penalty

    # scoring matrix
    dp: list[list[float]] = [[0] * (n + 1) for _ in range(m + 1)]

    # "best move" matrix. Based on the score matrix (AKA: traceback matrix)
    trace: list[list[Move]] = [[Move.NONE] * (n + 1) for _ in range(m + 1)]

    # Initialize first column boundary conditions: i * α
    for i in range(1, m + 1):
        dp[i][0] = i * gap_penalty
        trace[i][0] = Move.UP  # only way to go if you end up here

    # Initialize first row boundary conditions: j * α
    for j in range(1, n + 1):
        dp[0][j] = j * gap_penalty
        trace[0][j] = Move.LEFT  # only way to go if you end here

    # Main DP loop (Scoring Process)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            match = dp[i - 1][j - 1] + s(seq1[i - 1], seq2[j - 1])
            delete = dp[i - 1][j] + gap_penalty
            insert = dp[i][j - 1] + gap_penalty

            best = max(match, delete, insert)

            dp[i][j] = best

            if best == match:
                trace[i][j] = Move.DIAGONAL
            elif best == delete:
                trace[i][j] = Move.UP
            elif best == insert:
                trace[i][j] = Move.LEFT
            else:
                raise RuntimeError("Impossible branch hit")

    moves: list[Move] = []
    aligned1: list[str] = []
    aligned2: list[str] = []
    i = m
    j = n

    # Alignment Process
    while i > 0 or j > 0:
        move = trace[i][j]
        moves.append(move)

        if move == Move.DIAGONAL:  # (i-1, j-1) ancestor
            aligned1.append(seq1[i - 1])
            aligned2.append(seq2[j - 1])
            i -= 1
            j -= 1

        elif move == Move.UP:  # (i-1, j) ancestor
            aligned1.append(seq1[i - 1])
            aligned2.append(GAP)
            i -= 1

        elif move == Move.LEFT:  # (i, j-1) ancestor
            aligned1.append(GAP)
            aligned2.append(seq2[j - 1])
            j -= 1

        else:
            raise RuntimeError("Impossible branch hit.")

    aligned1.reverse()
    aligned2.reverse()
    moves.reverse()

    return AlignmentResult(
        aligned_seq1="".join(aligned1),
        aligned_seq2="".join(aligned2),
        score=dp[m][n],
        score_matrix=dp,
        moves=moves,
    )


# Small demo program to play with the algorithm
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("seq1", type=str)
    parser.add_argument("seq2", type=str)
    parser.add_argument("id", type=int)

    args = parser.parse_args()

    alpha = 2 - args.id % 2

    print(f"α = {alpha}")
    print(f"A = {args.seq1}")
    print(f"B = {args.seq2}")

    result = align(args.seq1, args.seq2, alpha)

    print("\nAlignment\n--------")
    print(result.aligned_seq1)
    print(result.aligned_seq2)
