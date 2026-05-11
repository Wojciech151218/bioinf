from __future__ import annotations

from dna import Dna


def max_suffix_prefix_overlap(left: str, right: str) -> int:
    max_k = min(len(left), len(right))
    for k in range(max_k, -1, -1):
        if k == 0 or left[-k:] == right[:k]:
            return k
    return 0


def transition_cost(kmer_length: int, left: str, right: str) -> int:
    return kmer_length - max_suffix_prefix_overlap(left, right)


def build_full_cost_matrices(dna: Dna) -> tuple[list[list[int]], list[list[int]]]:
    n = len(dna.kmers)
    k = dna.kmer_length
    out_cost: list[list[int]] = [[0] * n for _ in range(n)]
    for i in range(n):
        seq_i = dna.kmers[i].sequence
        for j in range(n):
            seq_j = dna.kmers[j].sequence
            out_cost[i][j] = transition_cost(k, seq_i, seq_j)
    in_cost = [[out_cost[j][i] for j in range(n)] for i in range(n)]
    return out_cost, in_cost
