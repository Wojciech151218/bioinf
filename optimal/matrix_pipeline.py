from __future__ import annotations

from dna import Dna

MISSING_NUCLEOTIDE = "_"


def gap_kmer_index(dna: Dna) -> int:
    """Matrix row/column index for the optional missing-nucleotide node."""
    return len(dna.kmers)


def is_gap_kmer_index(dna: Dna, ki: int) -> bool:
    return ki == gap_kmer_index(dna)


def node_sequence_length(dna: Dna, ki: int) -> int:
    return 1 if is_gap_kmer_index(dna, ki) else dna.kmer_length


def max_suffix_prefix_overlap(left: str, right: str) -> int:
    max_k = min(len(left), len(right))
    for k in range(max_k, -1, -1):
        if k == 0 or left[-k:] == right[:k]:
            return k
    return 0


def overlap_between(dna: Dna, ki: int, kj: int, seq_i: str, seq_j: str) -> int:
    if is_gap_kmer_index(dna, ki) or is_gap_kmer_index(dna, kj):
        return 0
    return max_suffix_prefix_overlap(seq_i, seq_j)


def transition_cost(dna: Dna, ki: int, kj: int, seq_i: str, seq_j: str) -> int:
    return node_sequence_length(dna, kj) - overlap_between(dna, ki, kj, seq_i, seq_j)


def merge_sequences(kmer_length: int, parts: list[str]) -> str:
    if not parts:
        return ""
    merged = parts[0]
    for nxt in parts[1:]:
        if merged == MISSING_NUCLEOTIDE or nxt == MISSING_NUCLEOTIDE:
            o = 0
        else:
            o = max_suffix_prefix_overlap(merged, nxt)
        merged = merged + nxt[o:]
    return merged


def merged_length_along_path(dna: Dna, parts: list[str]) -> int:
    """Length of merge along a tour, matching the solver length constraint."""
    if not parts:
        return 0
    total = len(parts[0])
    for idx in range(1, len(parts)):
        left, right = parts[idx - 1], parts[idx]
        if left == MISSING_NUCLEOTIDE or right == MISSING_NUCLEOTIDE:
            overlap = 0
        else:
            overlap = max_suffix_prefix_overlap(left, right)
        total += len(right) - overlap
    return total


def build_overlap_matrix(dna: Dna) -> list[list[int]]:
    n = len(dna.kmers)
    size = n + 1
    overlap: list[list[int]] = [[0] * size for _ in range(size)]
    for i in range(n):
        seq_i = dna.kmers[i].sequence
        for j in range(n):
            seq_j = dna.kmers[j].sequence
            overlap[i][j] = overlap_between(dna, i, j, seq_i, seq_j)
    return overlap


def build_full_cost_matrices(dna: Dna) -> tuple[list[list[int]], list[list[int]]]:
    n = len(dna.kmers)
    size = n + 1
    out_cost: list[list[int]] = [[0] * size for _ in range(size)]
    gap_seq = MISSING_NUCLEOTIDE
    for i in range(size):
        seq_i = gap_seq if is_gap_kmer_index(dna, i) else dna.kmers[i].sequence
        for j in range(size):
            seq_j = gap_seq if is_gap_kmer_index(dna, j) else dna.kmers[j].sequence
            out_cost[i][j] = transition_cost(dna, i, j, seq_i, seq_j)
    in_cost = [[out_cost[j][i] for j in range(size)] for i in range(size)]
    return out_cost, in_cost
