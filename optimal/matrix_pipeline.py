"""Build asymmetric overlap cost matrices from a :class:`dna.Dna` instance.

``cost[i][j] = dna.kmer_length - overlap(kmer_i.suffix, kmer_j.prefix)`` so
going from *i* to *j* pays for characters of *j* that are not overlapped by *i*.

Two matrices are returned: **out_cost** where ``out_cost[i][j]`` is the cost of
edge *i* → *j*, and **in_cost** where ``in_cost[i][j] = out_cost[j][i]`` (the
transpose), i.e. costs from the perspective of incoming edges to *i*.

See OR-Tools CP-SAT ``CpModel.add_circuit`` for modeling a tour on directed arcs:
https://developers.google.com/optimization/reference/python/sat/python/cp_model
"""

from __future__ import annotations

from dna import Dna


def max_suffix_prefix_overlap(left: str, right: str) -> int:
    """Maximum *k* such that ``left[-k:] == right[:k]`` (overlap for *left* then *right*)."""
    max_k = min(len(left), len(right))
    for k in range(max_k, -1, -1):
        if k == 0 or left[-k:] == right[:k]:
            return k
    return 0


def transition_cost(kmer_length: int, left: str, right: str) -> int:
    """Cost to append *right* after *left*: characters of *right* not covered by overlap."""
    return kmer_length - max_suffix_prefix_overlap(left, right)


def build_full_cost_matrices(dna: Dna) -> tuple[list[list[int]], list[list[int]]]:
    """Return ``(out_cost, in_cost)`` as full ``n × n`` integer matrices over ``dna.kmers``."""
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
