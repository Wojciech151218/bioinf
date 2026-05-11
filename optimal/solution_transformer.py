"""Turn a TSP visit order (kmer indices) into ordered :class:`dna.Kmer` and a merged sequence."""

from __future__ import annotations

from dna import Dna, Kmer

from .matrix_pipeline import max_suffix_prefix_overlap


def transform(dna: Dna, kmer_indices_in_order: list[int]) -> tuple[list[Kmer], str]:
    """
    Parameters
    ----------
    dna:
        Source DNA model (for kmer objects and ``kmer_length``).
    kmer_indices_in_order:
        Indices into ``dna.kmers`` along the solved tour (may repeat indices).

    Returns
    -------
    Ordered list of :class:`Kmer` instances and the superstring built by greedy
    overlap merge along that order.
    """
    ordered_kmers = [dna.kmers[i] for i in kmer_indices_in_order]
    sequences = [km.sequence for km in ordered_kmers]
    return ordered_kmers, merge_sequences(dna.kmer_length, sequences)


def merge_sequences(kmer_length: int, parts: list[str]) -> str:
    """Concatenate kmer strings by maximal prefix/suffix overlap at each step."""
    if not parts:
        return ""
    merged = parts[0]
    for nxt in parts[1:]:
        o = max_suffix_prefix_overlap(merged, nxt)
        merged = merged + nxt[o:]
    return merged
