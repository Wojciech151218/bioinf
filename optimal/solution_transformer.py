from __future__ import annotations

from dna import Dna, Kmer

from .matrix_pipeline import (
    MISSING_NUCLEOTIDE,
    gap_kmer_index,
    is_gap_kmer_index,
    merge_sequences,
)


def transform(dna: Dna, kmer_indices_in_order: list[int]) -> tuple[list[Kmer], str]:
    gap_ix = gap_kmer_index(dna)
    ordered_kmers: list[Kmer] = []
    sequences: list[str] = []
    for i in kmer_indices_in_order:
        if is_gap_kmer_index(dna, i):
            sequences.append(MISSING_NUCLEOTIDE)
        else:
            km = dna.kmers[i]
            ordered_kmers.append(km)
            sequences.append(km.sequence)
    return ordered_kmers, merge_sequences(dna.kmer_length, sequences)
