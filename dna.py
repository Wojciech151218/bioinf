from dataclasses import dataclass, field


@dataclass
class Kmer:
    sequence: str
    intensity: int
    mapped_occurrences: list[int] = field(default_factory=list)



@
class Dna:
    key: str
    length: int
    start: str
    kmer_length: int
    kmers: list[Kmer] = field(default_factory=list)

    def map_occurrences(self, sequence: str | None = None) -> None:
        sequence_to_scan = sequence if sequence is not None else self.start

        for kmer in self.kmers:
            kmer.mapped_occurrences = find_occurrences(sequence_to_scan, kmer.sequence)


def find_occurrences(sequence: str, kmer: str) -> list[int]:
    if not kmer:
        return []

    return [
        index
        for index in range(len(sequence) - len(kmer) + 1)
        if sequence[index : index + len(kmer)] == kmer
    ]
